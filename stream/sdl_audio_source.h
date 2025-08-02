// -------------------------------------------------------------------------------------------------
//
// Copyright (C) all of the contributors. All rights reserved.
//
// This software, including documentation, is protected by copyright controlled by
// contributors. All rights are reserved. Copying, including reproducing, storing,
// adapting or translating, any or all of this material requires the prior written
// consent of all contributors.
//
// -------------------------------------------------------------------------------------------------

#pragma once

#include "common-sdl.h"
#include "whisper.h"
#include "audio_source.h"
#include <vector>
#include <memory>
#include <atomic>

/**
 * @class sdl_audio_source
 * @brief SDL2-based microphone audio source implementation
 *
 * This class manages audio input from microphones using SDL2, providing:
 * - Continuous audio capture from system audio devices
 * - Configurable buffer sizes and processing windows
 * - Overlap handling for context preservation
 * - Automatic sample rate configuration
 *
 * The audio is captured in fixed-time segments with configurable overlap
 * to ensure smooth processing and context preservation between chunks.
 *
 * @par Thread Safety:
 * This class is not thread-safe. External synchronization is required if
 * accessed from multiple threads.
 *
 * @par Usage Example:
 * @code
 * sdl_audio_source::config cfg;
 * cfg.step_ms = 3000;  // 3-second chunks
 *
 * auto source = std::make_unique<sdl_audio_source>(cfg);
 * if (source->initialize() && source->start()) {
 *     std::vector<float> samples;
 *     while (source->get_audio_samples(samples)) {
 *         // Process audio samples...
 *     }
 * }
 * @endcode
 */
class sdl_audio_source : public audio_source {
public:
    /// Default step size in milliseconds for audio processing
    static constexpr int DEFAULT_STEP_MS = 3000;

    /// Default total length of audio buffer in milliseconds
    static constexpr int DEFAULT_LENGTH_MS = 10000;

    /// Default amount of audio to keep between processing steps (for context)
    static constexpr int DEFAULT_KEEP_MS = 50;

    /// Sleep duration when waiting for audio data
    static constexpr int AUDIO_WAIT_SLEEP_MS = 1;

    /// Multiplier for detecting audio processing overload
    static constexpr int OVERLOAD_MULTIPLIER = 2;

    /// Duration for 30-second audio buffer (for memory pre-allocation)
    static constexpr double BUFFER_30S_DURATION = 30000.0;

    /// Milliseconds to seconds conversion factor
    static constexpr double MS_TO_SECONDS = 1e-3;

    /**
     * @struct config
     * @brief Configuration parameters for SDL audio capture
     */
    struct config {
        /// Time step between processing windows (milliseconds)
        int step_ms;

        /// Total length of audio buffer (milliseconds)
        int length_ms;

        /// Amount of previous audio to retain for context (milliseconds)
        int keep_ms;

        /// Audio sample rate (Hz)
        int sample_rate;

        /**
         * @brief Default constructor with optimal settings
         *
         * Initializes configuration with values optimized for real-time
         * speech recognition with minimal latency and good accuracy.
         */
        config()
            : step_ms(DEFAULT_STEP_MS)
            , length_ms(DEFAULT_LENGTH_MS)
            , keep_ms(DEFAULT_KEEP_MS)
            , sample_rate(WHISPER_SAMPLE_RATE) {}
    };

    /**
     * @brief Constructs SDL audio source with specified configuration
     * @param cfg Configuration parameters for audio processing
     *
     * Pre-allocates audio buffers based on configuration to avoid
     * runtime memory allocations during processing.
     */
    explicit sdl_audio_source(const config& cfg = config());

    /**
     * @brief Destructor - ensures audio resources are properly released
     */
    ~sdl_audio_source() override;

    //
    // audio_source interface implementation
    //

    /**
     * @brief Initializes audio capture system
     * @return true if initialization successful, false otherwise
     */
    bool initialize() override { return initialize(-1); }

    /**
     * @brief Starts audio capture
     * @return true if started successfully, false otherwise
     */
    bool start() override;

    /**
     * @brief Stops audio capture
     */
    void stop() override;

    /**
     * @brief Retrieves processed audio samples
     * @param[out] samples Vector to store the retrieved audio samples
     * @return true if samples were retrieved, false if no samples available
     *
     * Returns audio samples in chunks with overlap for context preservation.
     * The chunk size is determined by the step_ms configuration parameter.
     */
    bool get_audio_samples(std::vector<float>& samples) override;

    /**
     * @brief Gets the name/identifier of this audio source
     * @return String identifier for this audio source
     */
    std::string get_name() const override { return "SDL Microphone"; }

    /**
     * @brief Checks if the audio source is active
     * @return true if active and providing samples, false otherwise
     */
    bool is_active() const override { return m_is_active; }

    //
    // SDL-specific interface
    //

    /**
     * @brief Initializes audio capture system with specific device
     * @param device_id Audio device ID (-1 for default device)
     * @return true if initialization successful, false otherwise
     *
     * Sets up the audio capture system with the specified device.
     * The audio system will be configured to use the sample rate
     * and buffer sizes specified in the configuration.
     */
    bool initialize(int device_id);

    /**
     * @brief Gets the current configuration
     * @return Reference to the current configuration object
     */
    const config& get_config() const { return m_config; }

    /**
     * @brief Gets the number of available audio devices
     * @return Number of audio input devices available
     */
    static int get_device_count();

    /**
     * @brief Gets the name of a specific audio device
     * @param device_id Device ID to query
     * @return Device name, or empty string if invalid ID
     */
    static std::string get_device_name(int device_id);

private:
    /// Audio capture configuration
    config m_config;

    /// Audio capture system interface
    std::unique_ptr<audio_async> m_audio;

    /// Main processing buffer for current audio segment
    std::vector<float> m_pcmf32;

    /// Buffer for newly captured audio samples
    std::vector<float> m_pcmf32_new;

    /// Buffer retaining previous audio for context
    std::vector<float> m_pcmf32_old;

    /// Number of samples in 30-second buffer (for memory allocation)
    int m_n_samples_30s;

    /// Number of samples in processing length window
    int m_n_samples_len;

    /// Number of samples in processing step
    int m_n_samples_step;

    /// Number of samples to keep for context between steps
    int m_n_samples_keep;

    /// Flag indicating if audio capture is active
    std::atomic<bool> m_is_active{false};

    /**
     * @brief Processes audio with sliding window and overlap
     *
     * Captures audio in fixed-size chunks with overlap to ensure
     * no speech is missed at segment boundaries. This method handles
     * the buffering and overlap management for smooth audio processing.
     */
    void process_audio();
};
