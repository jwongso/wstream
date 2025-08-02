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
#include <cmath>

/**
 * @file audio_processor.h
 * @brief Audio capture and preprocessing for real-time speech recognition
 * @author WStream Development Team
 * @version 2.0
 * @date 2024
 */

/**
 * @class sdl_audio_source
 * @brief Handles real-time audio capture and preprocessing for speech recognition
 *
 * This class manages audio input from microphones or other audio devices,
 * performing necessary preprocessing such as:
 * - Continuous audio buffering
 * - Sample rate conversion
 * - Voice Activity Detection (VAD) when enabled
 * - Audio segmentation for optimal recognition
 *
 * The processor supports both continuous processing and VAD-based processing modes.
 * In continuous mode, audio is processed in fixed-time segments with overlap.
 * In VAD mode, processing is triggered only when speech is detected.
 *
 * Implements the audio_source interface for seamless integration with the
 * audio source switching system.
 *
 * @par Thread Safety:
 * This class is not thread-safe. External synchronization is required if
 * accessed from multiple threads.
 */
class sdl_audio_source : public audio_source {
public:
    /// Default step size in milliseconds for continuous processing
    static constexpr int DEFAULT_STEP_MS = 3000;

    /// Default total length of audio buffer in milliseconds
    static constexpr int DEFAULT_LENGTH_MS = 10000;

    /// Default amount of audio to keep between processing steps (for context)
    static constexpr int DEFAULT_KEEP_MS = 200;

    /// Default VAD mode setting
    static constexpr bool DEFAULT_USE_VAD = false;

    /// VAD processing interval in milliseconds
    static constexpr int VAD_PROCESS_INTERVAL_MS = 2000;

    /// VAD detection length in milliseconds
    static constexpr int VAD_DETECTION_LENGTH_MS = 1000;

    /// VAD energy threshold (higher = more sensitive)
    static constexpr float VAD_ENERGY_THRESHOLD = 0.65f;

    /// VAD frequency threshold in Hz
    static constexpr float VAD_FREQ_THRESHOLD = 150.0f;

    /// Sleep duration when waiting for audio data
    static constexpr int AUDIO_WAIT_SLEEP_MS = 1;

    /// Sleep duration during VAD processing
    static constexpr int VAD_SLEEP_MS = 100;

    /// Multiplier for detecting audio processing overload
    static constexpr int OVERLOAD_MULTIPLIER = 2;

    /// Duration for 30-second audio buffer (for memory pre-allocation)
    static constexpr double BUFFER_30S_DURATION = 30000.0;

    /// Milliseconds to seconds conversion factor
    static constexpr double MS_TO_SECONDS = 1e-3;

    /**
     * @struct config
     * @brief Configuration parameters for audio processing
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

        /// Enable Voice Activity Detection
        bool use_vad;

        /// Force VAD detection (always detect speech in non-silent sections)
        bool force_vad_detection;

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
            , sample_rate(WHISPER_SAMPLE_RATE)
            , use_vad(DEFAULT_USE_VAD)
            , force_vad_detection(false) {}
    };

    /**
     * @brief Constructs audio processor with specified configuration
     * @param cfg Configuration parameters for audio processing
     *
     * Pre-allocates audio buffers based on configuration to avoid
     * runtime memory allocations during processing.
     */
    explicit sdl_audio_source(const config& cfg = config{});

    /**
     * @brief Destructor - ensures audio resources are properly released
     */
    ~sdl_audio_source() override;

    //
    // audio_source interface implementation
    //

    /**
     * @brief Initializes audio capture system (audio_source interface)
     * @return true if initialization successful, false otherwise
     */
    bool initialize() override { return initialize(-1); }

    /**
     * @brief Starts audio capture (audio_source interface)
     * @return true if started successfully, false otherwise
     */
    bool start() override { resume(); return true; }

    /**
     * @brief Stops audio capture (audio_source interface)
     */
    void stop() override { pause(); }

    /**
     * @brief Retrieves processed audio samples (audio_source interface)
     * @param[out] samples Vector to store the retrieved audio samples
     * @return true if samples were retrieved, false if no samples available
     */
    bool get_audio_samples(std::vector<float>& samples) override {
        return get_processed_samples(samples);
    }

    /**
     * @brief Gets the name/identifier of this audio source (audio_source interface)
     * @return String identifier for this audio source
     */
    std::string get_name() const override { return "SDL Microphone"; }

    /**
     * @brief Checks if the audio source is active (audio_source interface)
     * @return true if active and providing samples, false otherwise
     */
    bool is_active() const override { return m_is_active; }

    //
    // Original sdl_audio_source interface
    //

    /**
     * @brief Initializes audio capture system
     * @param device_id Audio device ID (-1 for default device)
     * @return true if initialization successful, false otherwise
     *
     * Sets up the audio capture system with the specified device.
     * The audio system will be configured to use the sample rate
     * and buffer sizes specified in the configuration.
     */
    bool initialize(int device_id);

    /**
     * @brief Pauses audio capture
     *
     * Temporarily stops audio capture while maintaining the audio
     * system state. Can be resumed with resume().
     */
    void pause();

    /**
     * @brief Resumes audio capture
     *
     * Restarts audio capture after it was paused with pause().
     */
    void resume();

    /**
     * @brief Retrieves processed audio samples ready for recognition
     * @param samples Output vector to receive audio samples
     * @return true if samples are available, false if no data ready
     *
     * This method processes raw audio input according to the configured
     * mode (continuous or VAD) and returns samples when ready for
     * speech recognition. The returned samples are normalized and
     * formatted for optimal Whisper processing.
     *
     * @par Processing Modes:
     * - **Continuous Mode**: Returns samples at regular intervals
     * - **VAD Mode**: Returns samples only when speech is detected
     */
    bool get_processed_samples(std::vector<float>& samples);

    /**
     * @brief Gets the current configuration
     * @return Reference to the current configuration object
     */
    const config& get_config() const { return m_config; }

private:
    /// Audio processing configuration
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

    /// Last VAD check timestamp
    std::chrono::steady_clock::time_point m_last_vad_time;

    /**
     * @brief Processes audio in continuous mode (fixed time intervals)
     *
     * Continuously captures audio in fixed-size chunks with overlap
     * to ensure no speech is missed at segment boundaries.
     */
    void process_non_vad();

    /**
     * @brief Processes audio using Voice Activity Detection
     *
     * Only processes audio when speech activity is detected,
     * reducing computational load during silence periods.
     */
    void process_vad();

    /**
     * @brief Apply high-pass filter to audio data
     * @param data Audio samples to filter (modified in place)
     * @param cutoff Cutoff frequency in Hz
     * @param sample_rate Sample rate in Hz
     */
    void high_pass_filter(std::vector<float>& data, float cutoff, float sample_rate);

    /**
     * @brief Simple VAD detection using energy comparison
     * @param pcmf32 Audio samples to analyze
     * @param sample_rate Sample rate in Hz
     * @param last_ms Duration of recent audio to check (milliseconds)
     * @param vad_thold Energy threshold (0.0 to 1.0)
     * @param freq_thold High-pass filter frequency threshold
     * @param verbose Enable debug output
     * @return true if speech detected, false otherwise
     */
    bool vad_simple(std::vector<float>& pcmf32, int sample_rate, int last_ms,
                    float vad_thold, float freq_thold, bool verbose);
};
