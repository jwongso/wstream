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
#include <vector>
#include <memory>

/**
 * @file audio_processor.h
 * @brief Audio capture and preprocessing for real-time speech recognition
 * @author WStream Development Team
 * @version 1.0
 * @date 2024
 */

/**
 * @class audio_processor
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
 * @par Thread Safety:
 * This class is not thread-safe. External synchronization is required if
 * accessed from multiple threads.
 */
class audio_processor {
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
    static constexpr float VAD_ENERGY_THRESHOLD = 0.85f;

    /// VAD frequency threshold in Hz
    static constexpr float VAD_FREQ_THRESHOLD = 100.0f;

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
            , use_vad(DEFAULT_USE_VAD) {}
    };

    /**
     * @brief Constructs audio processor with specified configuration
     * @param cfg Configuration parameters for audio processing
     *
     * Pre-allocates audio buffers based on configuration to avoid
     * runtime memory allocations during processing.
     */
    explicit audio_processor(const config& cfg = config{});

    /**
     * @brief Destructor - ensures audio resources are properly released
     */
    ~audio_processor();

    /**
     * @brief Initializes audio capture system
     * @param device_id Audio device ID (-1 for default device)
     * @return true if initialization successful, false otherwise
     *
     * Sets up the audio capture system with the specified device.
     * The audio system will be configured to use the sample rate
     * and buffer sizes specified in the configuration.
     */
    bool initialize(int device_id = -1);

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
};
