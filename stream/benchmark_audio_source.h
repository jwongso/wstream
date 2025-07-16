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

#include "audio_source.h"
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <atomic>
#include <functional>

/**
 * @class benchmark_audio_source
 * @brief Audio source implementation for benchmarking with pre-recorded WAV files
 *
 * @details This class provides a specialized audio source that reads PCM audio data
 * from WAV files and delivers it in controlled chunks to simulate real-time audio
 * processing. It closely mimics the buffer management behavior of audio_processor
 * for accurate benchmarking.
 *
 * Features:
 * - WAV file loading with format validation
 * - Sliding window processing with overlap (like audio_processor)
 * - Configurable step and overlap sizes
 * - Real-time playback simulation
 * - VAD simulation mode
 * - Audio looping for extended tests
 * - Automatic stereo to mono conversion
 * - Reference text loading for accuracy measurements
 * - Detailed performance statistics
 *
 * @par Thread Safety:
 * This class is thread-safe for concurrent access to statistics while
 * audio is being processed.
 *
 * @par Usage Example:
 * @code
 * benchmark_audio_source::config cfg;
 * cfg.wav_file_path = "test.wav";
 * cfg.use_vad = true;
 *
 * benchmark_audio_source source(cfg);
 * if (source.initialize() && source.start()) {
 *     std::vector<float> samples;
 *     while (source.get_audio_samples(samples)) {
 *         // Process samples...
 *     }
 * }
 * @endcode
 *
 * @see audio_source
 * @see audio_processor
 * @see benchmark_manager
 */
class benchmark_audio_source : public audio_source {
public:
    /**
     * @struct config
     * @brief Configuration parameters for benchmark audio source
     *
     * @details Controls various aspects of benchmark audio playback including
     * file paths, chunk sizes, and simulation modes. Parameters are designed
     * to match audio_processor behavior for realistic benchmarking.
     */
    struct config {
        /**
         * @brief Path to the WAV audio file to be used for benchmarking
         * @details Must be a valid 16-bit PCM WAV file
         */
        std::string wav_file_path;

        /**
         * @brief Path to the reference transcription text file
         * @details Used for accuracy metrics calculation (WER/CER)
         */
        std::string reference_text_path;

        /**
         * @brief Expected sample rate of the audio (Hz)
         * @details Common values: 16000, 22050, 44100, 48000
         */
        int sample_rate;

        /**
         * @brief Expected number of audio channels
         * @details 1 for mono, 2 for stereo. Stereo is automatically converted to mono
         */
        int channels;

        /**
         * @brief Time step between processing windows (milliseconds)
         * @details Matches audio_processor::step_ms for realistic simulation.
         * Controls how much audio is processed in each chunk.
         */
        int step_ms;

        /**
         * @brief Total length of audio buffer (milliseconds)
         * @details Matches audio_processor::length_ms. Maximum amount of audio
         * that can be buffered at once.
         */
        int length_ms;

        /**
         * @brief Amount of previous audio to retain for context (milliseconds)
         * @details Matches audio_processor::keep_ms. Creates overlap between
         * consecutive chunks for better accuracy at boundaries.
         */
        int keep_ms;

        /**
         * @brief Whether to simulate real-time audio playback speed
         * @details When true, audio is delivered at real-time rate.
         * When false, audio is delivered as fast as possible for benchmarking.
         */
        bool real_time_simulation;

        /**
         * @brief Whether to loop the audio file for extended testing
         * @details Useful for stress testing and long-duration benchmarks.
         * Audio restarts from beginning when end is reached.
         */
        bool loop_audio;

        /**
         * @brief Enable Voice Activity Detection simulation
         * @details When true, simulates VAD behavior like audio_processor.
         * Only processes audio when speech is detected.
         */
        bool use_vad;

        /**
         * @brief Maximum allowed audio file size in bytes
         * @details Prevents excessive memory usage from large files.
         * Default: 1GB
         */
        size_t max_file_size;

        /**
         * @brief Audio format validation strictness
         * @details When true, format mismatches cause initialization failure.
         * When false, warnings are issued but processing continues.
         */
        bool strict_format_validation;

        /**
         * @brief Force VAD detection for benchmark
         * @details When true, always detects speech in non-silent sections.
         * Useful for benchmarking with known speech files.
         */
        bool force_vad_detection;

        /**
         * @brief Default constructor with audio_processor-compatible defaults
         *
         * @details Initializes configuration with values that match
         * audio_processor defaults for accurate benchmarking.
         */
        config()
            : wav_file_path("./benchmark.wav"),
            reference_text_path("./benchmark.txt"),
            sample_rate(16000),
            channels(1),
            step_ms(3000),        // Match audio_processor::DEFAULT_STEP_MS
            length_ms(10000),     // Match audio_processor::DEFAULT_LENGTH_MS
            keep_ms(200),         // Match audio_processor::DEFAULT_KEEP_MS
            real_time_simulation(true),
            loop_audio(false),
            use_vad(false),       // Match audio_processor::DEFAULT_USE_VAD
            max_file_size(1024 * 1024 * 1024), // 1GB
            strict_format_validation(false),
            force_vad_detection(true)
        {
        }
    };

    /**
     * @typedef completion_callback_t
     * @brief Callback function type for completion notification
     */
    using completion_callback_t = std::function<void()>;

    /**
     * @brief Construct a benchmark audio source with specified configuration
     * @param cfg Configuration parameters
     *
     * @details Initializes the benchmark audio source with the provided configuration.
     * Pre-allocates buffers similar to audio_processor for consistent memory behavior.
     * The actual audio file loading occurs during initialize().
     *
     * @see initialize()
     */
    explicit benchmark_audio_source(const config& cfg = config());

    /**
     * @brief Destructor - ensures proper cleanup of resources
     *
     * @details Stops audio processing if active and releases all allocated resources.
     * Ensures all threads are properly terminated before destruction.
     */
    ~benchmark_audio_source() override;

    // audio_source interface implementation

    /**
     * @brief Initialize the benchmark audio source
     * @return true if initialization successful, false otherwise
     *
     * @details Performs the following initialization steps:
     * 1. Loads and validates the WAV audio file
     * 2. Loads the reference transcription text (if available)
     * 3. Validates audio format parameters
     * 4. Pre-allocates buffers for audio data (like audio_processor)
     *
     * @pre Configuration must be set with valid file paths
     * @post Audio data is loaded and ready for processing
     *
     * @note This method must be called before start()
     * @warning May consume significant memory for large audio files
     */
    bool initialize() override;

    /**
     * @brief Start audio playback/streaming
     * @return true if started successfully, false otherwise
     *
     * @details Begins the benchmark audio streaming process. Resets all statistics
     * and positions the playback cursor at the beginning of the audio.
     * Clears overlap buffers for fresh start.
     *
     * @pre initialize() must have been called successfully
     * @post is_active() returns true if successful
     * @post Statistics counters are reset to zero
     */
    bool start() override;

    /**
     * @brief Stop audio playback/streaming
     *
     * @details Stops audio streaming and outputs final statistics to console.
     * Does not reset the audio position, allowing resume from the same position.
     * Safe to call multiple times.
     *
     * @post is_active() returns false
     * @post Final statistics are printed to console
     */
    void stop() override;

    /**
     * @brief Get the next chunk of audio samples with overlap processing
     * @param[out] samples Vector to receive audio samples (cleared before filling)
     * @return true if samples were retrieved, false if no more audio available
     *
     * @details Retrieves the next chunk of audio samples using sliding window
     * processing similar to audio_processor. Maintains overlap between chunks
     * for context preservation. If real_time_simulation is enabled, this method
     * may block to maintain real-time playback rate.
     *
     * In VAD mode, returns samples only when speech is detected.
     * In non-VAD mode, returns samples at regular intervals with overlap.
     *
     * @note The samples vector is cleared before adding new samples
     * @note Thread-safe with respect to other methods
     *
     * @pre start() must have been called
     * @post samples contains normalized float audio data in range [-1.0, 1.0]
     */
    bool get_audio_samples(std::vector<float>& samples) override;

    /**
     * @brief Check if the audio source is actively streaming
     * @return true if active, false otherwise
     *
     * @details Returns true if the audio source has been started and has not
     * yet reached the end of the audio file (unless looping is enabled).
     *
     * @note Thread-safe
     */
    bool is_active() const override;

    /**
     * @brief Get the session identifier for this benchmark
     * @return Session ID string
     *
     * @details Returns a constant session identifier for benchmark runs.
     * Can be overridden by derived classes for custom session management.
     *
     * @note Thread-safe
     */
    std::string get_session_id() const override { return m_session_id; }

    /**
     * @brief Get the name of this audio source
     * @return Name string identifying this audio source type
     *
     * @details Returns a human-readable name for this audio source type.
     * Used for logging and user interface display.
     *
     * @note Thread-safe
     */
    std::string get_name() const override { return "Benchmark Audio"; }

    // Benchmark-specific methods

    /**
     * @brief Load audio data from a WAV file
     * @param file_path Path to the WAV file
     * @return true if loading successful, false otherwise
     *
     * @details Loads PCM audio data from a WAV file with the following requirements:
     * - Must be valid RIFF/WAVE format
     * - Must contain PCM audio (format code 1)
     * - Must be 16-bit samples
     * - File size must not exceed max_file_size
     * - Automatically converts stereo to mono if needed
     *
     * @note Replaces any previously loaded audio data
     * @warning Large files may consume significant memory
     */
    bool load_wav_file(const std::string& file_path);

    /**
     * @brief Load reference transcription text from file
     * @param file_path Path to the text file
     * @return true if loading successful, false otherwise
     *
     * @details Loads reference text for accuracy comparison. The text file should
     * contain the expected transcription of the audio file. Multiple lines are
     * concatenated with spaces.
     *
     * @note Used for WER/CER calculation in benchmark results
     */
    bool load_reference_text(const std::string& file_path);

    /**
     * @brief Get total number of audio samples processed
     * @return Number of samples processed since start()
     *
     * @details Returns the cumulative count of audio samples delivered through
     * get_audio_samples() since the last start() call.
     *
     * @note Thread-safe
     */
    size_t get_total_samples_processed() const { return m_total_samples_processed; }

    /**
     * @brief Get total number of chunks processed
     * @return Number of chunks delivered since start()
     *
     * @details Returns the number of times get_audio_samples() has successfully
     * returned audio data since the last start() call.
     *
     * @note Thread-safe
     */
    size_t get_total_chunks_processed() const { return m_total_chunks_processed; }

    /**
     * @brief Get the elapsed processing time in milliseconds
     * @return Processing duration in milliseconds
     *
     * @details Returns the wall-clock time elapsed since start() was called.
     * Useful for calculating real-time factors.
     *
     * @note Thread-safe
     */
    double get_processing_duration_ms() const;

    /**
     * @brief Get the total duration of loaded audio in milliseconds
     * @return Audio duration in milliseconds
     *
     * @details Calculates the total duration of the loaded audio based on
     * the number of samples and sample rate.
     *
     * @note Thread-safe
     */
    double get_audio_duration_ms() const;

    /**
     * @brief Get the loaded reference transcription text
     * @return Reference text string
     *
     * @details Returns the reference transcription text loaded from file,
     * or an empty string if no reference text was loaded.
     *
     * @note Thread-safe
     */
    std::string get_reference_text() const { return m_reference_text; }

    /**
     * @brief Get the current configuration
     * @return Const reference to the configuration object
     *
     * @details Returns the current configuration settings.
     * Useful for querying runtime parameters.
     *
     * @note Thread-safe
     */
    const config& get_config() const { return m_config; }

    /**
     * @brief Reset the audio source to initial state
     *
     * @details Resets playback position, statistics, and overlap buffers
     * without reloading files. Useful for running multiple benchmark iterations
     * with the same audio.
     *
     * @post Audio position reset to beginning
     * @post Statistics counters cleared
     * @post Overlap buffers cleared
     */
    void reset();

    /**
     * @brief Set the chunk size for audio delivery
     * @param ms Step size in milliseconds (like audio_processor step_ms)
     *
     * @details Changes the step size of audio chunks delivered by get_audio_samples().
     * Valid range is MIN_CHUNK_SIZE_MS to MAX_CHUNK_SIZE_MS.
     * Values outside this range are clamped.
     *
     * @note Changes take effect on the next call to get_audio_samples()
     */
    void set_chunk_size_ms(int ms);

    /**
     * @brief Enable or disable real-time simulation
     * @param enable true to enable, false to disable
     *
     * @details Controls whether audio is delivered at real-time rate or as fast
     * as possible. Real-time simulation is useful for testing realistic scenarios.
     *
     * @note Thread-safe
     */
    void set_real_time_simulation(bool enable) { m_config.real_time_simulation = enable; }

    /**
     * @brief Set custom session identifier
     * @param session_id New session identifier
     *
     * @details Allows setting a custom session ID for tracking purposes.
     * Useful for correlating benchmark results.
     *
     * @note Thread-safe
     */
    void set_session_id(const std::string& session_id) { m_session_id = session_id; }

    /**
     * @brief Set completion callback
     * @param callback Function to call when audio processing completes
     *
     * @details Sets a callback function that will be invoked when the end
     * of the audio file is reached (unless looping is enabled).
     *
     * @note Callback is invoked from the audio processing thread
     */
    void set_completion_callback(completion_callback_t callback) { m_completion_callback = callback; }

    /**
     * @brief Check if end of file was reached
     * @return true if end of file reached, false otherwise
     *
     * @details Returns true if the audio source has processed all available
     * audio data and reached the end of the file.
     *
     * @note Thread-safe
     */
    bool is_end_of_file() const { return m_end_of_file_reported; }

    // Constants matching audio_processor

    /** @brief Minimum allowed chunk size in milliseconds */
    static constexpr int MIN_CHUNK_SIZE_MS = 100;

    /** @brief Maximum allowed chunk size in milliseconds */
    static constexpr int MAX_CHUNK_SIZE_MS = 10000;

    /** @brief Default sample rate for audio processing (Hz) */
    static constexpr int DEFAULT_SAMPLE_RATE = 16000;

    /** @brief Maximum supported audio channels */
    static constexpr int MAX_CHANNELS = 2;

    /** @brief Milliseconds to seconds conversion factor */
    static constexpr double MS_TO_SECONDS = 1e-3;

    /** @brief Duration for 30-second audio buffer (for memory pre-allocation) */
    static constexpr double BUFFER_30S_DURATION = 30000.0;

    // VAD-specific constants

    /** @brief VAD detection window size in milliseconds */
    static constexpr int VAD_DETECTION_WINDOW_MS = 500;

    /** @brief VAD segment size in milliseconds */
    static constexpr int VAD_SEGMENT_SIZE_MS = 3000;

    /** @brief VAD advance size in milliseconds (overlap between segments) */
    static constexpr int VAD_ADVANCE_MS = 1500;

    /** @brief VAD skip size when no speech detected in milliseconds */
    static constexpr int VAD_SKIP_MS = 500;

    /** @brief VAD energy threshold for speech detection */
    static constexpr float VAD_ENERGY_THRESHOLD = 0.0001f;

    /** @brief VAD minimum energy to consider as potential speech */
    static constexpr float VAD_MIN_ENERGY = 0.00001f;

private:
    /** @brief Configuration parameters */
    config m_config;

    /** @brief Flag indicating if audio source is active */
    std::atomic<bool> m_active{false};

    /** @brief Session identifier for tracking */
    std::string m_session_id{"benchmark"};

    // Audio data buffers

    /**
     * @brief Main audio buffer containing loaded samples
     * @details Normalized to [-1.0, 1.0] range
     */
    std::vector<float> m_audio_buffer;

    /**
     * @brief Buffer for newly read audio samples
     * @details Temporary storage for current chunk (like audio_processor::m_pcmf32_new)
     */
    std::vector<float> m_pcmf32_new;

    /**
     * @brief Buffer retaining previous audio for context
     * @details Overlap buffer for sliding window (like audio_processor::m_pcmf32_old)
     */
    std::vector<float> m_pcmf32_old;

    /** @brief Current position in the main audio buffer (sample index) */
    size_t m_current_position{0};

    /** @brief Last processed position for VAD mode duplicate prevention */
    size_t m_last_processed_end{0};

    // Pre-calculated sample counts

    /** @brief Number of samples in 30-second buffer (for pre-allocation) */
    int m_n_samples_30s;

    /** @brief Number of samples in processing step */
    int m_n_samples_step;

    /** @brief Number of samples to keep for context between steps */
    int m_n_samples_keep;

    // Reference text for accuracy comparison

    /** @brief Reference transcription text for WER/CER calculation */
    std::string m_reference_text;

    // Statistics

    /** @brief Total number of samples processed (thread-safe counter) */
    std::atomic<size_t> m_total_samples_processed{0};

    /** @brief Total number of chunks processed (thread-safe counter) */
    std::atomic<size_t> m_total_chunks_processed{0};

    /** @brief Timestamp when processing started */
    std::chrono::steady_clock::time_point m_start_time;

    /** @brief Timestamp of last chunk delivery (for real-time simulation) */
    std::chrono::steady_clock::time_point m_last_chunk_time;

    /** @brief Timestamp for VAD simulation throttling */
    std::chrono::steady_clock::time_point m_last_vad_time;

    /** @brief Flag indicating end of file was reached */
    std::atomic<bool> m_end_of_file_reported{false};

    /** @brief Callback for completion notification */
    completion_callback_t m_completion_callback;

    // Helper methods

    /**
     * @brief Process audio in continuous mode with overlap
     * @param[out] samples Output vector for processed samples
     * @return true if samples were processed, false if no more audio
     *
     * @details Implements sliding window processing with overlap preservation
     * matching the behavior of audio_processor::process_non_vad().
     * Maintains context between chunks using overlap buffer.
     */
    bool process_non_vad(std::vector<float>& samples);

    /**
     * @brief Process audio using Voice Activity Detection simulation
     * @param[out] samples Output vector for processed samples
     * @return true if samples were processed, false if no speech detected
     *
     * @details Simulates VAD behavior matching audio_processor::process_vad().
     * Only returns audio when speech energy exceeds threshold.
     * Prevents duplicate processing of same audio segments.
     */
    bool process_vad(std::vector<float>& samples);

    /**
     * @brief Simulate real-time delay between chunks
     *
     * @details Sleeps to maintain real-time playback rate based on step_ms.
     * Used when real_time_simulation is enabled.
     */
    void simulate_real_time_delay();
};
