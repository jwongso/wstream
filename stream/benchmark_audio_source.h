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
 * processing. It supports various playback modes and provides detailed statistics
 * for performance analysis.
 *
 * Features:
 * - WAV file loading with format validation
 * - Configurable chunk sizes for different latency scenarios
 * - Real-time playback simulation
 * - Audio looping for extended tests
 * - Automatic stereo to mono conversion
 * - Reference text loading for accuracy measurements
 * - Detailed performance statistics
 *
 * @note This class is thread-safe for concurrent access to statistics while
 * audio is being processed.
 *
 * @see audio_source
 * @see benchmark_manager
 */
class benchmark_audio_source : public audio_source {
public:
    /**
 * @struct config
 * @brief Configuration parameters for benchmark audio source
 *
 * @details Controls various aspects of benchmark audio playback including
 * file paths, chunk sizes, and simulation modes.
 */
    struct config {
        /** @brief Path to the WAV audio file to be used for benchmarking */
        std::string wav_file_path;

        /** @brief Path to the reference transcription text file */
        std::string reference_text_path;

        /** @brief Expected sample rate of the audio (Hz) */
        int sample_rate;

        /** @brief Expected number of audio channels (1=mono, 2=stereo) */
        int channels;

        /** @brief Size of audio chunks to process in milliseconds
        *  @details Smaller chunks simulate lower latency but may increase overhead */
        int chunk_size_ms;

        /** @brief Whether to simulate real-time audio playback speed
        *  @details When true, audio is delivered at real-time rate; when false,
        *  audio is delivered as fast as possible */
        bool real_time_simulation;

        /** @brief Whether to loop the audio file for extended testing
        *  @details Useful for stress testing and long-duration benchmarks */
        bool loop_audio;

        /** @brief Maximum allowed audio file size in bytes (default: 1GB)
        *  @details Prevents excessive memory usage from large files */
        size_t max_file_size;

        /** @brief Audio format validation strictness
        *  @details When true, format mismatches cause initialization failure;
        *  when false, warnings are issued but processing continues */
        bool strict_format_validation;

        /**
     * @brief Default constructor with reasonable defaults
     */
        config()
            : wav_file_path("./benchmark.wav"),
            reference_text_path("./benchmark.txt"),
            sample_rate(16000),
            channels(1),
            chunk_size_ms(3000),
            real_time_simulation(true),
            loop_audio(false),
            max_file_size(1024 * 1024 * 1024), // 1GB
            strict_format_validation(false)
        {
        }
    };

    using completion_callback_t = std::function<void()>;

    /**
     * @brief Construct a benchmark audio source with specified configuration
     * @param cfg Configuration parameters
     *
     * @details Initializes the benchmark audio source with the provided configuration.
     * The actual audio file loading occurs during initialize().
     *
     * @see initialize()
     */
    explicit benchmark_audio_source(const config& cfg = config());

    /**
     * @brief Destructor - ensures proper cleanup of resources
     *
     * @details Stops audio processing if active and releases all allocated resources.
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
     * 4. Pre-allocates buffers for audio data
     *
     * @note This method must be called before start()
     *
     * @throws std::runtime_error if strict_format_validation is true and format mismatch occurs
     */
    bool initialize() override;

    /**
     * @brief Start audio playback/streaming
     * @return true if started successfully, false otherwise
     *
     * @details Begins the benchmark audio streaming process. Resets all statistics
     * and positions the playback cursor at the beginning of the audio.
     *
     * @pre initialize() must have been called successfully
     * @post is_active() returns true if successful
     */
    bool start() override;

    /**
     * @brief Stop audio playback/streaming
     *
     * @details Stops audio streaming and outputs final statistics to console.
     * Does not reset the audio position, allowing resume from the same position.
     *
     * @post is_active() returns false
     */
    void stop() override;

    /**
     * @brief Get the next chunk of audio samples
     * @param[out] samples Vector to receive audio samples (cleared before filling)
     * @return true if samples were retrieved, false if no more audio available
     *
     * @details Retrieves the next chunk of audio samples based on chunk_size_ms.
     * The samples are normalized to [-1.0, 1.0] range. If real_time_simulation
     * is enabled, this method may block to maintain real-time playback rate.
     *
     * @note The samples vector is cleared before adding new samples
     * @note Thread-safe with respect to other methods
     *
     * @pre start() must have been called
     */
    bool get_audio_samples(std::vector<float>& samples) override;

    /**
     * @brief Check if the audio source is actively streaming
     * @return true if active, false otherwise
     *
     * @details Returns true if the audio source has been started and has not
     * yet reached the end of the audio file (unless looping is enabled).
     */
    bool is_active() const override;

    /**
     * @brief Get the session identifier for this benchmark
     * @return Session ID string
     *
     * @details Returns a constant session identifier for benchmark runs.
     * Can be overridden by derived classes for custom session management.
     */
    std::string get_session_id() const override { return m_session_id; }

    /**
     * @brief Get the name of this audio source
     * @return Name string identifying this audio source type
     *
     * @details Returns a human-readable name for this audio source type.
     * Used for logging and user interface display.
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
     * - File size must not exceed max_file_size
     * - Automatically converts stereo to mono if needed
     *
     * @note Replaces any previously loaded audio data
     *
     * @throws std::runtime_error if file exceeds max_file_size
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
     */
    bool load_reference_text(const std::string& file_path);

    /**
     * @brief Get total number of audio samples processed
     * @return Number of samples processed since start()
     *
     * @details Returns the cumulative count of audio samples delivered through
     * get_audio_samples() since the last start() call.
     */
    size_t get_total_samples_processed() const { return m_total_samples_processed; }

    /**
     * @brief Get total number of chunks processed
     * @return Number of chunks delivered since start()
     *
     * @details Returns the number of times get_audio_samples() has successfully
     * returned audio data since the last start() call.
     */
    size_t get_total_chunks_processed() const { return m_total_chunks_processed; }

    /**
     * @brief Get the elapsed processing time in milliseconds
     * @return Processing duration in milliseconds
     *
     * @details Returns the wall-clock time elapsed since start() was called.
     * Useful for calculating real-time factors.
     */
    double get_processing_duration_ms() const;

    /**
     * @brief Get the total duration of loaded audio in milliseconds
     * @return Audio duration in milliseconds
     *
     * @details Calculates the total duration of the loaded audio based on
     * the number of samples and sample rate.
     */
    double get_audio_duration_ms() const;

    /**
     * @brief Get the loaded reference transcription text
     * @return Reference text string
     *
     * @details Returns the reference transcription text loaded from file,
     * or an empty string if no reference text was loaded.
     */
    std::string get_reference_text() const { return m_reference_text; }

    /**
     * @brief Reset the audio source to initial state
     *
     * @details Resets playback position and statistics without reloading files.
     * Useful for running multiple benchmark iterations with the same audio.
     *
     * @post Audio position reset to beginning, statistics cleared
     */
    void reset();

    /**
     * @brief Set the chunk size for audio delivery
     * @param ms Chunk size in milliseconds
     *
     * @details Changes the size of audio chunks delivered by get_audio_samples().
     * Valid range is 1-1000ms. Values outside this range are ignored.
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
     */
    void set_real_time_simulation(bool enable) { m_config.real_time_simulation = enable; }

    /**
     * @brief Set custom session identifier
     * @param session_id New session identifier
     *
     * @details Allows setting a custom session ID for tracking purposes.
     */
    void set_session_id(const std::string& session_id) { m_session_id = session_id; }

    void set_completion_callback(completion_callback_t callback) {
        m_completion_callback = callback;
    }

    // Check if end of file was reached
    bool is_end_of_file() const { return m_end_of_file_reported; }

    // Constants

    /** @brief Minimum allowed chunk size in milliseconds */
    static constexpr int MIN_CHUNK_SIZE_MS = 1;

    /** @brief Maximum allowed chunk size in milliseconds */
    static constexpr int MAX_CHUNK_SIZE_MS = 1000;

    /** @brief Default sample rate for audio processing */
    static constexpr int DEFAULT_SAMPLE_RATE = 16000;

    /** @brief Maximum supported audio channels */
    static constexpr int MAX_CHANNELS = 2;

private:
    /** @brief Configuration parameters */
    config m_config;

    /** @brief Flag indicating if audio source is active */
    std::atomic<bool> m_active{false};

    /** @brief Session identifier */
    std::string m_session_id{"benchmark"};

    // Audio data

    /** @brief Buffer containing loaded audio samples (normalized to [-1.0, 1.0]) */
    std::vector<float> m_audio_buffer;

    /** @brief Current position in the audio buffer */
    size_t m_current_position{0};

    // Reference text for accuracy comparison

    /** @brief Reference transcription text */
    std::string m_reference_text;

    // Statistics

    /** @brief Total number of samples processed */
    size_t m_total_samples_processed{0};

    /** @brief Total number of chunks processed */
    size_t m_total_chunks_processed{0};

    /** @brief Timestamp when processing started */
    std::chrono::steady_clock::time_point m_start_time;

    /** @brief Timestamp of last chunk delivery */
    std::chrono::steady_clock::time_point m_last_chunk_time;

    std::vector<float> m_accumulated_buffer;  // Buffer for accumulating samples
    std::vector<float> m_overlap_buffer;      // Buffer for overlap samples

    bool m_end_of_file_reported{false};
    completion_callback_t m_completion_callback;

    // Helper methods

    /**
     * @brief Read and validate WAV file header
     * @param[in] file Input file stream
     * @param[out] sample_rate Detected sample rate
     * @param[out] channels Detected channel count
     * @param[out] bits_per_sample Detected bits per sample
     * @return true if valid WAV header, false otherwise
     */
    bool read_wav_header(std::ifstream& file, int& sample_rate,
                         int& channels, int& bits_per_sample);

    /**
     * @brief Validate WAV format parameters
     * @param sample_rate Sample rate from WAV header
     * @param channels Channel count from WAV header
     * @param bits_per_sample Bits per sample from WAV header
     * @return true if format is acceptable, false otherwise
     */
    bool validate_wav_format(int sample_rate, int channels, int bits_per_sample);

    /**
     * @brief Convert PCM samples to normalized float format
     * @param[in] pcm Input PCM samples
     * @param[out] output Output float samples (normalized to [-1.0, 1.0])
     */
    void convert_pcm_to_float(const std::vector<int16_t>& pcm,
                              std::vector<float>& output);

    /**
     * @brief Simulate real-time delay between chunks
     *
     * @details Sleeps to maintain real-time playback rate based on chunk_size_ms
     */
    void simulate_real_time_delay();
};
