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

#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <functional>

/**
 * @class benchmark_manager
 * @brief Manages benchmarking operations and metrics collection for ASR systems
 *
 * @details This class provides a complete benchmarking framework for ASR systems,
 * handling transcription accuracy measurement, timing statistics, and performance
 * metrics. It supports real-time progress monitoring and comprehensive result export.
 *
 * Key features:
 * - Word Error Rate (WER) and Character Error Rate (CER) calculation
 * -Levenshtein distance computation with operation breakdown
 * - Real-time factor (RTF) measurement
 * - Latency profiling (min/max/average)
 * - Confidence score tracking
 * - Detailed result export with segment-level analysis
 * - Progress callbacks for real-time monitoring
 *
 * Usage example:
 * @code
 * benchmark_manager bench;
 * bench.set_reference_text("the quick brown fox");
 * bench.start();
 *
 * // During ASR processing
 * bench.add_transcription("the quick brown", 0.95, 16000);
 * bench.add_transcription("fox", 0.92, 8000);
 *
 * // Get results
 * auto results = bench.stop();
 * bench.export_results(results, "benchmark_results.txt");
 * @endcode
 *
 * @note Thread-safe for concurrent access to results while benchmarking
 *
 * @see benchmark_audio_source
 */
class benchmark_manager {
public:
    /**
     * @struct transcription_segment
     * @brief Individual transcription segment with timing and quality metrics
     *
     * @details Represents a single transcribed segment with associated metadata
     * for detailed analysis of ASR performance.
     */
    struct transcription_segment {
        /** @brief Transcribed text for this segment */
        std::string text;

        /** @brief Timestamp when audio for this segment started processing */
        std::chrono::steady_clock::time_point start_time;

        /** @brief Timestamp when transcription was completed */
        std::chrono::steady_clock::time_point end_time;

        /** @brief Confidence score for this transcription (0.0-1.0)
         *  @details Higher values indicate higher confidence in transcription accuracy */
        double confidence{0.0};

        /** @brief Number of audio samples processed for this segment */
        size_t audio_samples_processed{0};

        double processing_latency_ms{0.0};

        /**
         * @brief Get the processing latency for this segment
         * @return Latency in milliseconds
         */
        double get_latency_ms() const {
            return processing_latency_ms;
        }
    };

    /**
     * @struct benchmark_results
     * @brief Comprehensive benchmark results and statistics
     *
     * @details Contains all metrics and statistics collected during a benchmark run,
     * including accuracy metrics, timing information, and quality indicators.
     */
    struct benchmark_results {
        // Accuracy metrics

        /** @brief Word Error Rate as percentage (0-100)
         *  @details WER = (S + D + I) / N * 100, where S=substitutions,
         *  D=deletions, I=insertions, N=reference words */
        double word_error_rate{0.0};

        /** @brief Character Error Rate as percentage (0-100)
         *  @details Similar to WER but calculated at character level */
        double character_error_rate{0.0};

        /** @brief Total number of words in reference text */
        int total_words{0};

        /** @brief Total number of word errors (S + D + I) */
        int word_errors{0};

        /** @brief Number of word substitutions */
        int word_substitutions{0};

        /** @brief Number of word deletions */
        int word_deletions{0};

        /** @brief Number of word insertions */
        int word_insertions{0};

        // Timing metrics

        /** @brief Total duration of processed audio in milliseconds */
        double total_audio_duration_ms{0.0};

        /** @brief Total wall-clock processing time in milliseconds */
        double total_processing_time_ms{0.0};

        /** @brief Real-time factor (processing time / audio duration)
         *  @details RTF < 1.0 means faster than real-time */
        double real_time_factor{0.0};

        /** @brief Average latency from audio input to transcription output */
        double average_latency_ms{0.0};

        /** @brief Minimum observed latency */
        double min_latency_ms{0.0};

        /** @brief Maximum observed latency */
        double max_latency_ms{0.0};

        // Throughput metrics

        /** @brief Total number of audio samples processed */
        size_t total_samples_processed{0};

        /** @brief Total number of transcription segments */
        size_t total_segments{0};

        /** @brief Audio processing throughput in samples per second */
        double samples_per_second{0.0};

        // Quality metrics

        /** @brief Average confidence score across all segments (0.0-1.0) */
        double average_confidence{0.0};

        // Full transcription

        /** @brief Reference (ground truth) text */
        std::string reference_text;

        /** @brief Hypothesis (ASR output) text */
        std::string hypothesis_text;

        // Individual segments

        /** @brief Detailed information for each transcription segment */
        std::vector<transcription_segment> segments;

        /**
         * @brief Check if results contain valid data
         * @return true if results are valid
         */
        bool is_valid() const {
            return total_segments > 0 && total_samples_processed > 0;
        }
    };

    // Constants

    /** @brief Default assumed sample rate for audio duration calculations */
    static constexpr double DEFAULT_SAMPLE_RATE = 16000.0;

    /** @brief Minimum confidence score threshold */
    static constexpr double MIN_CONFIDENCE = 0.0;

    /** @brief Maximum confidence score threshold */
    static constexpr double MAX_CONFIDENCE = 1.0;

    /** @brief Small value to prevent division by zero */
    static constexpr double EPSILON = 1e-10;

    /**
     * @brief Default constructor
     *
     * @details Initializes the benchmark manager in idle state.
     * Call set_reference_text() and start() to begin benchmarking.
     */
    benchmark_manager();

    /**
     * @brief Destructor - ensures proper cleanup
     *
     * @details Automatically calls stop() if benchmarking is still active.
     */
    ~benchmark_manager();

    /**
     * @brief Set the reference (ground truth) text for accuracy calculation
     * @param text Reference transcription text
     *
     * @details The reference text is normalized (lowercase, punctuation removed)
     * before storage. This text is used as the ground truth for WER/CER calculation.
     *
     * @note Should be called before start() for meaningful accuracy metrics
     */
    void set_reference_text(const std::string& text);

    /**
     * @brief Start benchmarking session
     *
     * @details Resets all metrics and begins timing. Previous results are cleared.
     *
     * @post Benchmark is in active state, ready to receive transcriptions
     */
    void start();

    /**
     * @brief Stop benchmarking and calculate final results
     * @return Final benchmark results
     *
     * @details Stops the benchmark timer and calculates all final metrics.
     * The benchmark manager can be restarted after calling stop().
     *
     * @post Benchmark is in idle state
     */
    benchmark_results stop();

    /**
     * @brief Add a transcription segment to the benchmark
     * @param text Transcribed text segment
     * @param confidence Confidence score for this transcription (0.0-1.0)
     * @param audio_samples Number of audio samples processed for this segment
     *
     * @details Records a new transcription segment with timing information.
     * The segment is timestamped and added to the results for analysis.
     *
     * @note Only processes segments when benchmark is active (after start())
     *
     * @param text Must not be empty for meaningful results
     * @param confidence Should be in range [0.0, 1.0], values outside are clamped
     * @param audio_samples Used for throughput calculations
     * @param processing_latency_ms Time taken to process this segment (optional)
     */
    void add_transcription(const std::string& text,
                           double confidence = 0.0,
                           size_t audio_samples = 0,
                           double processing_latency_ms = -1.0);

    /**
     * @brief Get current results without stopping the benchmark
     * @return Current benchmark results
     *
     * @details Returns a snapshot of current metrics without affecting the
     * ongoing benchmark. Useful for real-time monitoring.
     *
     * @note Results may be incomplete if benchmark is still running
     */
    benchmark_results get_current_results() const;

    /**
     * @brief Calculate Word Error Rate between two texts
     * @param reference Reference (ground truth) text
     * @param hypothesis Hypothesis (ASR output) text
     * @param[out] substitutions Optional pointer to store substitution count
     * @param[out] deletions Optional pointer to store deletion count
     * @param[out] insertions Optional pointer to store insertion count
     * @return WER as percentage (0-100)
     *
     * @details Calculates WER using Levenshtein distance at word level.
     * Both texts are normalized before comparison.
     *
     * Formula: WER = (S + D + I) / N * 100
     * - S = substitutions
     * - D = deletions
     * - I = insertions
     * - N = words in reference
     *
     * @note This is a static method and can be used independently
     */
    static double calculate_wer(const std::string& reference,
                                const std::string& hypothesis,
                                int* substitutions = nullptr,
                                int* deletions = nullptr,
                                int* insertions = nullptr);

    /**
     * @brief Calculate Character Error Rate between two texts
     * @param reference Reference (ground truth) text
     * @param hypothesis Hypothesis (ASR output) text
     * @return CER as percentage (0-100)
     *
     * @details Similar to WER but operates at character level.
     * Whitespace is ignored in the calculation.
     *
     * @note This is a static method and can be used independently
     */
    static double calculate_cer(const std::string& reference,
                                const std::string& hypothesis);

    /**
     * @brief Export benchmark results to a formatted text file
     * @param results Results to export
     * @param output_path Path to output file
     * @param model_path Path to the model file (optional, for metadata)
     *
     * @details Exports comprehensive benchmark results including:
     * - Accuracy metrics (WER, CER, error breakdown)
     * - Timing metrics (RTF, latencies)
     * - Throughput statistics
     * - Full transcriptions (reference and hypothesis)
     * - Segment-level details
     *
     * @note Creates or overwrites the output file
     */
    void export_results(const benchmark_results& results,
                        const std::string& output_path,
                        const std::string& model_path = "") const;

    /**
     * @brief Callback function type for progress updates
     * @param results Current benchmark results
     *
     * @details Called after each transcription segment is added
     */
    using progress_callback_t = std::function<void(const benchmark_results&)>;

    /**
     * @brief Set callback for real-time progress updates
     * @param callback Function to call on progress updates
     *
     * @details The callback is invoked after each add_transcription() call,
     * allowing real-time monitoring of benchmark progress.
     *
     * @note Callback is executed synchronously in add_transcription()
     */
    void set_progress_callback(progress_callback_t callback) {
        m_progress_callback = callback;
    }

    // Static helper methods

    /**
     * @brief Tokenize text into words
     * @param text Input text
     * @return Vector of word tokens
     *
     * @details Splits text into words, converts to lowercase, and removes
     * punctuation. Used for WER calculation.
     */
    static std::vector<std::string> tokenize(const std::string& text);

    /**
     * @brief Normalize text for comparison
     * @param text Input text
     * @return Normalized text
     *
     * @details Converts to lowercase, removes extra whitespace, and trims
     * leading/trailing spaces. Applied before accuracy calculations.
     */
    static std::string normalize_text(const std::string& text);

    /**
     * @brief Calculate Levenshtein distance between two sequences
     * @param ref Reference sequence
     * @param hyp Hypothesis sequence
     * @param[out] subs Optional pointer for substitution count
     * @param[out] dels Optional pointer for deletion count
     * @param[out] ins Optional pointer for insertion count
     * @return Edit distance
     *
     * @details Implements dynamic programming algorithm for edit distance
     * calculation with operation tracking.
     *
     * @note Template-like implementation using string vectors
     */
    static int levenshtein_distance(const std::vector<std::string>& ref,
                                    const std::vector<std::string>& hyp,
                                    int* subs, int* dels, int* ins);

    /**
     * @brief Get the reference text
     * @return Reference text string
     */
    std::string get_reference_text() const { return m_reference_text; }

private:
    /** @brief Reference text for accuracy calculation */
    std::string m_reference_text;

    /** @brief Collection of transcription segments */
    std::vector<transcription_segment> m_segments;

    /** @brief Benchmark start timestamp */
    std::chrono::steady_clock::time_point m_start_time;

    /** @brief Timestamp of last segment */
    std::chrono::steady_clock::time_point m_last_segment_time;

    /** @brief Flag indicating if benchmark is running */
    bool m_is_running;

    /** @brief Total audio samples processed */
    size_t m_total_samples;

    /** @brief Progress callback function */
    progress_callback_t m_progress_callback;
};
