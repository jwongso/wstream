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

#include "whisper.h"
#include <string>
#include <vector>
#include <memory>

struct transcription_result {
    std::string text;
    float avg_logprob = 0.0f;       // Average log probability (higher = better)
    float no_speech_prob = 0.0f;    // Probability of no speech (lower = better)
    float entropy = 0.0f;           // Entropy (lower = more confident)
    int n_tokens = 0;               // Number of tokens
};

/**
 * @class whisper_engine
 * @brief High-level interface for Whisper speech recognition
 *
 * This class provides a simplified interface to OpenAI's Whisper speech
 * recognition system, handling:
 * - Model loading and initialization
 * - Audio preprocessing for optimal recognition
 * - Inference parameter optimization
 * - Multi-threaded processing when available
 * - GPU acceleration when supported
 *
 * The engine is optimized for real-time streaming applications with
 * minimal latency while maintaining high accuracy.
 *
 * @par Performance Considerations:
 * - Uses greedy decoding for speed
 * - Optimizes thread count based on hardware
 * - Supports GPU acceleration when available
 * - Minimizes memory allocations during inference
 *
 * @par Thread Safety:
 * This class is not thread-safe. Create separate instances for
 * concurrent processing or provide external synchronization.
 */
class whisper_engine {
public:
    /// Default GPU usage setting
    static constexpr bool DEFAULT_USE_GPU = true;

    /// Auto-detect thread count (0 = automatic)
    static constexpr int AUTO_DETECT_THREADS = 0;

    /// Default language for recognition
    static constexpr const char* DEFAULT_LANGUAGE = "en";

    /// Default temperature for sampling (0.0 = deterministic)
    static constexpr float DEFAULT_TEMPERATURE = 0.0f;

    /// Default maximum tokens per segment
    static constexpr int DEFAULT_MAX_TOKENS = 16;

    /// Default VAD usage setting
    static constexpr bool DEFAULT_USE_VAD = false;

    /// Number of threads to reserve for system/other processes
    static constexpr int RESERVED_THREADS = 2;

    /// Minimum number of threads to use
    static constexpr int MIN_THREADS = 1;

    /**
     * @struct config
     * @brief Configuration parameters for Whisper engine
     */
    struct config {
        /// Enable GPU acceleration (CUDA/OpenCL)
        bool use_gpu;

        /// Number of CPU threads (0 = auto-detect optimal count)
        int n_threads;

        /// Recognition language code (ISO 639-1)
        std::string language;

        /// Sampling temperature (0.0 = greedy/deterministic, higher = more random)
        float temperature;

        /// Maximum tokens per recognition segment
        int max_tokens;

        /// Enable Voice Activity Detection integration
        bool use_vad;

        /**
         * @brief Default constructor with optimal settings
         *
         * Configures the engine for real-time processing with:
         * - GPU acceleration enabled
         * - Automatic thread detection
         * - English language recognition
         * - Deterministic (greedy) decoding
         * - Optimized token limits for streaming
         */
        config()
            : use_gpu(DEFAULT_USE_GPU)
            , n_threads(AUTO_DETECT_THREADS)
            , language(DEFAULT_LANGUAGE)
            , temperature(DEFAULT_TEMPERATURE)
            , max_tokens(DEFAULT_MAX_TOKENS)
            , use_vad(DEFAULT_USE_VAD) {}
    };

    /**
     * @brief Constructs Whisper engine with specified model and configuration
     * @param model_path Path to the Whisper model file (.bin format)
     * @param cfg Engine configuration parameters
     *
     * @par Model Requirements:
     * - Must be a valid Whisper model in .bin format
     * - Model should be compatible with the whisper.cpp library
     * - Recommended models: small.en, base.en, medium.en for real-time use
     */
    explicit whisper_engine(const std::string& model_path, const config& cfg = config{});

    /**
     * @brief Destructor - ensures proper cleanup of Whisper resources
     */
    ~whisper_engine();

    /**
     * @brief Initializes the Whisper model and processing parameters
     * @param wasm Initialize for WebAssembly application, default to false
     * @return true if initialization successful, false otherwise
     *
     * This method:
     * - Loads the specified Whisper model
     * - Configures processing parameters
     * - Sets up GPU acceleration if available
     * - Optimizes thread allocation
     *
     * @note Must be called before transcribe() can be used
     */
    bool initialize(bool wasm = false);

    /**
     * @brief Transcribes audio data to text
     * @param audio_data Vector of audio samples (32-bit float, normalized)
     * @return Transcribed text string, empty if transcription failed
     *
     * Processes the provided audio samples through the Whisper model
     * and returns the recognized text. The audio should be:
     * - Single channel (mono)
     * - Sample rate matching WHISPER_SAMPLE_RATE
     * - Normalized float values (-1.0 to 1.0)
     *
     * @par Performance Notes:
     * - Processing time depends on audio length and model size
     * - GPU acceleration significantly improves performance
     * - Multiple short segments often process faster than single long segments
     */
    std::string transcribe(const std::vector<float>& audio_data);

    transcription_result transcribe_with_confidence(const std::vector<float>& audio_data);

    /**
     * @brief Checks if the engine is properly initialized
     * @return true if ready for transcription, false otherwise
     */
    bool is_initialized() const { return m_ctx != nullptr; }

private:
    /// Path to the Whisper model file
    std::string m_model_path;

    /// Engine configuration
    config m_config;

    /// Whisper context handle
    whisper_context* m_ctx = nullptr;

    /// Whisper processing parameters
    whisper_full_params m_wparams;

    /// WASM flag
    bool m_wasm;

    /**
     * @brief Determines optimal thread count for current hardware
     * @return Recommended number of threads for processing
     *
     * Analyzes system capabilities and returns an optimal thread count
     * that balances performance with system responsiveness.
     */
    int get_optimal_thread_count() const;

    /**
     * @brief Configures Whisper processing parameters
     *
     * Sets up all Whisper parameters for optimal real-time performance:
     * - Disables verbose output for speed
     * - Configures threading
     * - Sets language and sampling parameters
     * - Optimizes for streaming use case
     */
    void setup_whisper_params();
};
