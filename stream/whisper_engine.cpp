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

#include "whisper_engine.h"
#include <thread>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <string.h>

bool is_repetitive_text(const std::string& text) {
    if (text.length() < 4) return false;

    // Check for simple repetitions (e.g., "aa", "abab", etc.)
    const size_t half_len = text.length() / 2;
    if (half_len > 0) {
        std::string first_half = text.substr(0, half_len);
        std::string second_half = text.substr(half_len, half_len);
        if (first_half == second_half) {
            return true;
        }
    }

    // Check for character repetition (e.g., "aaaa", "....", etc.)
    char first_char = text[0];
    bool all_same = true;
    for (char c : text) {
        if (c != first_char) {
            all_same = false;
            break;
        }
    }

    return all_same;
}

float calculate_entropy(const std::vector<float>& probabilities) {
    float entropy = 0.0f;
    for (float p : probabilities) {
        if (p > 1e-10f) {  // Avoid log(0)
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

whisper_engine::whisper_engine(const std::string& model_path, const config& cfg)
    : m_model_path(model_path), m_config(cfg) {
}

whisper_engine::~whisper_engine() {
    if (m_ctx) {
        whisper_free(m_ctx);
    }
}

bool whisper_engine::initialize(bool wasm) {
    m_wasm = wasm;
    // Set up context parameters
    struct whisper_context_params cparams = whisper_context_default_params();
    if (m_wasm) {
        cparams.use_gpu = false;
    }
    else {
        cparams.use_gpu = m_config.use_gpu;
    }

    m_ctx = whisper_init_from_file_with_params(m_model_path.c_str(), cparams);
    if (!m_ctx) {
        std::cerr << "Failed to initialize whisper context from " << m_model_path << std::endl;
        return false;
    }

    setup_whisper_params();
    return true;
}

void whisper_engine::setup_whisper_params() {
    m_wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    m_wparams.print_special = false;
    m_wparams.print_progress = false;
    m_wparams.print_realtime = false;
    m_wparams.print_timestamps = false;
    m_wparams.suppress_blank = true;
    m_wparams.no_context = true;
    m_wparams.translate = false;
    m_wparams.temperature = 0.0f;

    if (m_wasm) {
        m_wparams.n_threads = 4;
        m_wparams.audio_ctx = 512;
        m_wparams.max_tokens = 64;
        m_wparams.max_len = 0;
        m_wparams.entropy_thold = 2.4f;
        m_wparams.logprob_thold = -1.0f;
        m_wparams.no_speech_thold = 0.6f;
        m_wparams.temperature = 0.0f;
        m_wparams.temperature_inc = -1.0f;
        m_wparams.single_segment = true;
        m_wparams.language = "en";
    }
    else {
        // Native optimizations
        m_wparams.language = m_config.language.c_str();
        m_wparams.n_threads = m_config.n_threads > 0 ? m_config.n_threads : get_optimal_thread_count();
        m_wparams.single_segment = !m_config.use_vad;
        m_wparams.max_tokens = m_config.max_tokens;
        m_wparams.audio_ctx = 0;

        m_wparams.no_context = true;

        // Additional anti-repetition settings
        m_wparams.entropy_thold = 2.4f;     // Skip low-entropy (repetitive) outputs
        m_wparams.logprob_thold = -1.0f;    // Skip low-confidence outputs
        m_wparams.no_speech_thold = 0.6f;   // Skip segments that are likely not speech

        // Use temperature fallback only if needed
        m_wparams.temperature_inc = m_config.temperature > 0.0f ? 0.2f : -1.0f;
    }
}

std::string whisper_engine::transcribe(const std::vector<float>& audio_data) {
    if (!m_ctx) {
        std::cerr << "Error: Whisper context is null!" << std::endl;
        return "";
    }

    if (audio_data.empty()) {
        return "";
    }

    // Check if audio has actual content (not just silence)
    const float silence_threshold = 0.001f;
    const float max_val = *std::max_element(audio_data.begin(),
                                            audio_data.end(),
                                            [](float a, float b) {
                                return std::abs(a) < std::abs(b);
                            });

    if (std::abs(max_val) < silence_threshold) {
        return "";
    }

    whisper_reset_timings(m_ctx);

    // Run inference
    int ret = whisper_full(m_ctx, m_wparams, audio_data.data(), audio_data.size());

    if (ret != 0) {
        std::cerr << "whisper_full failed with code: " << ret << std::endl;
        return "";
    }

    // Get number of segments
    int n_segments = whisper_full_n_segments(m_ctx);
    if (n_segments == 0) {
        return "";
    }

    // Build result string
    std::string result;
    result.reserve(n_segments * 20);
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(m_ctx, i);
        if (text && strlen(text) > 0) {
            std::string segment_text(text);
            if (segment_text.length() > 1 && !is_repetitive_text(segment_text)) {
                if (!result.empty()) {
                    result += " ";
                }
                result += segment_text;
            }
        }
    }

    return result;
}

transcription_result whisper_engine::transcribe_with_confidence(const std::vector<float>& audio_data) {
    transcription_result result;

    if (!m_ctx || audio_data.empty()) {
        return result;
    }

    // OPTIMIZATION: More efficient silence detection
    const float silence_threshold = 0.001f;
    const float max_val = *std::max_element(audio_data.begin(), audio_data.end(),
                                            [](float a, float b) { return std::abs(a) < std::abs(b); });

    if (std::abs(max_val) < silence_threshold) {
        return result;
    }

    // CRITICAL: Reset context state before processing
    whisper_reset_timings(m_ctx);

    // Run inference
    int ret = whisper_full(m_ctx, m_wparams, audio_data.data(), audio_data.size());

    if (ret != 0) {
        std::cerr << "whisper_full failed with code: " << ret << std::endl;
        return result;
    }

    // Get number of segments
    int n_segments = whisper_full_n_segments(m_ctx);
    if (n_segments == 0) {
        return result;
    }

    // OPTIMIZATION: Pre-allocate string
    result.text.reserve(n_segments * 20);

    // Process segments with confidence calculation
    float total_logprob = 0.0f;
    int total_tokens = 0;
    std::vector<float> segment_probs;  // For entropy calculation
    segment_probs.reserve(n_segments);

    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(m_ctx, i);
        if (!text || strlen(text) == 0) continue;

        std::string segment_text(text);

        // OPTIMIZATION: Skip repetitive segments
        if (segment_text.length() <= 1 || is_repetitive_text(segment_text)) {
            continue;
        }

        // Add to result
        if (!result.text.empty()) {
            result.text += " ";
        }
        result.text += segment_text;

        // Calculate confidence metrics
        float segment_logprob = 0.0f;
        int n_tokens = whisper_full_n_tokens(m_ctx, i);

        if (n_tokens > 0) {
            float segment_sum = 0.0f;
            for (int j = 0; j < n_tokens; ++j) {
                whisper_token_data td = whisper_full_get_token_data(m_ctx, i, j);
                float token_prob = td.p;

                // OPTIMIZATION: Improved probability handling
                float log_prob;
                if (token_prob > 0 && token_prob <= 1.0f) {
                    log_prob = std::log(std::max(token_prob, 1e-10f));  // Prevent log(0)
                } else if (token_prob > 1.0f) {
                    log_prob = -std::log(1.0f + std::exp(-token_prob));
                } else {
                    log_prob = token_prob;
                }

                segment_sum += log_prob;
            }
            segment_logprob = segment_sum / n_tokens;
            segment_probs.push_back(std::exp(segment_logprob));  // Convert back to prob for entropy
        }

        total_logprob += segment_logprob;
        total_tokens += n_tokens;
    }

    // Calculate final metrics
    if (total_tokens > 0) {
        result.avg_logprob = total_logprob / total_tokens;
        result.n_tokens = total_tokens;

        // OPTIMIZATION: Calculate entropy from segment probabilities
        if (!segment_probs.empty()) {
            result.entropy = calculate_entropy(segment_probs);
        }
    }

    return result;
}

int whisper_engine::get_optimal_thread_count() const {
    int hardware_threads = std::thread::hardware_concurrency();
    return std::max(1, hardware_threads <= 4 ?
                           hardware_threads - 1 : hardware_threads - RESERVED_THREADS);
}
