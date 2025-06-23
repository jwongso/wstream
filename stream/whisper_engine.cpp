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
#include <iostream>

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
    if (m_wasm) {
        // Core performance settings
        m_wparams.n_threads = 4; // Sweet spot for browsers
        m_wparams.audio_ctx = 512; // Reduced from 768

        // Token and context limits
        m_wparams.max_tokens = 64; // Increased from 32
        m_wparams.max_len = 0; // No max length

        // Quality thresholds - these are standard
        m_wparams.entropy_thold = 2.4f;
        m_wparams.logprob_thold = -1.0f;
        m_wparams.no_speech_thold = 0.6f;

        // Temperature settings
        m_wparams.temperature = 0.0f;
        m_wparams.temperature_inc = -1.0f; // Disable fallback

        // Standard flags that exist in all versions
        m_wparams.print_special = false;
        m_wparams.print_progress = false;
        m_wparams.print_realtime = false;
        m_wparams.print_timestamps = false;
        m_wparams.suppress_blank = true;
        m_wparams.single_segment = true;
        m_wparams.no_context = true;
        m_wparams.translate = false;

        // Language
        m_wparams.language = "en";
    }
    else {
        m_wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        m_wparams.print_progress = false;
        m_wparams.print_realtime = false;
        m_wparams.no_context = true;
        m_wparams.language = m_config.language.c_str();
        m_wparams.n_threads = m_config.n_threads > 0 ? m_config.n_threads : get_optimal_thread_count();
        m_wparams.temperature = m_config.temperature;
        m_wparams.single_segment = !m_config.use_vad;
        m_wparams.max_tokens = m_config.max_tokens;
        m_wparams.audio_ctx = 0;
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
    float max_val = 0.0f;
    for (const auto& sample : audio_data) {
        max_val = std::max(max_val, std::abs(sample));
    }

    if (max_val < 0.001f) {
        return "";
    }

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
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(m_ctx, i);
        result += text;
        if (i < n_segments - 1) {
            result += " ";
        }
    }

    return result;
}

int whisper_engine::get_optimal_thread_count() const {
    int hardware_threads = std::thread::hardware_concurrency();
    return std::max(1, hardware_threads - 2);
}
