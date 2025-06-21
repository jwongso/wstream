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

bool whisper_engine::initialize() {
    // Set up context parameters
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = m_config.use_gpu;

    m_ctx = whisper_init_from_file_with_params(m_model_path.c_str(), cparams);
    if (!m_ctx) {
        return false;
    }

    setup_whisper_params();
    return true;
}

void whisper_engine::setup_whisper_params() {
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

std::string whisper_engine::transcribe(const std::vector<float>& audio_data) {
    if (!m_ctx || audio_data.empty()) {
        return "";
    }

    // Run inference
    if (whisper_full(m_ctx, m_wparams, audio_data.data(), audio_data.size()) != 0) {
        std::cerr << "Failed to process audio.\n";
        return "";
    }

    // Get the transcription
    const int n_segments = whisper_full_n_segments(m_ctx);
    if (n_segments <= 0) {
        return "";
    }

    std::string result;
    result.reserve(n_segments * 64); // Estimate

    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(m_ctx, i);
        if (text) {
            result += text;
        }
    }

    return result;
}

int whisper_engine::get_optimal_thread_count() const {
    int hardware_threads = std::thread::hardware_concurrency();
    return std::max(1, hardware_threads - 2);
}
