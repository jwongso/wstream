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

#include "audio_processor.h"
#include "common.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <iostream>
#include <cstring>

audio_processor::audio_processor(const config& cfg)
    : m_config(cfg) {

    m_n_samples_30s = (1e-3 * 30000.0) * m_config.sample_rate;
    m_n_samples_len = (1e-3 * m_config.length_ms) * m_config.sample_rate;
    m_n_samples_step = (1e-3 * m_config.step_ms) * m_config.sample_rate;
    m_n_samples_keep = (1e-3 * m_config.keep_ms) * m_config.sample_rate;

    // Pre-allocate vectors
    m_pcmf32.resize(m_n_samples_30s, 0.0f);
    m_pcmf32_new.resize(m_n_samples_30s, 0.0f);
    m_pcmf32_old.reserve(m_n_samples_keep);
}

audio_processor::~audio_processor() {
    if (m_audio) {
        m_audio.reset();
    }
}

bool audio_processor::initialize(int device_id) {
    m_audio = std::make_unique<audio_async>(m_config.length_ms);

    if (!m_audio->init(device_id, m_config.sample_rate)) {
        return false;
    }

    return true;
}

void audio_processor::pause() {
    if (m_audio) {
        m_audio->pause();
    }
}

void audio_processor::resume() {
    if (m_audio) {
        m_audio->resume();
    }
}

bool audio_processor::get_processed_samples(std::vector<float>& samples) {
    if (!m_audio) return false;

    if (!m_config.use_vad) {
        process_non_vad();
    } else {
        process_vad();
    }

    if (!m_pcmf32.empty()) {
        samples = m_pcmf32;
        return true;
    }

    return false;
}

void audio_processor::process_non_vad() {
    while (true) {
        m_audio->get(m_config.step_ms, m_pcmf32_new);

        if (static_cast<int>(m_pcmf32_new.size()) > 2 * m_n_samples_step) {
            std::cerr << "WARNING: cannot process audio fast enough, dropping audio..." << std::endl;
            m_audio->clear();
            continue;
        }

        if (static_cast<int>(m_pcmf32_new.size()) >= m_n_samples_step) {
            m_audio->clear();
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const int n_samples_new = m_pcmf32_new.size();
    const int n_samples_take = std::min(
        static_cast<int>(m_pcmf32_old.size()),
        std::max(0, m_n_samples_keep + m_n_samples_len - n_samples_new)
        );

    m_pcmf32.resize(n_samples_new + n_samples_take);

    for (int i = 0; i < n_samples_take; i++) {
        m_pcmf32[i] = m_pcmf32_old[m_pcmf32_old.size() - n_samples_take + i];
    }

    std::memcpy(m_pcmf32.data() + n_samples_take, m_pcmf32_new.data(),
                n_samples_new * sizeof(float));
    m_pcmf32_old = m_pcmf32;
}

void audio_processor::process_vad() {
    static auto t_last = std::chrono::high_resolution_clock::now();

    const auto t_now = std::chrono::high_resolution_clock::now();
    const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

    if (t_diff < 2000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        m_pcmf32.clear();
        return;
    }

    m_audio->get(2000, m_pcmf32_new);

    if (::vad_simple(m_pcmf32_new, m_config.sample_rate, 1000, 0.85f, 100.0f, false)) {
        m_audio->get(m_config.length_ms, m_pcmf32);
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        m_pcmf32.clear();
        return;
    }

    t_last = t_now;
}
