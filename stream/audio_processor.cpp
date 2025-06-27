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
    : m_config(cfg), m_is_active(false) {

    // Pre-calculate sample counts for efficiency
    m_n_samples_30s = static_cast<int>((MS_TO_SECONDS * BUFFER_30S_DURATION) * m_config.sample_rate);
    m_n_samples_len = static_cast<int>((MS_TO_SECONDS * m_config.length_ms) * m_config.sample_rate);
    m_n_samples_step = static_cast<int>((MS_TO_SECONDS * m_config.step_ms) * m_config.sample_rate);
    m_n_samples_keep = static_cast<int>((MS_TO_SECONDS * m_config.keep_ms) * m_config.sample_rate);

    // Pre-allocate vectors with capacity to avoid reallocations
    m_pcmf32.reserve(m_n_samples_30s);
    m_pcmf32_new.reserve(m_n_samples_30s);
    m_pcmf32_old.reserve(m_n_samples_keep);

    // Initialize to empty (not filled with zeros)
    m_pcmf32.clear();
    m_pcmf32_new.clear();
    m_pcmf32_old.clear();
}

audio_processor::~audio_processor() {
    if (m_audio) {
        m_audio.reset();
    }
}

bool audio_processor::initialize(int device_id) {
    // If a specific device ID is requested, validate it first
    if (device_id >= 0) {
        int num_devices = SDL_GetNumAudioDevices(SDL_TRUE);
        if (device_id >= num_devices) {
            std::cerr << "Invalid audio device ID: " << device_id
                      << " (only " << num_devices << " devices available)" << std::endl;
            return false;
        }
    }

    m_audio = std::make_unique<audio_async>(m_config.length_ms);

    if (!m_audio->init(device_id, m_config.sample_rate)) {
        m_audio.reset();
        return false;
    }

    return true;
}

void audio_processor::pause() {
    if (m_audio) {
        m_audio->pause();
        m_is_active = false;
    }
}

void audio_processor::resume() {
    if (m_audio) {
        m_audio->resume();
        m_is_active = true;
    }
}

bool audio_processor::get_processed_samples(std::vector<float>& samples) {
    if (!m_audio || !m_is_active) return false;

    samples.clear();

    if (!m_config.use_vad) {
        process_non_vad();
    } else {
        process_vad();
    }

    if (m_pcmf32.empty()) return false;

    // Use move semantics for better performance
    samples = std::move(m_pcmf32);

    // Re-initialize m_pcmf32 with capacity but empty
    m_pcmf32.clear();

    return !samples.empty();
}

void audio_processor::process_non_vad() {
    // Step 1: Collect enough audio data
    while (true) {
        m_audio->get(m_config.step_ms, m_pcmf32_new);
        // Safety check: Drop audio if we can't process fast enough
        if (static_cast<int>(m_pcmf32_new.size()) > 2 * m_n_samples_step) {
            std::cerr << "WARNING: cannot process audio fast enough, dropping audio..." << std::endl;
            m_audio->clear();
            continue;
        }
        // Break when we have enough samples
        if (static_cast<int>(m_pcmf32_new.size()) >= m_n_samples_step) {
            // Clear audio buffer to prevent duplicate processing
            m_audio->clear();
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(AUDIO_WAIT_SLEEP_MS));
    }

    // Step 2: Determine overlap amount
    const int n_samples_new = m_pcmf32_new.size();
    // Only take needed overlap amount
    const int n_samples_take = std::min(static_cast<int>(m_pcmf32_old.size()), m_n_samples_keep);

    // Step 3: Resize the processing buffer to fit overlap + new samples
    m_pcmf32.resize(n_samples_new + n_samples_take);

    // Step 4: Copy overlap samples first (use std::copy for better optimization)
    if (n_samples_take > 0) {
        std::copy(m_pcmf32_old.end() - n_samples_take, m_pcmf32_old.end(), m_pcmf32.begin());
    }

    // Step 5: Copy new samples (use std::copy instead of memcpy for type safety)
    std::copy(m_pcmf32_new.begin(), m_pcmf32_new.end(), m_pcmf32.begin() + n_samples_take);

    // Step 6: Keep only overlap amount for next iteration
    if (static_cast<int>(m_pcmf32.size()) >= m_n_samples_keep) {
        m_pcmf32_old.assign(m_pcmf32.end() - m_n_samples_keep, m_pcmf32.end());
    } else {
        m_pcmf32_old = m_pcmf32;
    }
}

void audio_processor::process_vad() {
    static auto t_last = std::chrono::steady_clock::now();

    const auto t_now = std::chrono::steady_clock::now();
    const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

    if (t_diff < VAD_PROCESS_INTERVAL_MS) {
        std::this_thread::sleep_for(std::chrono::milliseconds(VAD_SLEEP_MS));
        m_pcmf32.clear();  // Clear buffer when not processing
        return;
    }

    // Get a small chunk for VAD detection
    m_pcmf32_new.clear();  // Clear before getting new data
    m_audio->get(VAD_DETECTION_LENGTH_MS, m_pcmf32_new);

    // Run VAD
    if (::vad_simple(m_pcmf32_new, m_config.sample_rate, VAD_DETECTION_LENGTH_MS,
                     VAD_ENERGY_THRESHOLD, VAD_FREQ_THRESHOLD, false)) {
        // Speech detected - get full audio segment
        m_pcmf32.clear();  // Clear before getting new data
        m_audio->get(m_config.length_ms, m_pcmf32);
    } else {
        // No speech detected
        std::this_thread::sleep_for(std::chrono::milliseconds(VAD_SLEEP_MS));
        m_pcmf32.clear();  // Clear buffer when not processing
        return;
    }

    t_last = t_now;
}
