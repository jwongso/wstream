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

#include "sdl_audio_source.h"
#include "common.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <iostream>
#include <cstring>

sdl_audio_source::sdl_audio_source(const config& cfg)
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

    // Initialize to empty
    m_pcmf32.clear();
    m_pcmf32_new.clear();
    m_pcmf32_old.clear();
}

sdl_audio_source::~sdl_audio_source() {
    stop();
    if (m_audio) {
        m_audio.reset();
    }
}

bool sdl_audio_source::initialize(int device_id) {
    // If a specific device ID is requested, validate it first
    if (device_id >= 0) {
        int num_devices = SDL_GetNumAudioDevices(SDL_TRUE);
        if (device_id >= num_devices) {
            std::cerr << "[SDL Audio] Invalid audio device ID: " << device_id
                      << " (only " << num_devices << " devices available)" << std::endl;
            return false;
        }

        std::cout << "[SDL Audio] Initializing device " << device_id
                  << ": " << get_device_name(device_id) << std::endl;
    } else {
        std::cout << "[SDL Audio] Initializing default audio device" << std::endl;
    }

    m_audio = std::make_unique<audio_async>(m_config.length_ms);

    if (!m_audio->init(device_id, m_config.sample_rate)) {
        std::cerr << "[SDL Audio] Failed to initialize audio device" << std::endl;
        m_audio.reset();
        return false;
    }

    std::cout << "[SDL Audio] Successfully initialized"
              << " (sample_rate: " << m_config.sample_rate << " Hz"
              << ", step: " << m_config.step_ms << " ms"
              << ", buffer: " << m_config.length_ms << " ms"
              << ", overlap: " << m_config.keep_ms << " ms)" << std::endl;

    return true;
}

bool sdl_audio_source::start() {
    if (!m_audio) {
        std::cerr << "[SDL Audio] Cannot start - not initialized" << std::endl;
        return false;
    }

    m_audio->clear();

    m_audio->resume();
    m_is_active = true;

    // Clear buffers for fresh start
    m_pcmf32.clear();
    m_pcmf32_new.clear();
    m_pcmf32_old.clear();

    std::cout << "[SDL Audio] Started audio capture" << std::endl;
    return true;
}

void sdl_audio_source::stop() {
    if (m_audio) {
        m_audio->pause();
        m_is_active = false;

        m_audio->clear();
        m_pcmf32.clear();
        m_pcmf32_new.clear();
        m_pcmf32_old.clear();

        std::cout << "[SDL Audio] Stopped audio capture" << std::endl;
    }
}

bool sdl_audio_source::get_audio_samples(std::vector<float>& samples) {
    if (!m_audio || !m_is_active) {
        return false;
    }

    samples.clear();

    // Process audio with sliding window
    process_audio();

    if (m_pcmf32.empty()) {
        return false;
    }

    // Use move semantics for better performance
    samples = std::move(m_pcmf32);

    // Re-initialize m_pcmf32 for next iteration
    m_pcmf32.clear();

    return !samples.empty();
}

void sdl_audio_source::process_audio() {
    // Step 1: Collect enough audio data
    while (true) {
        m_audio->get(m_config.step_ms, m_pcmf32_new);

        // Safety check: Drop audio if we can't process fast enough
        if (static_cast<int>(m_pcmf32_new.size()) > OVERLOAD_MULTIPLIER * m_n_samples_step) {
            std::cerr << "[SDL Audio] WARNING: Cannot process audio fast enough, dropping "
                      << m_pcmf32_new.size() << " samples" << std::endl;
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
    const int n_samples_take = std::min(static_cast<int>(m_pcmf32_old.size()), m_n_samples_keep);

    // Step 3: Resize the processing buffer to fit overlap + new samples
    m_pcmf32.resize(n_samples_new + n_samples_take);

    // Step 4: Copy overlap samples first (for context)
    if (n_samples_take > 0) {
        std::copy(m_pcmf32_old.end() - n_samples_take, m_pcmf32_old.end(), m_pcmf32.begin());
    }

    // Step 5: Copy new samples
    std::copy(m_pcmf32_new.begin(), m_pcmf32_new.end(), m_pcmf32.begin() + n_samples_take);

    // Step 6: Keep overlap amount for next iteration
    if (static_cast<int>(m_pcmf32.size()) >= m_n_samples_keep) {
        m_pcmf32_old.assign(m_pcmf32.end() - m_n_samples_keep, m_pcmf32.end());
    } else {
        m_pcmf32_old = m_pcmf32;
    }
}

int sdl_audio_source::get_device_count() {
    return SDL_GetNumAudioDevices(SDL_TRUE);  // SDL_TRUE for capture devices
}

std::string sdl_audio_source::get_device_name(int device_id) {
    if (device_id < 0 || device_id >= get_device_count()) {
        return "";
    }

    const char* name = SDL_GetAudioDeviceName(device_id, SDL_TRUE);
    return name ? std::string(name) : "";
}
