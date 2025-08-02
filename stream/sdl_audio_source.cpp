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
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    // Initialize to empty (not filled with zeros)
    m_pcmf32.clear();
    m_pcmf32_new.clear();
    m_pcmf32_old.clear();

    // Initialize VAD timestamp
    m_last_vad_time = std::chrono::steady_clock::now();
}

sdl_audio_source::~sdl_audio_source() {
    if (m_audio) {
        m_audio.reset();
    }
}

bool sdl_audio_source::initialize(int device_id) {
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

void sdl_audio_source::pause() {
    if (m_audio) {
        m_audio->pause();
        m_is_active = false;
    }
}

void sdl_audio_source::resume() {
    if (m_audio) {
        m_audio->resume();
        m_is_active = true;
        // Reset VAD timestamp
        m_last_vad_time = std::chrono::steady_clock::now();
    }
}

bool sdl_audio_source::get_processed_samples(std::vector<float>& samples) {
    if (!m_audio || !m_is_active) {
        return false;
    }

    samples.clear();

    if (!m_config.use_vad) {
        process_non_vad();
    } else {
        process_vad();
    }

    if (m_pcmf32.empty()) {
        return false;
    }

    // Use move semantics for better performance
    samples = std::move(m_pcmf32);

    // Re-initialize m_pcmf32 with capacity but empty
    m_pcmf32.clear();

    return !samples.empty();
}

void sdl_audio_source::process_non_vad() {
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

void sdl_audio_source::high_pass_filter(std::vector<float>& data, float cutoff, float sample_rate) {
    const float rc = 1.0f / (2.0f * M_PI * cutoff);
    const float dt = 1.0f / sample_rate;
    const float alpha = dt / (rc + dt);

    float y = data[0];

    for (size_t i = 1; i < data.size(); i++) {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

bool sdl_audio_source::vad_simple(std::vector<float>& pcmf32, int sample_rate, int last_ms,
                                 float vad_thold, float freq_thold, bool verbose) {
    const int n_samples = pcmf32.size();
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples) {
        // not enough samples - assume no speech
        return false;
    }

    // Create a copy for filtering if needed
    std::vector<float> pcmf32_filtered = pcmf32;

    if (freq_thold > 0.0f) {
        high_pass_filter(pcmf32_filtered, freq_thold, sample_rate);
    }

    float energy_all = 0.0f;
    float energy_last = 0.0f;

    for (int i = 0; i < n_samples; i++) {
        float sample_energy = fabsf(pcmf32_filtered[i]);
        energy_all += sample_energy;

        if (i >= n_samples - n_samples_last) {
            energy_last += sample_energy;
        }
    }

    energy_all /= n_samples;
    energy_last /= n_samples_last;

    if (verbose) {
        float energy_ratio = (energy_all > 0) ? (energy_last / energy_all) : 0.0f;
        std::cerr << "[VAD] energy_all: " << energy_all
                  << ", energy_last: " << energy_last
                  << ", ratio: " << energy_ratio
                  << ", threshold: " << vad_thold << std::endl;
    }

    // If recent energy is too high compared to average, it's likely noise
    if (energy_last > vad_thold * energy_all) {
        return false;
    }

    return true;
}

void sdl_audio_source::process_vad() {
    static auto t_last = std::chrono::steady_clock::now();

    const auto t_now = std::chrono::steady_clock::now();
    const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

    if (t_diff < VAD_PROCESS_INTERVAL_MS) {
        std::this_thread::sleep_for(std::chrono::milliseconds(VAD_SLEEP_MS));
        m_pcmf32.clear();
        return;
    }

    // Get a chunk of audio for VAD detection
    m_pcmf32_new.clear();
    m_audio->get(VAD_DETECTION_LENGTH_MS, m_pcmf32_new);

    // Check if we got any samples
    if (m_pcmf32_new.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(VAD_SLEEP_MS));
        m_pcmf32.clear();
        return;
    }

    // Need enough samples for VAD
    size_t expected_samples = (VAD_DETECTION_LENGTH_MS * m_config.sample_rate) / 1000;
    if (m_pcmf32_new.size() < expected_samples) {
        std::this_thread::sleep_for(std::chrono::milliseconds(VAD_SLEEP_MS));
        m_pcmf32.clear();
        return;
    }

    // Run VAD on the audio chunk
    // Use a smaller window for "last" comparison (e.g., 200ms out of 1000ms)
    const int VAD_LAST_MS = 200;  // Check last 200ms against full 1000ms

    bool speech_detected = vad_simple(m_pcmf32_new, m_config.sample_rate,
                                      VAD_LAST_MS,  // Use smaller window
                                      VAD_ENERGY_THRESHOLD,
                                      VAD_FREQ_THRESHOLD,
                                      false);  // Disable verbose output

    if (speech_detected) {
        // Speech detected - get full audio segment for processing
        m_pcmf32.clear();
        m_audio->get(m_config.length_ms, m_pcmf32);

        // Make sure we got enough audio
        size_t min_samples = (m_config.step_ms * m_config.sample_rate) / 1000;
        if (m_pcmf32.size() < min_samples) {
            std::this_thread::sleep_for(std::chrono::milliseconds(VAD_SLEEP_MS));
            m_pcmf32.clear();
            return;
        }

        // IMPORTANT: Clear the audio buffer to prevent repetition
        m_audio->clear();

        // Reset the timer to prevent immediate reprocessing
        t_last = t_now + std::chrono::milliseconds(500);  // Add extra delay after speech
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(VAD_SLEEP_MS));
        m_pcmf32.clear();
        return;
    }
}
