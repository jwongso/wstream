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

#include "websocket_audio_source.h"
#include <iostream>

websocket_audio_source::websocket_audio_source()
    : m_last_packet_time(std::chrono::steady_clock::now()) {
}

websocket_audio_source::~websocket_audio_source() {
    stop();
}

bool websocket_audio_source::initialize() {
    // Nothing to initialize for WebSocket source
    return true;
}

bool websocket_audio_source::start() {
    m_active = true;
    return true;
}

void websocket_audio_source::stop() {
    m_active = false;

    // Clear any pending audio packets
    audio_packet packet;
    while (m_audio_queue.try_dequeue(packet)) {
        // Just discard
    }

    // Clear accumulator
    {
        std::lock_guard<std::mutex> lock(m_accumulator_mutex);
        m_accumulated_samples.clear();
    }
}

bool websocket_audio_source::get_audio_samples(std::vector<float>& samples) {
    if (!m_active) {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_accumulator_mutex);

    // Add new packets to accumulator
    audio_packet packet;
    while (m_audio_queue.try_dequeue(packet)) {
        m_accumulated_samples.insert(m_accumulated_samples.end(),
                                     packet.samples.begin(),
                                     packet.samples.end());
        m_current_session_id = packet.session_id;
        m_current_language = packet.language;
        m_last_packet_time = std::chrono::steady_clock::now();
    }

    // Whisper needs at least 1 second (16000 samples at 16kHz)
    const size_t CHUNK_SIZE = 16000;
    const size_t MIN_CHUNK_SIZE = 1600;  // Minimum 100ms for Whisper

    // Check if we have enough samples for a full chunk
    if (m_accumulated_samples.size() >= CHUNK_SIZE) {
        // Extract a chunk
        samples.clear();
        samples.assign(m_accumulated_samples.begin(),
                       m_accumulated_samples.begin() + CHUNK_SIZE);

        // Remove the extracted samples
        m_accumulated_samples.erase(m_accumulated_samples.begin(),
                                    m_accumulated_samples.begin() + CHUNK_SIZE);
        return true;
    }

    // Check if we should flush partial buffer due to timeout
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_packet = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      now - m_last_packet_time).count();

    // If no new audio for 2 seconds and we have some samples, flush them
    if (time_since_last_packet > 2000 && m_accumulated_samples.size() >= MIN_CHUNK_SIZE) {
        samples = std::move(m_accumulated_samples);
        m_accumulated_samples.clear();

        return true;
    }

    // Not enough samples yet - only log occasionally to avoid spam
    if (m_accumulated_samples.size() > 0) {
        static auto last_log_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() >= 5) {
            last_log_time = now;
        }
    }

    return false;
}

bool websocket_audio_source::is_active() const {
    if (!m_active) return false;

    // Check if we've received audio recently
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now - m_last_packet_time).count();

    return elapsed < ACTIVITY_TIMEOUT_MS;
}

void websocket_audio_source::handle_audio_data(const std::vector<int16_t>& pcm_samples,
                                               const std::string& session_id,
                                               const std::string& language) {
    if (!m_active) return;

    // Dump audio if enabled
    if (m_dump_enabled) {
        std::lock_guard<std::mutex> lock(m_dump_mutex);
        if (m_audio_dump_file.is_open()) {
            m_audio_dump_file.write(reinterpret_cast<const char*>(pcm_samples.data()),
                                    pcm_samples.size() * sizeof(int16_t));
            m_audio_dump_file.flush();
            m_total_samples_dumped += pcm_samples.size();
        }
    }

    audio_packet packet;

    // Convert int16_t to float
    packet.samples.reserve(pcm_samples.size());
    const float scale = 1.0f / 32768.0f;
    for (const auto& sample : pcm_samples) {
        packet.samples.push_back(sample * scale);
    }

    packet.session_id = session_id;
    packet.language = language;
    packet.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch()).count();

    m_audio_queue.enqueue(std::move(packet));
    m_last_packet_time = std::chrono::steady_clock::now();
}

void websocket_audio_source::enable_audio_dump(const std::string& filename) {
    std::lock_guard<std::mutex> lock(m_dump_mutex);
    if (m_audio_dump_file.is_open()) {
        m_audio_dump_file.close();
    }
    m_audio_dump_file.open(filename, std::ios::binary);
    m_dump_enabled = m_audio_dump_file.is_open();
    m_total_samples_dumped = 0;
    if (m_dump_enabled) {
        std::cout << "[WebSocket Audio] Audio dump enabled to file: " << filename << std::endl;
    }
}

void websocket_audio_source::disable_audio_dump() {
    std::lock_guard<std::mutex> lock(m_dump_mutex);
    m_dump_enabled = false;
    if (m_audio_dump_file.is_open()) {
        m_audio_dump_file.close();
        std::cout << "[WebSocket Audio] Audio dump disabled. Total samples dumped: "
                  << m_total_samples_dumped << std::endl;
    }
}
