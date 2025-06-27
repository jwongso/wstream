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
}

bool websocket_audio_source::get_audio_samples(std::vector<float>& samples) {
    if (!m_active) return false;

    audio_packet packet;
    if (m_audio_queue.try_dequeue(packet)) {
        samples = std::move(packet.samples);
        m_current_session_id = packet.session_id;
        m_current_language = packet.language;
        m_last_packet_time = std::chrono::steady_clock::now();
        return true;
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
