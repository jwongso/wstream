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

#include "audio_source.h"
#include "concurrentqueue.h"
#include <vector>
#include <string>
#include <atomic>
#include <chrono>

/**
 * @class websocket_audio_source
 * @brief WebSocket-based audio source implementation
 *
 * Receives audio data from WebSocket clients and provides it to the whisper engine.
 * Implements the audio_source interface for seamless integration with the
 * audio source switching system.
 */
class websocket_audio_source : public audio_source {
public:
    /**
     * @brief Default constructor
     */
    websocket_audio_source();

    /**
     * @brief Destructor - ensures proper cleanup
     */
    ~websocket_audio_source() override;

    /**
     * @brief Initializes the WebSocket audio source
     * @return true if initialization successful, false otherwise
     */
    bool initialize() override;

    /**
     * @brief Starts audio reception
     * @return true if started successfully, false otherwise
     */
    bool start() override;

    /**
     * @brief Stops audio reception
     */
    void stop() override;

    /**
     * @brief Retrieves processed audio samples
     * @param[out] samples Vector to store the retrieved audio samples
     * @return true if samples were retrieved, false if no samples available
     */
    bool get_audio_samples(std::vector<float>& samples) override;

    /**
     * @brief Gets the name/identifier of this audio source
     * @return String identifier for this audio source
     */
    std::string get_name() const override { return "WebSocket Client"; }

    /**
     * @brief Checks if the audio source is active
     * @return true if active and providing samples, false otherwise
     */
    bool is_active() const override;

    /**
     * @brief Gets session ID for current audio stream
     * @return Session ID string for the current audio stream
     */
    std::string get_session_id() const override { return m_current_session_id; }

    /**
     * @brief Gets language hint for current audio stream
     * @return Language code string for the current audio stream
     */
    std::string get_language() const override { return m_current_language; }

    /**
     * @brief Handles incoming audio data from WebSocket clients
     * @param samples PCM audio samples (16-bit)
     * @param session_id Client session identifier
     * @param language Language hint (optional)
     */
    void handle_audio_data(const std::vector<int16_t>& samples,
                           const std::string& session_id = "",
                           const std::string& language = "");

private:
    /**
     * @struct audio_packet
     * @brief Structure representing a packet of audio data
     */
    struct audio_packet {
        /// Float audio samples
        std::vector<float> samples;

        /// Session identifier
        std::string session_id;

        /// Language hint
        std::string language;

        /// Timestamp when packet was received
        uint64_t timestamp;
    };

    /// Queue for incoming audio packets
    moodycamel::ConcurrentQueue<audio_packet> m_audio_queue;

    /// Flag indicating if audio reception is active
    std::atomic<bool> m_active{false};

    /// Current session identifier
    std::string m_current_session_id;

    /// Current language hint
    std::string m_current_language;

    /// Time of last received audio packet
    std::chrono::steady_clock::time_point m_last_packet_time;

    /// Timeout for considering source inactive (ms)
    static constexpr int ACTIVITY_TIMEOUT_MS = 5000;
};
