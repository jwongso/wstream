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

#include <vector>
#include <string>
#include <atomic>
#include <memory>

/**
 * @class audio_source
 * @brief Abstract interface for audio data sources
 *
 * This class defines a common interface for different audio sources
 * (SDL2 microphone, WebSocket client, file, etc.) to provide audio samples
 * to the whisper engine.
 */
class audio_source {
public:
    /**
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~audio_source() = default;

    /**
     * @brief Initializes the audio source
     * @return true if initialization successful, false otherwise
     */
    virtual bool initialize() = 0;

    /**
     * @brief Starts audio capture/reception
     * @return true if started successfully, false otherwise
     */
    virtual bool start() = 0;

    /**
     * @brief Stops audio capture/reception
     */
    virtual void stop() = 0;

    /**
     * @brief Retrieves processed audio samples
     * @param[out] samples Vector to store the retrieved audio samples
     * @return true if samples were retrieved, false if no samples available
     */
    virtual bool get_audio_samples(std::vector<float>& samples) = 0;

    /**
     * @brief Gets the name/identifier of this audio source
     * @return String identifier for this audio source
     */
    virtual std::string get_name() const = 0;

    /**
     * @brief Checks if the audio source is active
     * @return true if active and providing samples, false otherwise
     */
    virtual bool is_active() const = 0;

    /**
     * @brief Gets session ID for this audio source (if applicable)
     * @return Session ID string, empty if not applicable
     */
    virtual std::string get_session_id() const { return ""; }

    /**
     * @brief Gets language hint for this audio source (if applicable)
     * @return Language code string, empty if not applicable
     */
    virtual std::string get_language() const { return ""; }
};
