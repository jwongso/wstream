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
#include <memory>
#include <string>

/**
 * @enum audio_source_type
 * @brief Types of audio sources available for server-side deployment
 */
enum class audio_source_type {
    SDL_MICROPHONE,   ///< Local microphone via SDL2
    WEBSOCKET_CLIENT, ///< Remote client via WebSocket
    BENCHMARK
};

/**
 * @class audio_source_factory
 * @brief Factory for creating audio source instances
 *
 * Creates the appropriate audio source based on the deployment context.
 * Only one audio source type is active at a time, determined at startup.
 *
 * The factory pattern provides clean separation of concerns and makes it
 * easy to add new audio source types in the future without modifying
 * existing code.
 */
class audio_source_factory {
public:
    /**
     * @brief Creates an audio source of the specified type
     * @param type Type of audio source to create
     * @return Unique pointer to the created audio source, nullptr on failure
     *
     * The factory initializes the audio source and returns it ready for use.
     * If initialization fails, nullptr is returned.
     */
    static std::unique_ptr<audio_source> create(audio_source_type type);

    /**
     * @brief Gets the human-readable name of an audio source type
     * @param type Audio source type
     * @return Human-readable name of the audio source type
     */
    static std::string get_type_name(audio_source_type type);

    /**
     * @brief Parses audio source type from string representation
     * @param type_str String representation of audio source type
     * @return Audio source type, or SDL_MICROPHONE if invalid string
     *
     * Supported strings:
     * - "microphone", "sdl", "mic" -> SDL_MICROPHONE
     * - "websocket", "ws", "client" -> WEBSOCKET_CLIENT
     */
    static audio_source_type parse_type(const std::string& type_str);

    /**
     * @brief Gets string representation of audio source type
     * @param type Audio source type
     * @return String representation suitable for command line or config files
     */
    static std::string type_to_string(audio_source_type type);

    /**
     * @brief Gets all available audio source types
     * @return Vector of all available audio source types
     */
    static std::vector<audio_source_type> get_available_types();

    /**
     * @brief Validates if an audio source type is supported
     * @param type Audio source type to validate
     * @return true if the type is supported, false otherwise
     */
    static bool is_type_supported(audio_source_type type);
};
