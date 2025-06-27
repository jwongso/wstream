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

#include "audio_source_factory.h"
#include "audio_processor.h"
#include "websocket_audio_source.h"
#include <iostream>
#include <algorithm>
#include <cctype>

std::unique_ptr<audio_source> audio_source_factory::create(audio_source_type type) {
    switch (type) {
    case audio_source_type::SDL_MICROPHONE: {
        std::cout << "[Factory] Creating SDL microphone audio source..." << std::endl;
        auto source = std::make_unique<audio_processor>();
        if (source->initialize()) {
            std::cout << "[Factory] SDL microphone audio source created successfully" << std::endl;
            return source;
        } else {
            std::cerr << "[Factory] Failed to initialize SDL microphone audio source" << std::endl;
            return nullptr;
        }
    }

    case audio_source_type::WEBSOCKET_CLIENT: {
        std::cout << "[Factory] Creating WebSocket client audio source..." << std::endl;
        auto source = std::make_unique<websocket_audio_source>();
        if (source->initialize()) {
            std::cout << "[Factory] WebSocket client audio source created successfully" << std::endl;
            return source;
        } else {
            std::cerr << "[Factory] Failed to initialize WebSocket client audio source" << std::endl;
            return nullptr;
        }
    }

    default:
        std::cerr << "[Factory] Unknown audio source type: " << static_cast<int>(type) << std::endl;
        return nullptr;
    }
}

std::string audio_source_factory::get_type_name(audio_source_type type) {
    switch (type) {
    case audio_source_type::SDL_MICROPHONE:
        return "SDL Microphone";
    case audio_source_type::WEBSOCKET_CLIENT:
        return "WebSocket Client";
    default:
        return "Unknown";
    }
}

audio_source_type audio_source_factory::parse_type(const std::string& type_str) {
    // Convert to lowercase for case-insensitive comparison
    std::string lower_str = type_str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lower_str == "microphone" || lower_str == "sdl" || lower_str == "mic") {
        return audio_source_type::SDL_MICROPHONE;
    } else if (lower_str == "websocket" || lower_str == "ws" || lower_str == "client") {
        return audio_source_type::WEBSOCKET_CLIENT;
    }

    std::cerr << "[Factory] Unknown audio source type string: '" << type_str
              << "'. Defaulting to SDL_MICROPHONE" << std::endl;
    return audio_source_type::SDL_MICROPHONE; // Default fallback
}

std::string audio_source_factory::type_to_string(audio_source_type type) {
    switch (type) {
    case audio_source_type::SDL_MICROPHONE:
        return "microphone";
    case audio_source_type::WEBSOCKET_CLIENT:
        return "websocket";
    default:
        return "unknown";
    }
}

std::vector<audio_source_type> audio_source_factory::get_available_types() {
    return {
        audio_source_type::SDL_MICROPHONE,
        audio_source_type::WEBSOCKET_CLIENT
    };
}

bool audio_source_factory::is_type_supported(audio_source_type type) {
    const auto available = get_available_types();
    return std::find(available.begin(), available.end(), type) != available.end();
}
