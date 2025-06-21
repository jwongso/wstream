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

#include "wstream_app.h"
#include "websocket_server.h"
#include "audio_processor.h"
#include "whisper_engine.h"
#include "text_processor.h"
#include <iostream>
#include <chrono>
#include <csignal>
#include <cstdlib>

// Global shutdown flag
std::atomic<bool> g_shutdown_requested{false};

// Static flag for signal handler
static std::atomic<bool> s_sigint_received{false};

// Signal handler
static void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        s_sigint_received = true;
        g_shutdown_requested = true;
    }
}

wstream_app::wstream_app(const std::string& model_path)
    : m_model_path(model_path) {
}

wstream_app::~wstream_app() {
    shutdown();
}

bool wstream_app::initialize(int argc, char* argv[]) {
    // Install signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);
#endif

    if (argc > 1) {
        std::string user_path = argv[1];
        if (validate_model_path(user_path)) {
            m_model_path = user_path;
        } else {
            std::cerr << "Warning: Provided model path '" << user_path
                      << "' does not exist. Using default.\n";
        }
    }

    m_whisper_engine = std::make_unique<whisper_engine>(m_model_path);
    if (!m_whisper_engine->initialize()) {
        std::cerr << "Failed to initialize Whisper engine.\n";
        return false;
    }

    m_audio_processor = std::make_unique<audio_processor>();
    if (!m_audio_processor->initialize()) {
        std::cerr << "Failed to initialize audio processor.\n";
        return false;
    }

    m_text_processor = std::make_unique<text_processor>();
    m_websocket_server = std::make_unique<websocket_server>();
    m_websocket_server->start();

    return true;
}

bool wstream_app::validate_model_path(const std::string& path) {
    return fs::exists(path);
}

void wstream_app::run() {
    m_audio_processor->resume();
    process_audio_loop();
}

void wstream_app::process_audio_loop() {
    std::vector<float> audio_samples;

    while (m_is_running) {
        // Check SDL events (window close)
        m_is_running = sdl_poll_events();

        // Also check for SIGINT
        if (s_sigint_received) {
            std::cout << "\nShutdown requested (CTRL-C)..." << std::endl;
            m_is_running = false;
        }

        if (!m_is_running) {
            g_shutdown_requested = true;
            break;
        }

        if (!m_audio_processor->get_processed_samples(audio_samples)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (audio_samples.empty()) continue;

        std::string transcription = m_whisper_engine->transcribe(audio_samples);

        if (!transcription.empty()) {
            std::string processed_text = m_text_processor->process(transcription);

            if (!processed_text.empty()) {
                std::cout << processed_text << std::endl;
                m_websocket_server->queue_transcription(processed_text);
            }
        }
    }
}

void wstream_app::shutdown() {
    if (m_audio_processor) {
        m_audio_processor->pause();
    }

    if (m_websocket_server) {
        m_websocket_server->stop();
    }

    SDL_Quit();
    std::cout << "Shutting down..." << std::endl;
    m_is_running = false;
    g_shutdown_requested = true;

    // Force exit after a short delay if we're still here
    std::thread([]{
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::exit(0);
    }).detach();
}
