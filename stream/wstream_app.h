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

#include "whisper.h"
#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "concurrentqueue.h"
#include <memory>
#include <atomic>
#include <string>
#include <vector>
#include <thread>
#include <filesystem>

namespace fs = std::filesystem;

// Forward declarations
class websocket_server;
class audio_processor;
class whisper_engine;
class text_processor;

/**
 * @class wstream_app
 * @brief Main application class that coordinates all components for real-time audio transcription
 *
 * This class serves as the central coordinator for the WStream application, managing:
 * - Audio capture and processing
 * - Whisper-based speech recognition
 * - Text processing and filtering
 * - WebSocket server for real-time broadcasting
 *
 * The application follows a modular design where each component has specific responsibilities
 * and communicates through well-defined interfaces.
 *
 * @par Usage Example:
 * @code
 * wstream_app app("path/to/model.bin");
 * if (app.initialize(argc, argv)) {
 *     app.run();
 * }
 * @endcode
 */
class wstream_app {
public:
    /// Default model path for Whisper
    static constexpr const char* DEFAULT_MODEL_PATH = "models/ggml-small.en-q5_1.bin";

    /**
     * @brief Constructs a new wstream_app object
     * @param model_path Path to the Whisper model file (default: DEFAULT_MODEL_PATH)
     */
    explicit wstream_app(const std::string& model_path = DEFAULT_MODEL_PATH);

    /**
     * @brief Destructor - ensures clean shutdown of all components
     */
    ~wstream_app();

    // Non-copyable, non-movable to ensure single instance
    wstream_app(const wstream_app&) = delete;
    wstream_app& operator=(const wstream_app&) = delete;
    wstream_app(wstream_app&&) = delete;
    wstream_app& operator=(wstream_app&&) = delete;

    /**
     * @brief Initializes all application components
     * @param argc Command line argument count
     * @param argv Command line arguments (argv[1] can specify custom model path)
     * @return true if initialization successful, false otherwise
     *
     * This method:
     * - Validates and sets the model path
     * - Initializes the Whisper engine
     * - Sets up audio processing
     * - Starts the WebSocket server
     * - Configures text processing
     */
    bool initialize(int argc, char* argv[]);

    /**
     * @brief Starts the main application loop
     *
     * This method runs the main processing loop that:
     * - Captures audio samples
     * - Processes them through Whisper
     * - Filters and cleans the transcribed text
     * - Broadcasts results to connected WebSocket clients
     *
     * The loop continues until shutdown is requested via SDL events or system signals.
     */
    void run();

    /**
     * @brief Gracefully shuts down all application components
     *
     * Performs cleanup in the correct order:
     * - Stops audio capture
     * - Shuts down WebSocket server
     * - Releases Whisper resources
     * - Cleans up SDL
     */
    void shutdown();

private:
    /// Atomic flag controlling the main application loop
    std::atomic<bool> m_is_running{true};

    /// WebSocket server for broadcasting transcriptions
    std::unique_ptr<websocket_server> m_websocket_server;

    /// Audio capture and preprocessing component
    std::unique_ptr<audio_processor> m_audio_processor;

    /// Whisper speech recognition engine
    std::unique_ptr<whisper_engine> m_whisper_engine;

    /// Text processing and filtering component
    std::unique_ptr<text_processor> m_text_processor;

    /// Path to the Whisper model file
    std::string m_model_path;

    /// Static pointer to current instance for signal handler
    static wstream_app* s_instance;

    /**
     * @brief Validates that a model file exists and is accessible
     * @param path Path to validate
     * @return true if path exists and is readable, false otherwise
     */
    bool validate_model_path(const std::string& path);

    /**
     * @brief Main audio processing loop
     *
     * Continuously:
     * - Retrieves processed audio samples
     * - Runs speech recognition
     * - Processes and filters text
     * - Queues results for broadcasting
     */
    void process_audio_loop();
};
