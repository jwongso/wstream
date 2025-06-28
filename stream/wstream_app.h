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
#include "audio_source.h"
#include "audio_source_factory.h"
#include "websocket_server.h"
#include <memory>
#include <atomic>
#include <string>
#include <vector>
#include <thread>
#include <filesystem>
#include <mutex>

namespace fs = std::filesystem;

// Forward declarations
class websocket_server;
class whisper_engine;
class text_processor;
class websocket_audio_source;

/**
 * @class wstream_app
 * @brief Main application class that coordinates all components for real-time audio transcription
 *
 * This class serves as the central coordinator for the WStream application, managing:
 * - Single audio source (SDL2 microphone or WebSocket clients) via factory pattern
 * - Whisper-based speech recognition
 * - Text processing and filtering
 * - WebSocket server for real-time broadcasting and client communication
 *
 * The application uses a factory pattern to create the appropriate audio source at startup,
 * ensuring only the necessary components are instantiated and resources are used efficiently.
 *
 * @par Usage Example:
 * @code
 * wstream_app app("path/to/model.bin", audio_source_type::WEBSOCKET_CLIENT);
 * if (app.initialize(argc, argv)) {
 *     app.run();
 * }
 * @endcode
 */
class wstream_app {
public:
    /// Default model path for Whisper
    static constexpr const char* DEFAULT_MODEL_PATH = "models/ggml-small.en-q5_1.bin";

    /// Default WebSocket server port
    static constexpr uint16_t DEFAULT_WEBSOCKET_PORT = 8080;

    /// Static pointer to current instance for signal handler
    static wstream_app* s_instance;

    /**
     * @brief Constructs wstream_app with specified audio source type
     * @param model_path Path to the Whisper model file (default: DEFAULT_MODEL_PATH)
     * @param source_type Type of audio source to use (default: SDL_MICROPHONE)
     * @param websocket_port Port for WebSocket server (default: DEFAULT_WEBSOCKET_PORT)
     */
    explicit wstream_app(const std::string& model_path = DEFAULT_MODEL_PATH,
                         audio_source_type source_type = audio_source_type::SDL_MICROPHONE,
                         uint16_t websocket_port = DEFAULT_WEBSOCKET_PORT);

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
     * @param argv Command line arguments
     * @return true if initialization successful, false otherwise
     *
     * Command line arguments:
     * - argv[1]: Optional custom model path
     * - --audio-source <type>: Audio source type (microphone, websocket)
     * - --port <port>: WebSocket server port
     * - --help: Display help information
     */
    bool initialize(int argc, char* argv[]);

    /**
     * @brief Starts the main application loop
     *
     * This method runs the main processing loop that:
     * - Captures audio samples from the configured source
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
     * - Exits the process after a timeout
     */
    void shutdown();

    /**
     * @brief Gets the active audio source type
     * @return Current audio source type
     */
    audio_source_type get_audio_source_type() const { return m_audio_source_type; }

    /**
     * @brief Gets the active audio source name
     * @return Human-readable name of the current audio source
     */
    std::string get_audio_source_name() const;

    /**
     * @brief Sets the audio source type at runtime
     * @param source_type New audio source type to use
     * @return true if switch successful, false otherwise
     *
     * Note: This creates a new audio source and replaces the current one.
     * Any ongoing audio processing will be interrupted.
     */
    bool set_audio_source_runtime(audio_source_type source_type);

    /**
     * @brief Handles incoming audio data from WebSocket clients
     * @param samples PCM audio samples (16-bit)
     * @param session_id Client session identifier
     * @param language Language hint (optional)
     *
     * This method is called by the WebSocket server when audio data
     * is received from a client. Only functional when using WEBSOCKET_CLIENT
     * audio source type.
     */
    void handle_websocket_audio(const std::vector<int16_t>& samples,
                                const std::string& session_id = "",
                                const std::string& language = "");

    /**
     * @brief Gets the latest transcription result
     * @return Latest transcription text, empty if none available
     *
     * This method is thread-safe and can be called from external threads
     * to retrieve the most recent transcription result.
     */
    std::string get_latest_transcription();

    /**
     * @brief Checks if the application is currently running
     * @return true if running, false otherwise
     */
    bool is_running() const { return m_is_running; }

private:
    /// Atomic flag controlling the main application loop
    std::atomic<bool> m_is_running{true};

    /// WebSocket server for broadcasting transcriptions and receiving audio
    std::unique_ptr<websocket_server> m_websocket_server;

    /// Whisper speech recognition engine
    std::unique_ptr<whisper_engine> m_whisper_engine;

    /// Text processing and filtering component
    std::unique_ptr<text_processor> m_text_processor;

    /// Path to the Whisper model file
    std::string m_model_path;

    /// WebSocket server port
    uint16_t m_websocket_port;

    /// Audio source type being used
    audio_source_type m_audio_source_type;

    /// The single active audio source
    std::unique_ptr<audio_source> m_audio_source;

    /// WebSocket audio source reference (if using WebSocket)
    websocket_audio_source* m_websocket_audio_source = nullptr;

    /// Latest transcription result (thread-safe access)
    std::string m_latest_transcription;
    std::mutex m_transcription_mutex;

    /// Switching audio source mutex
    mutable std::mutex m_audio_source_mutex;
    std::atomic<bool> m_switching_source{false};

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
     * - Retrieves processed audio samples from the active source
     * - Runs speech recognition
     * - Processes and filters text
     * - Queues results for broadcasting
     */
    void process_audio_loop();

    /**
     * @brief Parses command line arguments
     * @param argc Argument count
     * @param argv Argument values
     * @return true if parsing successful, false if help was requested or error occurred
     */
    bool parse_command_line(int argc, char* argv[]);

    /**
     * @brief Displays help information
     * @param program_name Name of the program executable
     */
    void show_help(const std::string& program_name);

    /**
     * @brief Sets up WebSocket server with appropriate callbacks
     * @return true if setup successful, false otherwise
     */
    bool setup_websocket_server();

    /**
     * @brief Updates the latest transcription in a thread-safe manner
     * @param transcription New transcription text
     */
    void update_latest_transcription(const std::string& transcription);

    /**
     * @brief Safely switches to a new audio source
     * @param new_source_type The new audio source type
     * @return true if successful, false otherwise
     */
    bool switch_audio_source(audio_source_type new_source_type);

    /**
     * @brief Creates and initializes a new audio source
     * @param source_type Type of audio source to create
     * @return Unique pointer to created audio source, nullptr on failure
     */
    std::unique_ptr<audio_source> create_audio_source(audio_source_type source_type);

    void setup_websocket_audio_callback() {
        if (m_websocket_server) {
            m_websocket_server->set_audio_callback(
                [this](const audio_data& audio, websocket::stream<tcp::socket>* client_ws) {
                    handle_websocket_audio(audio.samples, audio.session_id, audio.language);

                    // Send acknowledgment back to client
                    if (client_ws) {
                        nlohmann::json ack_response;
                        ack_response["type"] = "audio_ack";
                        ack_response["status"] = "received";
                        ack_response["samples_count"] = audio.samples.size();
                        ack_response["session_id"] = audio.session_id;

                        m_websocket_server->send_to_client(client_ws, ack_response);
                    }
                }
            );
        }
    }
};
