// -------------------------------------------------------------------------------------------------
//
// Copyright (C) all of the contributors. All rights reserved.
//
//
// This software, including documentation, is protected by copyright controlled by
// contributors. All rights are reserved. Copying, including reproducing, storing,
// adapting or translating, any or all of this material requires the prior written
// consent of all contributors.
//
// -------------------------------------------------------------------------------------------------

#include "wstream_app.h"
#include "whisper_engine.h"
#include "text_processor.h"
#include "websocket_audio_source.h"
#include <iostream>
#include <chrono>
#include <csignal>
#include <cstdlib>

// Global shutdown flag
std::atomic<bool> g_shutdown_requested{false};

// Static flag for signal handler
static std::atomic<bool> s_sigint_received{false};

// Static instance pointer for signal handler
wstream_app* wstream_app::s_instance = nullptr;

// Signal handler
static void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        s_sigint_received = true;
        g_shutdown_requested = true;

        if (wstream_app::s_instance) {
            std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
        }
    }
}

wstream_app::wstream_app(const std::string& model_path,
                         audio_source_type source_type,
                         uint16_t websocket_port)
    : m_model_path(model_path), m_websocket_port(websocket_port), m_audio_source_type(source_type) {
    s_instance = this;
}

wstream_app::~wstream_app() {
    shutdown();
    s_instance = nullptr;
}

bool wstream_app::initialize(int argc, char* argv[]) {
    // Install signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);
#endif

    // Parse command line arguments
    if (!parse_command_line(argc, argv)) {
        return false;
    }

    std::cout << "---> Initializing wstream with audio source: "
              << get_audio_source_name() << std::endl;

    // Validate model path
    if (!validate_model_path(m_model_path)) {
        std::cerr << "Model file not found: " << m_model_path << std::endl;
        return false;
    }

    // Initialize Whisper engine
    m_whisper_engine = std::make_unique<whisper_engine>(m_model_path);
    if (!m_whisper_engine->initialize()) {
        std::cerr << "Failed to initialize Whisper engine." << std::endl;
        return false;
    }

    // Initialize text processor
    m_text_processor = std::make_unique<text_processor>();

    // Create the audio source using factory
    m_audio_source = audio_source_factory::create(m_audio_source_type);
    if (!m_audio_source) {
        std::cerr << "Failed to create audio source: " << get_audio_source_name() << std::endl;
        return false;
    }

    // Store typed reference for WebSocket audio source
    if (m_audio_source_type == audio_source_type::WEBSOCKET_CLIENT) {
        m_websocket_audio_source = static_cast<websocket_audio_source*>(m_audio_source.get());
    }

    // Setup WebSocket server
    if (!setup_websocket_server()) {
        std::cerr << "Failed to setup WebSocket server." << std::endl;
        return false;
    }

    std::cout << "<--- wstream initialized successfully." << std::endl;
    return true;
}

bool wstream_app::parse_command_line(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            show_help(argv[0]);
            return false;
        } else if (arg == "--audio-source" && i + 1 < argc) {
            m_audio_source_type = audio_source_factory::parse_type(argv[++i]);
        } else if (arg == "--port" && i + 1 < argc) {
            try {
                m_websocket_port = static_cast<uint16_t>(std::stoi(argv[++i]));
            } catch (const std::exception& e) {
                std::cerr << "Invalid port number: " << argv[i] << std::endl;
                return false;
            }
        } else if (arg.find("--") == 0) {
            std::cerr << "Unknown option: " << arg << std::endl;
            show_help(argv[0]);
            return false;
        } else {
            // Assume it's a model path
            m_model_path = arg;
        }
    }

    return true;
}

void wstream_app::show_help(const std::string& program_name) {
    std::cout << "Usage: " << program_name << " [options] [model_path]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --audio-source <type>  Audio source type (microphone, websocket) [default: microphone]\n";
    std::cout << "  --port <port>          WebSocket server port [default: " << DEFAULT_WEBSOCKET_PORT << "]\n";
    std::cout << "  --help, -h             Show this help message\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  model_path             Path to Whisper model file [default: " << DEFAULT_MODEL_PATH << "]\n\n";
    std::cout << "Audio source types:\n";
    std::cout << "  microphone, sdl, mic   Use local microphone via SDL2\n";
    std::cout << "  websocket, ws, client  Receive audio from WebSocket clients\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << "                                    # Use default settings\n";
    std::cout << "  " << program_name << " --audio-source websocket           # Use WebSocket audio\n";
    std::cout << "  " << program_name << " --port 9090 models/base.en.bin     # Custom port and model\n";
}

bool wstream_app::validate_model_path(const std::string& path) {
    return fs::exists(path) && fs::is_regular_file(path);
}

bool wstream_app::setup_websocket_server() {
    m_websocket_server = std::make_unique<websocket_server>(m_websocket_port);

    // Set up audio callback for WebSocket audio source
    if (m_audio_source_type == audio_source_type::WEBSOCKET_CLIENT) {
        setup_websocket_audio_callback();
    }

    // Set up command handler for client commands
    m_websocket_server->set_command_handler(
        [this](const std::string& command, const nlohmann::json& params,
               websocket::stream<tcp::socket>* client_ws) -> nlohmann::json {
            nlohmann::json response;
            response["type"] = "response";
            response["action"] = command;

            if (command == "set_audio_source") {
                // Handle set_audio_source command
                if (!params.contains("source") || !params["source"].is_string()) {
                    response["status"] = "error";
                    response["message"] = "Missing or invalid 'source' parameter";
                } else {
                    std::string source_str = params["source"].get<std::string>();
                    audio_source_type new_source_type = audio_source_factory::parse_type(source_str);

                    if (!audio_source_factory::is_type_supported(new_source_type)) {
                        response["status"] = "error";
                        response["message"] = "Unsupported audio source type: " + source_str;
                    } else {
                        bool success = set_audio_source_runtime(new_source_type);
                        response["status"] = success ? "success" : "error";
                        response["source"] = source_str;
                        response["audio_source"] =
                            audio_source_factory::type_to_string(m_audio_source_type);
                        response["audio_source_name"] = get_audio_source_name();

                        if (success) {
                            response["message"] = "Audio source switched to " + source_str;
                        } else {
                            response["message"] = "Failed to switch audio source to " + source_str;
                        }
                    }
                }

            } else if (command == "get_status") {
                response["status"] = "success";
                response["audio_source"] = audio_source_factory::type_to_string(m_audio_source_type);
                response["audio_source_name"] = get_audio_source_name();
                response["is_running"] = is_running();
                response["client_count"] = m_websocket_server->get_client_count();

            } else if (command == "get_transcription") {
                response["status"] = "success";
                response["transcription"] = get_latest_transcription();

            } else {
                response["status"] = "error";
                response["message"] = "Unknown command: " + command;
            }

            return response;
        }
    );

    m_websocket_server->start();
    return true;
}

void wstream_app::run() {
    // Start the audio source
    if (!m_audio_source->start()) {
        std::cerr << "Failed to start audio source: " << get_audio_source_name() << std::endl;
        return;
    }

    std::cout << "WStream is running. Audio source: " << get_audio_source_name() << std::endl;
    std::cout << "WebSocket server listening on port " << m_websocket_port << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;

    process_audio_loop();
}

void wstream_app::process_audio_loop() {
    std::vector<float> audio_samples;
    int loop_count = 0;
    int audio_received_count = 0;
    int whisper_call_count = 0;
    int transcription_count = 0;

    while (m_is_running) {
        loop_count++;

        // Check SDL events (window close)
        m_is_running = sdl_poll_events();

        // Check for SIGINT
        if (s_sigint_received) {
            std::cout << "\nShutdown requested (CTRL-C)..." << std::endl;
            m_is_running = false;
        }

        if (!m_is_running) {
            g_shutdown_requested = true;
            break;
        }

        if (m_switching_source) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Safely get audio from the active source
        bool got_audio = false;
        std::string session_id;
        std::string language;

        {
            std::lock_guard<std::mutex> lock(m_audio_source_mutex);
            if (m_audio_source && !m_switching_source) {
                got_audio = m_audio_source->get_audio_samples(audio_samples);
                if (got_audio && !audio_samples.empty()) {
                    session_id = m_audio_source->get_session_id();
                    language = m_audio_source->get_language();
                    audio_received_count++;

                    std::cout << "[PROCESS_AUDIO] Retrieved audio chunk #" << audio_received_count
                              << " with " << audio_samples.size() << " samples"
                              << " (duration: " << (audio_samples.size() * 1000.0 / 16000.0) << " ms)"
                              << std::endl;

                    // Check audio levels
                    float max_val = 0.0f;
                    float min_val = 0.0f;
                    float sum = 0.0f;
                    for (const auto& sample : audio_samples) {
                        if (sample > max_val) max_val = sample;
                        if (sample < min_val) min_val = sample;
                        sum += std::abs(sample);
                    }
                    float avg = sum / audio_samples.size();

                    std::cout << "[PROCESS_AUDIO] Audio levels - Min: " << min_val
                              << ", Max: " << max_val
                              << ", Avg: " << avg << std::endl;

                    if (avg < 0.001f) {
                        std::cout << "[PROCESS_AUDIO] WARNING: Audio appears to be silence!" << std::endl;
                    }
                }
            }
        }

        // Debug log every 1000 loops if no audio
        if (loop_count % 1000 == 0 && audio_received_count == 0) {
            std::cout << "[PROCESS_AUDIO] No audio received yet. Loop count: " << loop_count << std::endl;

            // Check if audio source is active
            if (m_audio_source) {
                std::cout << "[PROCESS_AUDIO] Audio source '" << m_audio_source->get_name()
                << "' is " << (m_audio_source->is_active() ? "active" : "inactive") << std::endl;
            }
        }

        if (!got_audio || audio_samples.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Process with whisper
        whisper_call_count++;
        std::cout << "[PROCESS_AUDIO] Sending chunk #" << whisper_call_count
                  << " to Whisper (" << audio_samples.size() << " samples)" << std::endl;

        std::string transcription = m_whisper_engine->transcribe(audio_samples);

        if (!transcription.empty()) {
            transcription_count++;
            std::cout << "[PROCESS_AUDIO] Whisper returned transcription #" << transcription_count
                      << ": '" << transcription << "'" << std::endl;

            std::string processed_text = m_text_processor->process(transcription);

            if (!processed_text.empty()) {
                std::cout << "[PROCESS_AUDIO] Text processor output: '" << processed_text << "'" << std::endl;
                std::cout << "[" << get_audio_source_name() << "] " << processed_text << std::endl;

                // Update latest transcription
                update_latest_transcription(processed_text);

                // Broadcast to WebSocket clients
                if (!session_id.empty()) {
                    m_websocket_server->queue_transcription(processed_text, session_id);
                } else {
                    m_websocket_server->queue_transcription(processed_text);
                }
            } else {
                std::cout << "[PROCESS_AUDIO] Text processor returned empty string" << std::endl;
            }
        } else {
            std::cout << "[PROCESS_AUDIO] Whisper returned empty transcription for chunk #"
                      << whisper_call_count << std::endl;
        }

        audio_samples.clear();
    }

    std::cout << "[PROCESS_AUDIO] Exiting audio loop. Stats:" << std::endl;
    std::cout << "  Total loops: " << loop_count << std::endl;
    std::cout << "  Audio chunks received: " << audio_received_count << std::endl;
    std::cout << "  Whisper calls: " << whisper_call_count << std::endl;
    std::cout << "  Transcriptions: " << transcription_count << std::endl;
}

std::string wstream_app::get_audio_source_name() const {
    return audio_source_factory::get_type_name(m_audio_source_type);
}

bool wstream_app::set_audio_source_runtime(audio_source_type source_type) {
    if (source_type == m_audio_source_type) {
        // Already using this source type
        return true;
    }

    std::cout << "Switching audio source from " << get_audio_source_name()
              << " to " << audio_source_factory::get_type_name(source_type) << std::endl;

    return switch_audio_source(source_type);
}

bool wstream_app::switch_audio_source(audio_source_type new_source_type) {
    // Set switching flag to pause the main loop
    m_switching_source = true;

    // Wait a bit for the main loop to stop accessing the audio source
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Now safe to switch
    std::lock_guard<std::mutex> lock(m_audio_source_mutex);
    // Stop current audio source
    if (m_audio_source) {
        m_audio_source->stop();
    }

    // Create new audio source
    auto new_audio_source = create_audio_source(new_source_type);
    if (!new_audio_source) {
        std::cerr << "Failed to create new audio source: "
                  << audio_source_factory::get_type_name(new_source_type) << std::endl;

        // Try to restart the old source
        if (m_audio_source) {
            m_audio_source->start();
        }
        return false;
    }

    // Start new audio source
    if (!new_audio_source->start()) {
        std::cerr << "Failed to start new audio source: "
                  << audio_source_factory::get_type_name(new_source_type) << std::endl;

        // Try to restart the old source
        if (m_audio_source) {
            m_audio_source->start();
        }
        return false;
    }

    // Clear the old source first (this will destroy it)
    m_audio_source.reset();
    // Small delay to ensure destruction is complete
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Switch to new source
    m_audio_source = std::move(new_audio_source);
    m_audio_source_type = new_source_type;

    // Clear the switching flag
    m_switching_source = false;

    // Update typed references
    m_websocket_audio_source = nullptr;
    if (m_audio_source_type == audio_source_type::WEBSOCKET_CLIENT) {
        m_websocket_audio_source = dynamic_cast<websocket_audio_source*>(m_audio_source.get());
        setup_websocket_audio_callback();

        // Enable audio dump for debugging
        // if (m_websocket_audio_source) {
        //     m_websocket_audio_source->enable_audio_dump("websocket_audio_debug.raw");
        //     std::cout << "[DEBUG] Audio dumping enabled for WebSocket source" << std::endl;
        //     std::cout << "[DEBUG] You can play the dump file with:" << std::endl;
        //     std::cout << "  ffplay -f s16le -ar 16000 -ac 1 websocket_audio_debug.raw" << std::endl;
        // }
    }

    std::cout << "Successfully switched to audio source: " << get_audio_source_name() << std::endl;
    return true;
}

std::unique_ptr<audio_source> wstream_app::create_audio_source(audio_source_type source_type) {
    auto new_source = audio_source_factory::create(source_type);
    if (!new_source) {
        return nullptr;
    }

    // For WebSocket source, we need to update the WebSocket server callback
    if (source_type == audio_source_type::WEBSOCKET_CLIENT && m_websocket_server) {
        // The audio callback will be automatically routed to the new WebSocket source
        // when handle_websocket_audio is called, since we update m_websocket_audio_source
        // in switch_audio_source
    }

    return new_source;
}

void wstream_app::handle_websocket_audio(const std::vector<int16_t>& samples,
                                         const std::string& session_id,
                                         const std::string& language) {
    std::lock_guard<std::mutex> lock(m_audio_source_mutex);
    if (m_websocket_audio_source) {
        m_websocket_audio_source->handle_audio_data(samples, session_id, language);
    }
}

std::string wstream_app::get_latest_transcription() {
    std::lock_guard<std::mutex> lock(m_transcription_mutex);
    std::string result = m_latest_transcription;
    m_latest_transcription.clear(); // Clear after reading
    return result;
}

void wstream_app::update_latest_transcription(const std::string& transcription) {
    std::lock_guard<std::mutex> lock(m_transcription_mutex);
    m_latest_transcription = transcription;
}

void wstream_app::shutdown() {
    std::cout << "Shutting down WStream..." << std::endl;

    // Stop audio source
    if (m_audio_source) {
        m_audio_source->stop();
    }

    // Stop WebSocket server
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
