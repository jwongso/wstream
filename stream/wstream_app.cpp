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
#include "benchmark_audio_source.h"
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

// Helper function to extract reference segment
std::string extract_reference_segment(const std::string& full_reference,
                                      size_t& position,
                                      size_t approximate_length) {
    if (position >= full_reference.length()) {
        return "";
    }

    // Find a reasonable segment ending (at word boundary)
    size_t end_pos = position + approximate_length;
    if (end_pos < full_reference.length()) {
        // Find next word boundary
        while (end_pos < full_reference.length() && !std::isspace(full_reference[end_pos])) {
            end_pos++;
        }
    } else {
        end_pos = full_reference.length();
    }

    std::string segment = full_reference.substr(position, end_pos - position);
    position = end_pos;

    // Skip leading whitespace for next iteration
    while (position < full_reference.length() && std::isspace(full_reference[position])) {
        position++;
    }

    return segment;
}

// Simple word difference marker
std::string mark_word_differences(const std::string& hypothesis, const std::string& reference) {
    std::istringstream hyp_stream(hypothesis);
    std::istringstream ref_stream(reference);
    std::vector<std::string> hyp_words, ref_words;
    std::string word;

    while (hyp_stream >> word) hyp_words.push_back(word);
    while (ref_stream >> word) ref_words.push_back(word);

    std::stringstream result;
    size_t ref_idx = 0;

    for (size_t i = 0; i < hyp_words.size(); ++i) {
        if (i > 0) result << " ";

        // Normalize for comparison (lowercase, remove punctuation)
        auto normalize = [](const std::string& w) {
            std::string n = w;
            std::transform(n.begin(), n.end(), n.begin(), ::tolower);
            n.erase(std::remove_if(n.begin(), n.end(),
                                   [](char c) { return !std::isalnum(c); }),
                    n.end());
            return n;
        };

        bool found = false;
        // Look for matching word in reference (with small window)
        for (size_t j = ref_idx; j < std::min(ref_idx + 3, ref_words.size()); ++j) {
            if (normalize(hyp_words[i]) == normalize(ref_words[j])) {
                result << hyp_words[i];  // Correct word
                ref_idx = j + 1;
                found = true;
                break;
            }
        }

        if (!found) {
            // Mark as error
            if (ref_idx < ref_words.size()) {
                // Substitution - show what it should be
                result << "[" << hyp_words[i] << "]";
                ref_idx++;
            } else {
                // Insertion - extra word
                result << "[+" << hyp_words[i] << "]";
            }
        }
    }

    return result.str();
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

    // Setup WebSocket server
    if (!setup_websocket_server()) {
        std::cerr << "Failed to setup WebSocket server." << std::endl;
        return false;
    }

    // Handle benchmark mode if requested via command line
    if (m_benchmark_mode) {
        // Get benchmark WAV path from environment or use default
        std::string wav_path = "./benchmark.wav";
        if (const char* env_wav = std::getenv("WSTREAM_BENCHMARK_WAV")) {
            wav_path = env_wav;
        }

        // Start benchmark mode
        if (!start_benchmark(wav_path)) {
            std::cerr << "Failed to start benchmark mode." << std::endl;
            return false;
        }
    } else {
        // Normal mode - create audio source using factory
        m_audio_source = audio_source_factory::create(m_audio_source_type);
        if (!m_audio_source) {
            std::cerr << "Failed to create audio source: " << get_audio_source_name() << std::endl;
            return false;
        }

        // Store typed reference for WebSocket audio source
        if (m_audio_source_type == audio_source_type::WEBSOCKET_CLIENT) {
            m_websocket_audio_source = static_cast<websocket_audio_source*>(m_audio_source.get());
        }
    }

    std::cout << "<--- wstream initialized successfully." << std::endl;
    return true;
}

bool wstream_app::parse_command_line(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        m_benchmark_chunk_size_ms = 0;

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
        } else if (arg == "--benchmark") {
            m_benchmark_mode = true;
            m_audio_source_type = audio_source_type::BENCHMARK;  // Add this enum value
            std::cout << "Benchmark mode enabled" << std::endl;
        } else if (arg == "--chunk-size" && i + 1 < argc) {
            int chunk_ms = std::stoi(argv[++i]);
            if (chunk_ms >= 100 && chunk_ms <= 10000) {
                m_benchmark_chunk_size_ms = chunk_ms;
                std::cout << "Benchmark chunk size set to: " << chunk_ms << "ms" << std::endl;
            } else {
                std::cerr << "Invalid chunk size. Must be between 100-10000ms" << std::endl;
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
    std::cout << "  --benchmark            Run in benchmark mode with default WAV file (./benchmark.wav)\n";
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

            } /*else if (command == "benchmark") {
                // Handle benchmark command
                response["status"] = "success";
                response["action"] = "benchmark";

                // Get optional parameters
                std::string wav_path = "./benchmark.wav";
                bool real_time = true;

                if (params.contains("wav_file") && params["wav_file"].is_string()) {
                    wav_path = params["wav_file"].get<std::string>();
                }

                if (params.contains("real_time") && params["real_time"].is_boolean()) {
                    real_time = params["real_time"].get<bool>();
                }

                // Switch to benchmark audio source
                bool success = set_audio_source_runtime(audio_source_type::BENCHMARK);

                if (success) {
                    // Configure benchmark source if available
                    if (m_benchmark_audio_source) {
                        benchmark_audio_source::config config;
                        config.wav_file_path = wav_path;
                        config.real_time_simulation = real_time;

                        // Re-initialize with new config
                        m_benchmark_audio_source->stop();
                        // ... (code to reconfigure benchmark source)
                        m_benchmark_audio_source->start();
                    }

                    response["message"] = "Benchmark started with file: " + wav_path;
                    response["real_time"] = real_time;
                } else {
                    response["status"] = "error";
                    response["message"] = "Failed to start benchmark";
                }
            }*/ else if (command == "get_status") {
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
    // Start the audio source (unless already started by benchmark mode)
    if (!m_benchmark_mode && m_audio_source && !m_audio_source->is_active()) {
        if (!m_audio_source->start()) {
            std::cerr << "Failed to start audio source: " << get_audio_source_name() << std::endl;
            return;
        }
    }

    std::cout << "WStream is running. Audio source: " << get_audio_source_name() << std::endl;
    std::cout << "WebSocket server listening on port " << m_websocket_port << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;

    process_audio_loop();
}

void wstream_app::process_audio_loop() {
    std::vector<float> audio_samples;

    // Initialize transcription marker for benchmark mode
    transcription_marker marker;
    bool use_color_markers = true;  // Can be made configurable

    if (m_benchmark_mode && m_benchmark_manager) {
        std::string reference_text = m_benchmark_manager->get_reference_text();
        if (!reference_text.empty()) {
            marker.load_reference(reference_text);

            // Check if terminal supports colors
            if (!transcription_marker::is_color_supported()) {
                std::cout << "[Benchmark] Color output not supported, using brackets only" << std::endl;
                use_color_markers = false;
            } else {
                std::cout << "[Benchmark] Reference text loaded with color marking enabled" << std::endl;
            }
        }
    }

    // Start benchmark tracking if in benchmark mode
    if (m_benchmark_mode && m_benchmark_manager) {
        m_benchmark_manager->start();
    }

    while (m_is_running) {
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

        {
            std::lock_guard<std::mutex> lock(m_audio_source_mutex);
            if (m_audio_source && !m_switching_source) {
                got_audio = m_audio_source->get_audio_samples(audio_samples);
                if (got_audio && !audio_samples.empty()) {
                    session_id = m_audio_source->get_session_id();
                }
            }
        }

        if (!got_audio || audio_samples.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (m_benchmark_mode && m_playback_manager) {
            m_playback_manager->queue_audio(audio_samples);
        }

        // Process with whisper
        auto start_time = std::chrono::steady_clock::now();
        std::string transcription = m_whisper_engine->transcribe(audio_samples);
        auto end_time = std::chrono::steady_clock::now();

        // Calculate processing latency
        double processing_latency_ms = std::chrono::duration<double, std::milli>(
                                           end_time - start_time).count();

        if (!transcription.empty()) {
            std::string processed_text = m_text_processor->process(transcription);

            if (!processed_text.empty()) {
                // Display output with or without diff markers
                if (m_benchmark_mode && m_benchmark_manager) {
                    // Mark differences with colors
                    std::string marked_text;
                    if (use_color_markers) {
                        marked_text = marker.mark_differences_with_brackets(processed_text);
                    } else {
                        // Fallback to simple brackets without colors
                        marked_text = marker.mark_differences(processed_text);
                    }

                    std::cout << "[" << get_audio_source_name() << "] " << marked_text << std::endl;

                    // Add to benchmark manager with actual processing time
                    m_benchmark_manager->add_transcription(
                        processed_text,
                        0.95,  // Default confidence
                        audio_samples.size(),
                        processing_latency_ms
                        );
                } else {
                    // Normal output (non-benchmark mode)
                    std::cout << "[" << get_audio_source_name() << "] " << processed_text
                              << std::endl;
                }

                // Update latest transcription
                update_latest_transcription(processed_text);

                // Broadcast to WebSocket clients
                if (!session_id.empty()) {
                    m_websocket_server->queue_transcription(processed_text, session_id);
                } else {
                    m_websocket_server->queue_transcription(processed_text);
                }
            }
        }

        // Check if benchmark source is done
        if (m_benchmark_mode && m_audio_source_type == audio_source_type::BENCHMARK) {
            auto benchmark_source = dynamic_cast<benchmark_audio_source*>(m_audio_source.get());
            if (benchmark_source && benchmark_source->is_end_of_file()) {
                // Wait a moment for any final processing
                std::this_thread::sleep_for(std::chrono::seconds(1));
                stop_benchmark();
                break;  // Exit the loop after benchmark completes
            }
        }

        audio_samples.clear();
    }

    // If benchmark mode was active and we exited the loop, ensure it's stopped
    if (m_benchmark_mode) {
        stop_benchmark();
    }
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

    // Special handling for benchmark source
    // if (new_source_type == audio_source_type::BENCHMARK) {
    //     // Create benchmark manager if not already created
    //     if (!m_benchmark_manager) {
    //         m_benchmark_manager = std::make_unique<benchmark_manager>();
    //     }

    //     // Get reference text from benchmark audio source
    //     auto benchmark_source = dynamic_cast<benchmark_audio_source*>(new_audio_source.get());
    //     if (benchmark_source) {
    //         std::string ref_text = benchmark_source->get_reference_text();
    //         if (!ref_text.empty()) {
    //             m_benchmark_manager->set_reference_text(ref_text);
    //         }

    //         // Start benchmark tracking
    //         m_benchmark_manager->start();
    //     }
    // } else if (m_audio_source_type == audio_source_type::BENCHMARK) {
    //     // We're switching away from benchmark mode - finalize results
    //     if (m_benchmark_manager) {
    //         auto results = m_benchmark_manager->stop();
    //         m_benchmark_manager->export_results(results, "./benchmark_results.txt");
    //         std::cout << "Benchmark results exported to ./benchmark_results.txt" << std::endl;
    //     }
    //     // Turn off benchmark mode when switching away
    //     m_benchmark_mode = false;
    // }

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
    //m_benchmark_audio_source = nullptr;

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
    } /*else if (m_audio_source_type == audio_source_type::BENCHMARK) {
        m_benchmark_audio_source = dynamic_cast<benchmark_audio_source*>(m_audio_source.get());
    }*/

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

bool wstream_app::start_benchmark(const std::string& wav_path) {
    std::cout << "[Benchmark] Starting benchmark with WAV file: " << wav_path << std::endl;

    // Set environment variable for the factory to use
    setenv("WSTREAM_BENCHMARK_WAV", wav_path.c_str(), 1);

    // Create benchmark manager
    m_benchmark_manager = std::make_unique<benchmark_manager>();

    // Create benchmark audio source directly
    benchmark_audio_source::config config;
    config.wav_file_path = wav_path;
    config.reference_text_path = wav_path + ".txt";
    config.real_time_simulation = true;
    if (m_benchmark_chunk_size_ms >= 100 && m_benchmark_chunk_size_ms <= 10000) {
        config.chunk_size_ms = m_benchmark_chunk_size_ms;
    }

    auto benchmark_source = std::make_unique<benchmark_audio_source>(config);
    if (!benchmark_source->initialize()) {
        std::cerr << "[Benchmark] Failed to initialize benchmark audio source" << std::endl;
        m_benchmark_manager.reset();
        return false;
    }

    m_playback_manager = std::make_unique<audio_playback_manager>();
    if (!m_playback_manager->initialize()) {
        std::cerr << "[Benchmark] Warning: Failed to initialize audio playback" << std::endl;
        // Non-fatal, continue without playback
    } else {
        std::cout << "[Benchmark] Audio playback enabled for real-time evaluation" << std::endl;
    }

    // Get reference text
    std::string ref_text = benchmark_source->get_reference_text();
    if (!ref_text.empty()) {
        m_benchmark_manager->set_reference_text(ref_text);
        std::cout << "[Benchmark] Loaded reference text: "
                  << ref_text.length() << " characters" << std::endl;
    } else {
        std::cout << "[Benchmark] Warning: No reference text found. "
                  << "Accuracy metrics will not be available." << std::endl;
        std::cout << "[Benchmark] Expected file: " << wav_path << ".txt" << std::endl;
    }

    // Set completion callback
    benchmark_source->set_completion_callback([this]() {
        std::thread([this]() {
            std::cout << "[Benchmark] Audio processing complete, calculating final metrics..."
                      << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            stop_benchmark();
        }).detach();
    });

    // Set progress callback
    // m_benchmark_manager->set_progress_callback(
    //     [](const benchmark_manager::benchmark_results& results) {
    //     static int update_count = 0;
    //     update_count++;

    //     // Show progress every 10 updates
    //     if (update_count % 10 == 0) {
    //         std::cout << "[Progress] Processed: " << std::fixed << std::setprecision(1)
    //         << (results.total_audio_duration_ms / 1000.0) << "s, "
    //         << "WER: " << results.word_error_rate << "%, "
    //         << "RTF: " << results.real_time_factor << "x" << std::endl;
    //     }
    // });

    // Start the benchmark source
    if (!benchmark_source->start()) {
        std::cerr << "[Benchmark] Failed to start benchmark audio source" << std::endl;
        m_benchmark_manager.reset();
        return false;
    }

    // Switch audio source
    {
        std::lock_guard<std::mutex> lock(m_audio_source_mutex);
        m_audio_source = std::move(benchmark_source);
        m_audio_source_type = audio_source_type::BENCHMARK;
    }

    // Start benchmark tracking
    m_benchmark_manager->start();
    m_benchmark_mode = true;

    std::cout << "[Benchmark] Benchmark started with WAV file: " << wav_path << std::endl;
    return true;
}

void wstream_app::stop_benchmark() {
    static std::atomic<bool> stopping{false};

    if (!m_benchmark_mode || stopping) {
        return;  // Already stopped or stopping
    }

    stopping = true;

    if (m_playback_manager) {
        m_playback_manager->stop();
        m_playback_manager.reset();
    }

    // Stop benchmark manager if active and show results
    if (m_benchmark_manager) {
        auto results = m_benchmark_manager->stop();

        // Print comprehensive results
        std::cout << "\n=== BENCHMARK RESULTS ===" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        std::cout << "\nMODEL INFORMATION:" << std::endl;
        std::cout << "  Model path: " << m_model_path << std::endl;
        // Get model file size
        try {
            std::ifstream model_file(m_model_path, std::ios::binary | std::ios::ate);
            if (model_file.is_open()) {
                size_t file_size = model_file.tellg();
                std::cout << "  Size: " << std::fixed << std::setprecision(2)
                          << (file_size / (1024.0 * 1024.0)) << " MB" << std::endl;
                model_file.close();
            }
        } catch (...) {
            // Ignore errors
        }
        std::cout << std::endl;

        // Timing metrics
        std::cout << "TIMING METRICS:" << std::endl;
        std::cout << "  Total audio duration: " << std::fixed << std::setprecision(2)
                  << results.total_audio_duration_ms / 1000.0 << " seconds" << std::endl;
        std::cout << "  Total processing time: " << results.total_processing_time_ms / 1000.0
                  << " seconds" << std::endl;
        std::cout << "  Real-time factor (RTF): " << results.real_time_factor << "x"
                  << (results.real_time_factor < 1.0 ? " (faster than real-time)" :
                          " (slower than real-time)") << std::endl;
        std::cout << "  Average latency: " << results.average_latency_ms << " ms" << std::endl;
        std::cout << "  Min/Max latency: " << results.min_latency_ms << " / "
                  << results.max_latency_ms << " ms" << std::endl;
        std::cout << std::endl;

        // Accuracy metrics
        if (!results.reference_text.empty()) {
            std::cout << "ACCURACY METRICS:" << std::endl;
            std::cout << "  Word Error Rate (WER): " << std::fixed << std::setprecision(2)
                      << results.word_error_rate << "%" << std::endl;
            std::cout << "  Character Error Rate (CER): " << results.character_error_rate
                      << "%" << std::endl;
            std::cout << "  Word Accuracy: " << (100.0 - results.word_error_rate) << "%"
                      << std::endl;
            std::cout << "  Total words: " << results.total_words << std::endl;
            std::cout << "  Word errors: " << results.word_errors
                      << " (S:" << results.word_substitutions
                      << " D:" << results.word_deletions
                      << " I:" << results.word_insertions << ")" << std::endl;
            std::cout << std::endl;
        } else {
            std::cout << "ACCURACY METRICS: No reference text available" << std::endl;
            std::cout << std::endl;
        }

        // Throughput metrics
        std::cout << "THROUGHPUT METRICS:" << std::endl;
        std::cout << "  Total samples processed: " << results.total_samples_processed << std::endl;
        std::cout << "  Total segments: " << results.total_segments << std::endl;
        std::cout << "  Samples per second: " << std::fixed << std::setprecision(0)
                  << results.samples_per_second << std::endl;
        std::cout << "  Audio processing speed: " << std::fixed << std::setprecision(2)
                  << (results.total_audio_duration_ms / results.total_processing_time_ms)
                  << "x real-time" << std::endl;
        std::cout << std::endl;

        // Show transcription comparison
        if (!results.reference_text.empty()) {
            std::cout << "TRANSCRIPTION COMPARISON:" << std::endl;
            std::cout << std::string(50, '-') << std::endl;

            // Limit display length for readability
            const size_t MAX_DISPLAY_LENGTH = 200;

            std::cout << "REFERENCE (first " << MAX_DISPLAY_LENGTH << " chars):" << std::endl;
            std::string ref_display = results.reference_text.substr(0, MAX_DISPLAY_LENGTH);
            if (results.reference_text.length() > MAX_DISPLAY_LENGTH) ref_display += "...";
            std::cout << "  " << ref_display << std::endl;
            std::cout << std::endl;

            std::cout << "HYPOTHESIS (first " << MAX_DISPLAY_LENGTH << " chars):" << std::endl;
            std::string hyp_display = results.hypothesis_text.substr(0, MAX_DISPLAY_LENGTH);
            if (results.hypothesis_text.length() > MAX_DISPLAY_LENGTH) hyp_display += "...";
            std::cout << "  " << hyp_display << std::endl;
            std::cout << std::endl;
        }

        // Export detailed results
        m_benchmark_manager->export_results(results, "./benchmark_results.txt", m_model_path);
        std::cout << "Detailed results exported to: ./benchmark_results.txt" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        // Add model comparison table to results file
        std::ofstream comparison_file("benchmark_comparison.csv", std::ios::app);
        if (comparison_file.is_open()) {
            // Check if file is empty to add header
            comparison_file.seekp(0, std::ios::end);
            if (comparison_file.tellp() == 0) {
                comparison_file <<
                    "Timestamp,Model,Size_MB,WER,CER,RTF,Latency_ms,Audio_Duration_s,Processing_Time_s\n";
            }

            // Get model size
            std::ifstream model_file(m_model_path, std::ios::binary | std::ios::ate);
            size_t model_size_mb = model_file.is_open() ? model_file.tellg() / (1024 * 1024) : 0;
            model_file.close();

            // Add row
            auto now = std::chrono::system_clock::now();
            auto now_time_t = std::chrono::system_clock::to_time_t(now);
            std::string timestamp = std::ctime(&now_time_t);
            timestamp.pop_back(); // Remove trailing newline

            comparison_file << timestamp << ","
                            << fs::path(m_model_path).filename().string() << ","
                            << model_size_mb << ","
                            << results.word_error_rate << ","
                            << results.character_error_rate << ","
                            << results.real_time_factor << ","
                            << results.average_latency_ms << ","
                            << results.total_audio_duration_ms / 1000.0 << ","
                            << results.total_processing_time_ms / 1000.0 << "\n";

            comparison_file.close();
            std::cout << "Results added to benchmark_comparison.csv for easy comparison"
                      << std::endl;

            update_comparison_table(results);
        }
    }

    // Turn off benchmark mode
    m_benchmark_mode = false;

    // Switch back to default audio source if currently using benchmark
    if (m_audio_source_type == audio_source_type::BENCHMARK) {
        switch_audio_source(audio_source_type::SDL_MICROPHONE);
    }

    stopping = false;
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

void wstream_app::update_comparison_table(const benchmark_manager::benchmark_results& results) {
    // Get model name from path
    std::string model_name = fs::path(m_model_path).filename().string();

    // Get model size
    size_t model_size_mb = 0;
    try {
        std::ifstream model_file(m_model_path, std::ios::binary | std::ios::ate);
        if (model_file.is_open()) {
            model_size_mb = model_file.tellg() / (1024 * 1024);
        }
    } catch (...) {
        // Ignore errors
    }

    // Get timestamp
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::string timestamp = std::ctime(&now_time_t);
    timestamp.pop_back(); // Remove trailing newline

    // CSV format for machine-readable data
    std::ofstream csv_file("benchmark_comparison.csv", std::ios::app);
    if (csv_file.is_open()) {
        // Check if file is empty to add header
        csv_file.seekp(0, std::ios::end);
        if (csv_file.tellp() == 0) {
            csv_file <<
                "Timestamp,Model,Size_MB,WER,CER,RTF,Latency_ms,Processing_Time_s,Audio_Duration_s\n";
        }

        csv_file << timestamp << ","
                 << model_name << ","
                 << model_size_mb << ","
                 << results.word_error_rate << ","
                 << results.character_error_rate << ","
                 << results.real_time_factor << ","
                 << results.average_latency_ms << ","
                 << results.total_processing_time_ms / 1000.0 << ","
                 << results.total_audio_duration_ms / 1000.0 << "\n";

        csv_file.close();
    }

    // Create or update a Markdown table for human-readable comparison
    std::vector<std::vector<std::string>> table_data;
    std::ifstream csv_in("benchmark_comparison.csv");
    if (csv_in.is_open()) {
        std::string line;
        // Skip header
        std::getline(csv_in, line);

        while (std::getline(csv_in, line)) {
            std::stringstream ss(line);
            std::vector<std::string> row;
            std::string cell;

            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }

            if (row.size() >= 9) {
                table_data.push_back(row);
            }
        }
        csv_in.close();
    }

    // Generate markdown table
    std::ofstream md_file("benchmark_results.md");
    if (md_file.is_open()) {
        md_file << "# Whisper Benchmark Results\n\n";
        md_file << "Generated on: " << timestamp << "\n\n";

        md_file << "## Model Comparison\n\n";
        md_file << "| Model | Size (MB) | WER (%) | CER (%) | RTF | Latency (ms) | Processing Time (s) |\n";
        md_file << "|-------|-----------|---------|---------|-----|--------------|--------------------|\n";

        for (const auto& row : table_data) {
            md_file << "| " << row[1] << " | " // Model
                    << row[2] << " | "        // Size
                    << row[3] << " | "        // WER
                    << row[4] << " | "        // CER
                    << row[5] << " | "        // RTF
                    << row[6] << " | "        // Latency
                    << row[7] << " |"         // Processing Time
                    << "\n";
        }

        md_file << "\n## Ranking by Accuracy (WER)\n\n";

        // Sort by WER (ascending)
        std::vector<std::vector<std::string>> sorted_by_wer = table_data;
        std::sort(sorted_by_wer.begin(), sorted_by_wer.end(),
                  [](const std::vector<std::string>& a, const std::vector<std::string>& b) {
                      return std::stod(a[3]) < std::stod(b[3]);
                  });

        md_file << "| Rank | Model | WER (%) | CER (%) |\n";
        md_file << "|------|-------|---------|--------|\n";

        for (size_t i = 0; i < sorted_by_wer.size(); ++i) {
            md_file << "| " << (i + 1) << " | "
                    << sorted_by_wer[i][1] << " | "  // Model
                    << sorted_by_wer[i][3] << " | "  // WER
                    << sorted_by_wer[i][4] << " |"   // CER
                    << "\n";
        }

        md_file << "\n## Ranking by Speed (RTF)\n\n";

        // Sort by RTF (ascending - lower is better)
        std::vector<std::vector<std::string>> sorted_by_rtf = table_data;
        std::sort(sorted_by_rtf.begin(), sorted_by_rtf.end(),
                  [](const std::vector<std::string>& a, const std::vector<std::string>& b) {
                      return std::stod(a[5]) < std::stod(b[5]);
                  });

        md_file << "| Rank | Model | RTF | Latency (ms) |\n";
        md_file << "|------|-------|-----|-------------|\n";

        for (size_t i = 0; i < sorted_by_rtf.size(); ++i) {
            md_file << "| " << (i + 1) << " | "
                    << sorted_by_rtf[i][1] << " | "  // Model
                    << sorted_by_rtf[i][5] << " | "  // RTF
                    << sorted_by_rtf[i][6] << " |"   // Latency
                    << "\n";
        }

        md_file << "\n## Recommendations\n\n";
        md_file << "### Best Overall Model\n";

        // Calculate overall score (weighted: 60% accuracy, 40% speed)
        std::vector<std::pair<std::string, double>> model_scores;
        for (const auto& row : table_data) {
            double wer = std::stod(row[3]);
            double rtf = std::stod(row[5]);

            double accuracy_score = std::max(0.0, 100.0 - wer);
            double speed_score = std::max(0.0, 100.0 * (2.0 - rtf) / 2.0);
            double overall_score = 0.6 * accuracy_score + 0.4 * speed_score;

            model_scores.push_back({row[1], overall_score});
        }

        std::sort(model_scores.begin(), model_scores.end(),
                  [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                      return a.second > b.second;
                  });

        if (!model_scores.empty()) {
            md_file << "**" << model_scores[0].first << "** with overall score: "
                    << std::fixed << std::setprecision(1) << model_scores[0].second << "/100\n\n";
        }

        md_file << "### Best for Accuracy\n";
        md_file << "**" << sorted_by_wer[0][1] << "** with WER: " << sorted_by_wer[0][3] << "%\n\n";

        md_file << "### Best for Speed\n";
        md_file << "**" << sorted_by_rtf[0][1] << "** with RTF: " << sorted_by_rtf[0][5] << "x\n\n";

        md_file << "### Best for Low Latency\n";
        // Sort by latency
        std::vector<std::vector<std::string>> sorted_by_latency = table_data;
        std::sort(sorted_by_latency.begin(), sorted_by_latency.end(),
                  [](const std::vector<std::string>& a, const std::vector<std::string>& b) {
                      return std::stod(a[6]) < std::stod(b[6]);
                  });

        md_file << "**" << sorted_by_latency[0][1] << "** with latency: "
                << sorted_by_latency[0][6] << " ms\n\n";

        md_file << "## System Information\n\n";
        md_file << "- CPU: ";

#ifdef __linux__
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        std::string cpu_model;
        int cpu_cores = 0;

        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                cpu_model = line.substr(line.find(":") + 2);
                cpu_cores++;
            }
        }

        md_file << cpu_model << " (" << cpu_cores << " cores)\n";
#else
        md_file << "Unknown\n";
#endif

        md_file << "- OS: ";
#ifdef __linux__
        md_file << "Linux\n";
#elif defined(_WIN32)
        md_file << "Windows\n";
#elif defined(__APPLE__)
        md_file << "macOS\n";
#else
        md_file << "Unknown\n";
#endif

        md_file << "- Build: ";
#ifdef NDEBUG
        md_file << "Release\n";
#else
        md_file << "Debug\n";
#endif

        md_file << "\n## Benchmark Audio\n\n";
        md_file << "- Duration: " << results.total_audio_duration_ms / 1000.0 << " seconds\n";
        md_file << "- Sample Rate: 16000 Hz\n";
        md_file << "- Word Count: " << results.total_words << "\n";

        md_file.close();

        std::cout << "\nComprehensive benchmark results saved to benchmark_results.md\n";
    }
}
