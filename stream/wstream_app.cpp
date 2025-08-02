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
#include "whisper_engine.h"
#include "text_processor.h"
#include "websocket_audio_source.h"
#include "benchmark_audio_source.h"
#include "sdl_audio_source.h"
#include <iostream>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <fstream>

// Global shutdown flag
std::atomic<bool> g_shutdown_requested{false};

// Static flag for signal handler
static std::atomic<bool> s_sigint_received{false};

// Static instance pointer for signal handler
wstream_app* wstream_app::s_instance = nullptr;

/**
 * @brief Signal handler for graceful shutdown
 * @param signal Signal number received
 */
static void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        s_sigint_received = true;
        g_shutdown_requested = true;

        if (wstream_app::s_instance) {
            std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
        }
    }
}

//
// Constructor and Destructor
//

wstream_app::wstream_app(const std::string& model_path,
                         audio_source_type source_type,
                         uint16_t websocket_port)
    : m_model_path(model_path)
    , m_websocket_port(websocket_port)
    , m_audio_source_type(source_type) {
    s_instance = this;
}

wstream_app::~wstream_app() {
    shutdown();
    s_instance = nullptr;
}

//
// Public Methods
//

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

    // Handle benchmark mode
    if (m_benchmark_mode) {
        std::string wav_path = DEFAULT_BENCHMARK_WAV;
        if (const char* env_wav = std::getenv("WSTREAM_BENCHMARK_WAV")) {
            wav_path = env_wav;
        }

        if (!start_benchmark(wav_path)) {
            std::cerr << "Failed to start benchmark mode." << std::endl;
            return false;
        }
    } else {
        // Normal mode - create audio source
        m_audio_source = create_audio_source(m_audio_source_type);
        if (!m_audio_source) {
            std::cerr << "Failed to create audio source: " << get_audio_source_name() << std::endl;
            return false;
        }

        // Store typed reference for WebSocket audio source
        if (m_audio_source_type == audio_source_type::WEBSOCKET_CLIENT) {
            m_websocket_audio_source = dynamic_cast<websocket_audio_source*>(m_audio_source.get());
        }

        // Initialize the audio source
        if (!m_audio_source->initialize()) {
            std::cerr << "Failed to initialize audio source: " << get_audio_source_name() << std::endl;
            return false;
        }
    }

    std::cout << "<--- wstream initialized successfully." << std::endl;
    return true;
}

void wstream_app::run() {
    // Start the audio source
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

void wstream_app::shutdown() {
    std::cout << "Shutting down WStream..." << std::endl;

    m_is_running = false;
    g_shutdown_requested = true;

    // Stop audio source
    if (m_audio_source) {
        m_audio_source->stop();
    }

    // Stop WebSocket server
    if (m_websocket_server) {
        m_websocket_server->stop();
    }

    SDL_Quit();
    std::cout << "Shutdown complete." << std::endl;

    // Force exit after a short delay if we're still here
    std::thread([]{
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::exit(0);
    }).detach();
}

std::string wstream_app::get_audio_source_name() const {
    return audio_source_factory::get_type_name(m_audio_source_type);
}

bool wstream_app::set_audio_source_runtime(audio_source_type source_type) {
    if (source_type == m_audio_source_type) {
        return true; // Already using this source
    }

    std::cout << "Switching audio source from " << get_audio_source_name()
              << " to " << audio_source_factory::get_type_name(source_type) << std::endl;

    return switch_audio_source(source_type);
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
    std::string result = std::move(m_latest_transcription);
    m_latest_transcription.clear();
    return result;
}

bool wstream_app::start_benchmark(const std::string& wav_path) {
    std::cout << "[Benchmark] Starting benchmark with WAV file: " << wav_path << std::endl;

    // Display enabled features
    std::cout << "[Benchmark] Features: ";
    std::vector<std::string> features;
    if (m_benchmark_enable_marker) features.push_back("text marking");
    if (m_benchmark_enable_playback) features.push_back("audio playback");

    if (features.empty()) {
        std::cout << "continuous mode";
    } else {
        for (size_t i = 0; i < features.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << features[i];
        }
    }
    std::cout << std::endl;

    // Create benchmark manager
    m_benchmark_manager = std::make_unique<benchmark_manager>();

    // Initialize benchmark source
    auto benchmark_source = initialize_benchmark_source(wav_path);
    if (!benchmark_source) {
        m_benchmark_manager.reset();
        return false;
    }

    // Initialize audio playback if enabled
    if (m_benchmark_enable_playback) {
        m_playback_manager = std::make_unique<audio_playback_manager>();
        if (!m_playback_manager->initialize()) {
            std::cerr << "[Benchmark] Warning: Failed to initialize audio playback" << std::endl;
            m_playback_manager.reset();
            m_benchmark_enable_playback = false;
        } else {
            std::cout << "[Benchmark] Audio playback initialized successfully" << std::endl;
        }
    }

    // Load reference text
    std::string ref_text = benchmark_source->get_reference_text();
    if (!ref_text.empty()) {
        m_benchmark_manager->set_reference_text(ref_text);
        std::cout << "[Benchmark] Loaded reference text: "
                  << ref_text.length() << " characters" << std::endl;

        if (m_benchmark_enable_marker) {
            m_transcription_marker.load_reference(ref_text);
            m_transcription_marker.set_fuzzy_matching(true);
            m_transcription_marker.set_fuzzy_threshold(m_marker_config.fuzzy_threshold);
            m_transcription_marker.set_search_distance(m_marker_config.search_distance);
            m_transcription_marker.set_show_confidence(m_marker_config.show_confidence);
            m_transcription_marker.set_logging(false);
            m_transcription_marker.set_streaming_mode(true);
        }
    } else {
        std::cout << "[Benchmark] Warning: No reference text found." << std::endl;
        std::cout << "[Benchmark] Expected file: " << wav_path << ".txt" << std::endl;
    }

    // Set completion callback
    benchmark_source->set_completion_callback([this]() {
        std::thread([this]() {
            std::cout << "[Benchmark] Audio processing complete, calculating final metrics..."
                      << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(BENCHMARK_COMPLETION_DELAY_MS));
            stop_benchmark();
        }).detach();
    });

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

    std::cout << "[Benchmark] Benchmark started successfully" << std::endl;
    return true;
}

void wstream_app::stop_benchmark() {
    static std::atomic<bool> stopping{false};

    if (!m_benchmark_mode || stopping) {
        return;
    }

    stopping = true;

    // Stop audio playback
    if (m_playback_manager) {
        m_playback_manager->stop();
        m_playback_manager.reset();
    }

    // Stop benchmark and get results
    if (m_benchmark_manager) {
        auto results = m_benchmark_manager->stop();

        // Print results
        std::cout << "\n=== BENCHMARK RESULTS ===" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        // Model information
        std::cout << "\nMODEL INFORMATION:" << std::endl;
        std::cout << "  Model path: " << m_model_path << std::endl;

        try {
            std::ifstream model_file(m_model_path, std::ios::binary | std::ios::ate);
            if (model_file.is_open()) {
                size_t file_size = model_file.tellg();
                std::cout << "  Size: " << std::fixed << std::setprecision(2)
                          << (file_size / (1024.0 * 1024.0)) << " MB" << std::endl;
            }
        } catch (...) {
            // Ignore errors
        }

        // Timing metrics
        std::cout << "\nTIMING METRICS:" << std::endl;
        std::cout << "  Total audio duration: " << std::fixed << std::setprecision(2)
                  << results.total_audio_duration_ms / 1000.0 << " seconds" << std::endl;
        std::cout << "  Total processing time: "
                  << results.total_processing_time_ms / 1000.0 << " seconds" << std::endl;
        std::cout << "  Real-time factor (RTF): " << results.real_time_factor << "x"
                  << (results.real_time_factor < 1.0 ? " (faster than real-time)" :
                          " (slower than real-time)") << std::endl;
        std::cout << "  Average latency: " << results.average_latency_ms << " ms" << std::endl;

        // Accuracy metrics
        if (!results.reference_text.empty()) {
            std::cout << "\nACCURACY METRICS:" << std::endl;
            std::cout << "  Word Error Rate (WER): " << std::fixed << std::setprecision(2)
                      << results.word_error_rate << "%" << std::endl;
            std::cout << "  Character Error Rate (CER): "
                      << results.character_error_rate << "%" << std::endl;
            std::cout << "  Word Accuracy: "
                      << (100.0 - results.word_error_rate) << "%" << std::endl;
            std::cout << "  Total words: " << results.total_words << std::endl;
        }

        // Export results
        m_benchmark_manager->export_results(results, "./benchmark_results.txt", m_model_path);
        std::cout << "\nDetailed results exported to: ./benchmark_results.txt" << std::endl;

        update_comparison_table(results);

        std::cout << std::string(50, '=') << std::endl;
    }

    // Reset benchmark state
    m_benchmark_mode = false;
    //m_transcription_marker.reset();

    // Switch back to default audio source
    if (m_audio_source_type == audio_source_type::BENCHMARK) {
        switch_audio_source(audio_source_type::SDL_MICROPHONE);
    }

    stopping = false;
}

//
// Private Methods
//

bool wstream_app::validate_model_path(const std::string& path) const {
    return fs::exists(path) && fs::is_regular_file(path);
}

void wstream_app::process_audio_loop() {
    std::vector<float> audio_samples;

    while (m_is_running) {
        // Check SDL events
        m_is_running = sdl_poll_events();

        // Check for shutdown signals
        if (s_sigint_received || g_shutdown_requested) {
            std::cout << "\nShutdown requested..." << std::endl;
            m_is_running = false;
            break;
        }

        // Wait if switching audio source
        if (m_switching_source) {
            std::this_thread::sleep_for(std::chrono::milliseconds(SWITCH_SLEEP_MS));
            continue;
        }

        // Get audio samples
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
            std::this_thread::sleep_for(std::chrono::milliseconds(LOOP_SLEEP_MS));
            continue;
        }

        // Queue audio for playback in benchmark mode
        if (m_benchmark_mode && m_benchmark_enable_playback && m_playback_manager) {
            m_playback_manager->queue_audio(audio_samples);
        }

        // Process with Whisper
        auto start_time = std::chrono::steady_clock::now();
        std::string transcription = m_whisper_engine->transcribe(audio_samples);
        auto end_time = std::chrono::steady_clock::now();

        double processing_latency_ms = std::chrono::duration<double, std::milli>(
                                           end_time - start_time).count();

        // Process transcription result
        if (!transcription.empty()) {
            process_transcription_result(transcription, session_id, processing_latency_ms);
        }

        // Check if benchmark is complete
        if (m_benchmark_mode && m_audio_source_type == audio_source_type::BENCHMARK) {
            auto benchmark_source = dynamic_cast<benchmark_audio_source*>(m_audio_source.get());
            if (benchmark_source && benchmark_source->is_end_of_file()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(BENCHMARK_COMPLETION_DELAY_MS));
                stop_benchmark();
                break;
            }
        }

        audio_samples.clear();
    }

    // Cleanup
    if (m_benchmark_mode) {
        stop_benchmark();
    }
}

bool wstream_app::parse_command_line(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            show_help(argv[0]);
            return false;
        }
        else if (arg == "--audio-source" && i + 1 < argc) {
            m_audio_source_type = audio_source_factory::parse_type(argv[++i]);
        }
        else if (arg == "--port" && i + 1 < argc) {
            try {
                m_websocket_port = static_cast<uint16_t>(std::stoi(argv[++i]));
            } catch (const std::exception& e) {
                std::cerr << "Invalid port number: " << argv[i] << std::endl;
                return false;
            }
        }
        else if (arg == "--benchmark") {
            m_benchmark_mode = true;
            m_audio_source_type = audio_source_type::BENCHMARK;
            std::cout << "Benchmark mode enabled" << std::endl;
        }
        else if (arg == "--enable-marker") {
            m_benchmark_enable_marker = true;
            std::cout << "Text marking enabled for benchmark mode" << std::endl;
        }
        else if (arg == "--marker-threshold" && i + 1 < argc) {
            m_marker_config.fuzzy_threshold = std::stod(argv[++i]);
            std::cout << "Marker fuzzy threshold set to: " << m_marker_config.fuzzy_threshold << std::endl;
        }
        else if (arg == "--marker-simple") {
            m_marker_config.use_simple_mode = true;
            std::cout << "Using simple marker mode (less verbose)" << std::endl;
        }
        else if (arg == "--no-marker-confidence") {
            m_marker_config.show_confidence = false;
        }
        else if (arg == "--play-audio") {
            m_benchmark_enable_playback = true;
            std::cout << "Audio playback enabled" << std::endl;
        }
        else if (arg == "--chunk-size" && i + 1 < argc) {
            try {
                int chunk_ms = std::stoi(argv[++i]);
                if (chunk_ms >= MIN_CHUNK_SIZE_MS && chunk_ms <= MAX_CHUNK_SIZE_MS) {
                    m_chunk_size_ms = chunk_ms;
                    std::cout << "Chunk size set to: " << chunk_ms << "ms" << std::endl;
                } else {
                    std::cerr << "Invalid chunk size. Must be between "
                              << MIN_CHUNK_SIZE_MS << "-" << MAX_CHUNK_SIZE_MS << "ms" << std::endl;
                    return false;
                }
            } catch (const std::exception& e) {
                std::cerr << "Invalid chunk size value" << std::endl;
                return false;
            }
        }
        else if (arg.find("--") == 0) {
            std::cerr << "Unknown option: " << arg << std::endl;
            show_help(argv[0]);
            return false;
        }
        else {
            // Assume it's a model path
            m_model_path = arg;
        }
    }

    return true;
}

void wstream_app::show_help(const std::string& program_name) const {
    std::cout << "Usage: " << program_name << " [options] [model_path]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --audio-source <type>  Audio source type (microphone, websocket) [default: microphone]\n";
    std::cout << "  --port <port>          WebSocket server port [default: " << DEFAULT_WEBSOCKET_PORT << "]\n";
    std::cout << "  --chunk-size <ms>      Set audio chunk size in milliseconds ("
              << MIN_CHUNK_SIZE_MS << "-" << MAX_CHUNK_SIZE_MS << ")\n";
    std::cout << "  --benchmark            Run in benchmark mode with default WAV file\n";
    std::cout << "  --enable-marker        Enable text marking in benchmark mode\n";
    std::cout << "  --play-audio           Enable audio playback\n";
    std::cout << "  --help, -h             Show this help message\n\n";

    std::cout << "Arguments:\n";
    std::cout << "  model_path             Path to Whisper model file [default: " << DEFAULT_MODEL_PATH << "]\n\n";

    std::cout << "Audio source types:\n";
    std::cout << "  microphone, sdl, mic   Use local microphone via SDL2\n";
    std::cout << "  websocket, ws, client  Receive audio from WebSocket clients\n\n";

    std::cout << "Examples:\n";
    std::cout << "  " << program_name << "                                    # Use default settings\n";
    std::cout << "  " << program_name << " --benchmark                         # Run benchmark\n";
    std::cout << "  " << program_name << " --benchmark --enable-marker        # Benchmark with text marking\n";
    std::cout << "  " << program_name << " --chunk-size 2000                  # Microphone with custom chunk size\n";
    std::cout << "  " << program_name << " --audio-source websocket           # WebSocket audio source\n";

    std::cout << "\nNote: VAD support is prepared but not yet implemented. The --vad flag is accepted\n";
    std::cout << "      but will not affect audio processing until VAD wrapper is integrated.\n";
}

bool wstream_app::setup_websocket_server() {
    m_websocket_server = std::make_unique<hyni_websocket_server>(m_websocket_port);

    // Set up audio callback for WebSocket audio source
    if (m_audio_source_type == audio_source_type::WEBSOCKET_CLIENT) {
        setup_websocket_audio_callback();
    }

    // Set up command handler
    m_websocket_server->set_command_handler(
        [this](const std::string& command, const nlohmann::json& params,
               websocket::stream<tcp::socket>* client_ws) -> nlohmann::json {

            nlohmann::json response;
            response["type"] = "response";
            response["action"] = command;

            if (command == "set_audio_source") {
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
                        response["message"] = success ?
                                                  "Audio source switched to " + source_str :
                                                  "Failed to switch audio source";
                    }
                }
            }
            else if (command == "get_status") {
                response["status"] = "success";
                response["audio_source"] = audio_source_factory::type_to_string(m_audio_source_type);
                response["audio_source_name"] = get_audio_source_name();
                response["is_running"] = is_running();
                response["benchmark_mode"] = m_benchmark_mode.load();
                response["client_count"] = m_websocket_server->get_client_count();
            }
            else if (command == "get_transcription") {
                response["status"] = "success";
                response["transcription"] = get_latest_transcription();
            }
            else {
                response["status"] = "error";
                response["message"] = "Unknown command: " + command;
            }

            return response;
        }
        );

    m_websocket_server->start();
    return true;
}

void wstream_app::update_latest_transcription(const std::string& transcription) {
    std::lock_guard<std::mutex> lock(m_transcription_mutex);
    m_latest_transcription = transcription;
}

bool wstream_app::switch_audio_source(audio_source_type new_source_type) {
    // Set switching flag
    m_switching_source = true;

    // Wait for main loop to stop accessing audio source
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

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

        // Try to restart old source
        if (m_audio_source) {
            m_audio_source->start();
        }
        m_switching_source = false;
        return false;
    }

    // Initialize new source
    if (!new_audio_source->initialize()) {
        std::cerr << "Failed to initialize new audio source" << std::endl;
        if (m_audio_source) {
            m_audio_source->start();
        }
        m_switching_source = false;
        return false;
    }

    // Start new source
    if (!new_audio_source->start()) {
        std::cerr << "Failed to start new audio source" << std::endl;
        if (m_audio_source) {
            m_audio_source->start();
        }
        m_switching_source = false;
        return false;
    }

    // Replace audio source
    m_audio_source = std::move(new_audio_source);
    m_audio_source_type = new_source_type;

    // Update typed references
    m_websocket_audio_source = nullptr;
    if (m_audio_source_type == audio_source_type::WEBSOCKET_CLIENT) {
        m_websocket_audio_source = dynamic_cast<websocket_audio_source*>(m_audio_source.get());
        setup_websocket_audio_callback();
    }

    m_switching_source = false;

    std::cout << "Successfully switched to audio source: " << get_audio_source_name() << std::endl;
    return true;
}

std::unique_ptr<audio_source> wstream_app::create_audio_source(audio_source_type source_type) {
    // Create configuration for SDL audio source
    if (source_type == audio_source_type::SDL_MICROPHONE) {
        sdl_audio_source::config cfg;

        if (m_chunk_size_ms > 0) {
            cfg.step_ms = m_chunk_size_ms;
        }

        auto source = std::make_unique<sdl_audio_source>(cfg);

        if (m_chunk_size_ms > 0) {
            std::cout << "[Audio] Using chunk size: " << m_chunk_size_ms << "ms" << std::endl;
        }

        return source;
    }

    // Use factory for other types
    return audio_source_factory::create(source_type);
}

void wstream_app::setup_websocket_audio_callback() {
    if (m_websocket_server) {
        m_websocket_server->set_audio_callback(
            [this](const hyni_audio_data& audio, websocket::stream<tcp::socket>* client_ws) {
                handle_websocket_audio(audio.samples, audio.session_id, audio.language);

                // Send acknowledgment
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

void wstream_app::update_comparison_table(const benchmark_manager::benchmark_results& results) {
    // Update CSV file
    std::ofstream csv_file("benchmark_comparison.csv", std::ios::app);
    if (csv_file.is_open()) {
        // Check if file is empty to add header
        csv_file.seekp(0, std::ios::end);
        if (csv_file.tellp() == 0) {
            csv_file << "Timestamp,Model,Size_MB,WER,CER,RTF,Latency_ms,Processing_Time_s,Audio_Duration_s\n";
        }

        // Get timestamp
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::string timestamp = std::ctime(&now_time_t);
        timestamp.pop_back(); // Remove newline

        // Get model info
        std::string model_name = fs::path(m_model_path).filename().string();
        size_t model_size_mb = 0;
        try {
            std::ifstream model_file(m_model_path, std::ios::binary | std::ios::ate);
            if (model_file.is_open()) {
                model_size_mb = model_file.tellg() / (1024 * 1024);
            }
        } catch (...) {}

        // Write data
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
        std::cout << "Results added to benchmark_comparison.csv" << std::endl;
    }
}

std::unique_ptr<benchmark_audio_source> wstream_app::initialize_benchmark_source(
    const std::string& wav_path) {

    benchmark_audio_source::config config;
    config.wav_file_path = wav_path;
    config.reference_text_path = wav_path + ".txt";
    config.real_time_simulation = true;

    if (m_chunk_size_ms > 0) {
        config.step_ms = m_chunk_size_ms;
    }

    auto benchmark_source = std::make_unique<benchmark_audio_source>(config);

    if (!benchmark_source->initialize()) {
        std::cerr << "[Benchmark] Failed to initialize benchmark audio source" << std::endl;
        return nullptr;
    }

    return benchmark_source;
}

void wstream_app::process_transcription_result(const std::string& transcription,
                                               const std::string& session_id,
                                               double processing_latency_ms) {
    std::string processed_text = m_text_processor->process(transcription);

    if (processed_text.empty()) {
        return;
    }

    // Display output
    if (m_benchmark_mode && m_benchmark_manager) {
        if (m_benchmark_enable_marker && !m_benchmark_manager->get_reference_text().empty()) {
            std::string marked_text;
            if (m_marker_config.use_simple_mode) {
                marked_text = m_transcription_marker.mark_differences_simple(processed_text);
            } else {
                marked_text = m_transcription_marker.mark_streaming_chunk(processed_text);
            }

            std::cout << "[" << get_audio_source_name() << "] " << marked_text << std::endl;
        } else {
            // Normal output
            std::cout << "[" << get_audio_source_name() << "] " << processed_text << std::endl;
        }

        // Add to benchmark manager
        m_benchmark_manager->add_transcription(
            processed_text,
            0.95,  // Default confidence
            0,     // Sample count handled by benchmark manager
            processing_latency_ms
            );
    } else {
        // Normal output
        std::cout << "[" << get_audio_source_name() << "] " << processed_text << std::endl;
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
