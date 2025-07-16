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
#include "audio_source.h"
#include "audio_source_factory.h"
#include "benchmark_manager.h"
#include <hyni/hyni_websocket_server.h>
#include "transcription_marker.h"
#include "audio_playback_manager.h"
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
class audio_processor;
class benchmark_audio_source;

/**
 * @class wstream_app
 * @brief Main application class that coordinates all components for real-time audio transcription
 *
 * @details This class serves as the central coordinator for the WStream application, managing:
 * - Audio source management (SDL2 microphone, WebSocket clients, or benchmark files)
 * - Whisper-based speech recognition with configurable models
 * - Voice Activity Detection (VAD) support for all audio sources
 * - Text processing and filtering for improved accuracy
 * - WebSocket server for real-time broadcasting and client communication
 * - Comprehensive benchmarking capabilities with accuracy metrics
 * - Audio playback for quality verification
 *
 * The application uses a factory pattern to create the appropriate audio source at startup,
 * ensuring only the necessary components are instantiated and resources are used efficiently.
 *
 * @par Features:
 * - Multiple audio source support with runtime switching
 * - Real-time transcription with configurable chunk sizes
 * - VAD support for reduced processing during silence
 * - Benchmark mode with WER/CER accuracy metrics
 * - WebSocket API for remote control and monitoring
 * - Audio playback for quality verification
 * - Text marking for visual accuracy comparison
 *
 * @par Thread Safety:
 * The class uses mutex protection for audio source switching and transcription updates.
 * WebSocket server runs in its own thread. Audio processing runs in the main thread.
 *
 * @par Usage Example:
 * @code
 * wstream_app app("path/to/model.bin", audio_source_type::SDL_MICROPHONE);
 * if (app.initialize(argc, argv)) {
 *     app.run();
 * }
 * @endcode
 *
 * @see audio_source
 * @see whisper_engine
 * @see websocket_server
 * @see benchmark_manager
 */
class wstream_app {
public:
    // Constants

    /** @brief Default path to Whisper model file */
    static constexpr const char* DEFAULT_MODEL_PATH = "models/ggml-small.en-tdrz.bin";

    /** @brief Default WebSocket server port */
    static constexpr uint16_t DEFAULT_WEBSOCKET_PORT = 8080;

    /** @brief Minimum allowed chunk size in milliseconds */
    static constexpr int MIN_CHUNK_SIZE_MS = 100;

    /** @brief Maximum allowed chunk size in milliseconds */
    static constexpr int MAX_CHUNK_SIZE_MS = 10000;

    /** @brief Default benchmark WAV file path */
    static constexpr const char* DEFAULT_BENCHMARK_WAV = "./benchmark.wav";

    /** @brief Sleep duration for main loop throttling */
    static constexpr int LOOP_SLEEP_MS = 1;

    /** @brief Sleep duration during audio source switching */
    static constexpr int SWITCH_SLEEP_MS = 10;

    /** @brief Delay before benchmark completion */
    static constexpr int BENCHMARK_COMPLETION_DELAY_MS = 1000;

    /** @brief Static instance pointer for signal handler */
    static wstream_app* s_instance;

    /**
     * @brief Constructs wstream_app with specified audio source type
     * @param model_path Path to the Whisper model file
     * @param source_type Type of audio source to use
     * @param websocket_port Port for WebSocket server
     *
     * @details Initializes the application with the specified configuration.
     * The actual initialization of components happens in initialize().
     *
     * @note Model path can be relative or absolute
     */
    explicit wstream_app(const std::string& model_path = DEFAULT_MODEL_PATH,
                         audio_source_type source_type = audio_source_type::SDL_MICROPHONE,
                         uint16_t websocket_port = DEFAULT_WEBSOCKET_PORT);

    /**
     * @brief Destructor - ensures clean shutdown of all components
     *
     * @details Performs cleanup in the correct order:
     * - Stops audio processing
     * - Shuts down WebSocket server
     * - Releases all resources
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
     * @details Performs the following initialization steps:
     * 1. Parses command line arguments
     * 2. Validates model file existence
     * 3. Initializes Whisper engine
     * 4. Creates text processor
     * 5. Sets up WebSocket server
     * 6. Creates and configures audio source
     *
     * @pre Model file must exist at specified path
     * @post All components are initialized and ready to run
     *
     * @note Call this before run()
     */
    bool initialize(int argc, char* argv[]);

    /**
     * @brief Starts the main application loop
     *
     * @details This method runs the main processing loop that:
     * - Captures audio samples from the configured source
     * - Processes them through Whisper
     * - Filters and cleans the transcribed text
     * - Broadcasts results to connected WebSocket clients
     *
     * The loop continues until shutdown is requested via:
     * - SDL window close event
     * - SIGINT/SIGTERM signals
     * - WebSocket shutdown command
     *
     * @pre initialize() must have been called successfully
     * @post Application components are stopped
     *
     * @note This method blocks until shutdown is requested
     */
    void run();

    /**
     * @brief Gracefully shuts down all application components
     *
     * @details Performs cleanup in the correct order:
     * - Stops audio capture
     * - Shuts down WebSocket server
     * - Releases Whisper resources
     * - Cleans up SDL
     *
     * @note Safe to call multiple times
     * @note Automatically called by destructor
     */
    void shutdown();

    /**
     * @brief Gets the active audio source type
     * @return Current audio source type
     *
     * @note Thread-safe
     */
    audio_source_type get_audio_source_type() const { return m_audio_source_type; }

    /**
     * @brief Gets the active audio source name
     * @return Human-readable name of the current audio source
     *
     * @note Thread-safe
     */
    std::string get_audio_source_name() const;

    /**
     * @brief Sets the audio source type at runtime
     * @param source_type New audio source type to use
     * @return true if switch successful, false otherwise
     *
     * @details Creates a new audio source and replaces the current one.
     * Any ongoing audio processing will be interrupted briefly.
     *
     * @note Thread-safe - uses mutex protection
     * @warning May cause brief audio interruption
     */
    bool set_audio_source_runtime(audio_source_type source_type);

    /**
     * @brief Handles incoming audio data from WebSocket clients
     * @param samples PCM audio samples (16-bit)
     * @param session_id Client session identifier
     * @param language Language hint (optional)
     *
     * @details This method is called by the WebSocket server when audio data
     * is received from a client. Only functional when using WEBSOCKET_CLIENT
     * audio source type.
     *
     * @note Thread-safe
     */
    void handle_websocket_audio(const std::vector<int16_t>& samples,
                                const std::string& session_id = "",
                                const std::string& language = "");

    /**
     * @brief Gets the latest transcription result
     * @return Latest transcription text, empty if none available
     *
     * @details Retrieves and clears the latest transcription result.
     * Used by WebSocket API for polling transcription results.
     *
     * @note Thread-safe
     * @note Clears the transcription after reading
     */
    std::string get_latest_transcription();

    /**
     * @brief Checks if the application is currently running
     * @return true if running, false otherwise
     *
     * @note Thread-safe
     */
    bool is_running() const { return m_is_running; }

    /**
     * @brief Starts benchmark mode with specified WAV file
     * @param wav_path Path to WAV file for benchmarking
     * @return true if benchmark started successfully, false otherwise
     *
     * @details Switches to benchmark audio source and begins processing
     * the specified WAV file. Enables accuracy metrics if reference text
     * is available.
     *
     * @pre WAV file must exist and be valid
     * @post Benchmark mode is active
     */
    bool start_benchmark(const std::string& wav_path = DEFAULT_BENCHMARK_WAV);

    /**
     * @brief Stops benchmark mode and displays results
     *
     * @details Finalizes benchmark processing, calculates metrics,
     * and exports results to files. Switches back to default audio source.
     *
     * @note Safe to call even if benchmark is not running
     */
    void stop_benchmark();

    /**
     * @brief Checks if benchmark mode is active
     * @return true if in benchmark mode, false otherwise
     *
     * @note Thread-safe
     */
    bool is_benchmark_mode() const { return m_benchmark_mode; }

private:
    // Core components

    /** @brief Whisper speech recognition engine */
    std::unique_ptr<whisper_engine> m_whisper_engine;

    /** @brief Text processing and filtering component */
    std::unique_ptr<text_processor> m_text_processor;

    /** @brief WebSocket server for broadcasting and control */
    std::unique_ptr<hyni_websocket_server> m_websocket_server;

    /** @brief The single active audio source */
    std::unique_ptr<audio_source> m_audio_source;

    /** @brief WebSocket audio source reference (when applicable) */
    websocket_audio_source* m_websocket_audio_source{nullptr};

    // Benchmark components

    /** @brief Benchmark manager for accuracy metrics */
    std::unique_ptr<benchmark_manager> m_benchmark_manager;

    /** @brief Audio playback manager for benchmark verification */
    std::unique_ptr<audio_playback_manager> m_playback_manager;

    /** @brief Transcription marker for visual comparison */
    transcription_marker m_transcription_marker;

    // Configuration

    /** @brief Path to the Whisper model file */
    std::string m_model_path;

    /** @brief WebSocket server port */
    uint16_t m_websocket_port;

    /** @brief Current audio source type */
    audio_source_type m_audio_source_type;

    /** @brief Custom chunk size in milliseconds (0 = use default) */
    int m_chunk_size_ms{0};

    // State flags

    /** @brief Main application loop control */
    std::atomic<bool> m_is_running{true};

    /** @brief Audio source switching in progress */
    std::atomic<bool> m_switching_source{false};

    /** @brief Benchmark mode active */
    std::atomic<bool> m_benchmark_mode{false};

    /** @brief VAD enabled flag */
    bool m_vad_enabled{false};

    /** @brief Text marking enabled for benchmark */
    bool m_benchmark_enable_marker{false};

    /** @brief Audio playback enabled for benchmark */
    bool m_benchmark_enable_playback{false};

    // Thread safety

    /** @brief Mutex for audio source access */
    mutable std::mutex m_audio_source_mutex;

    /** @brief Mutex for transcription updates */
    mutable std::mutex m_transcription_mutex;

    /** @brief Latest transcription result */
    std::string m_latest_transcription;

    // Private methods

    /**
     * @brief Validates that a model file exists and is accessible
     * @param path Path to validate
     * @return true if path exists and is readable, false otherwise
     */
    bool validate_model_path(const std::string& path) const;

    /**
     * @brief Main audio processing loop
     *
     * @details Continuously:
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
    void show_help(const std::string& program_name) const;

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

    /**
     * @brief Sets up WebSocket audio callback
     *
     * @details Configures the WebSocket server to route incoming audio
     * to the appropriate audio source handler.
     */
    void setup_websocket_audio_callback();

    /**
     * @brief Updates benchmark comparison table with results
     * @param results Benchmark results to add to comparison
     */
    void update_comparison_table(const benchmark_manager::benchmark_results& results);

    /**
     * @brief Configures audio processor with VAD settings
     * @param processor Audio processor to configure
     * @return true if configuration successful, false otherwise
     */
    bool configure_audio_processor_vad(audio_processor* processor);

    /**
     * @brief Initializes benchmark components
     * @param wav_path Path to benchmark WAV file
     * @return Configured benchmark audio source, nullptr on failure
     */
    std::unique_ptr<benchmark_audio_source> initialize_benchmark_source(const std::string& wav_path);

    /**
     * @brief Processes a single transcription result
     * @param transcription Raw transcription from Whisper
     * @param session_id Optional session identifier
     * @param processing_latency_ms Processing time in milliseconds
     */
    void process_transcription_result(const std::string& transcription,
                                      const std::string& session_id,
                                      double processing_latency_ms);
};
