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

#include "concurrentqueue.h"
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <nlohmann/json.hpp>
#include <memory>
#include <thread>
#include <atomic>
#include <set>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = net::ip::tcp;

/**
 * @file websocket_server.h
 * @brief WebSocket server for real-time transcription broadcasting and audio processing
 * @author WStream Development Team
 * @version 2.0
 * @date 2024
 */

/**
 * @struct audio_data
 * @brief Structure representing audio data received from a client
 *
 * Contains PCM audio samples along with metadata for processing.
 */
struct audio_data {
    /// 16-bit PCM audio samples
    std::vector<int16_t> samples;

    /// Sample rate (typically 16000 Hz)
    uint32_t sample_rate = 16000;

    /// Number of channels (typically 1 for mono)
    uint16_t channels = 1;

    /// Session identifier for the client
    std::string session_id;

    /// Language hint for transcription (optional)
    std::string language;

    /// Timestamp when audio was captured (optional)
    uint64_t timestamp = 0;

    /**
     * @brief Default constructor
     */
    audio_data() = default;

    /**
     * @brief Constructs audio data with basic parameters
     * @param pcm_samples Vector of 16-bit PCM samples
     * @param rate Sample rate in Hz
     * @param ch Number of audio channels
     */
    audio_data(std::vector<int16_t> pcm_samples, uint32_t rate = 16000, uint16_t ch = 1)
        : samples(std::move(pcm_samples)), sample_rate(rate), channels(ch) {}

    /**
     * @brief Checks if audio data is valid
     * @return true if audio data contains samples and has valid parameters
     */
    bool is_valid() const {
        return !samples.empty() && sample_rate > 0 && channels > 0;
    }

    /**
     * @brief Gets duration of audio in milliseconds
     * @return Audio duration in milliseconds
     */
    double duration_ms() const {
        if (sample_rate == 0 || channels == 0) return 0.0;
        return (static_cast<double>(samples.size()) / (sample_rate * channels)) * 1000.0;
    }
};

/**
 * @class websocket_server
 * @brief High-performance bidirectional WebSocket server for transcription
 *
 * This class implements a multi-threaded WebSocket server optimized for
 * real-time broadcasting of speech transcription results and receiving
 * audio data from clients. Key features:
 *
 * - **Bidirectional Communication**: Receives audio data and broadcasts transcriptions
 * - **Multi-client Support**: Handles multiple concurrent WebSocket connections
 * - **Lock-free Queuing**: Uses concurrent queue for high-performance message passing
 * - **Audio Processing**: Handles 16k PCM audio data wrapped in JSON
 * - **Flexible Callbacks**: Customizable audio processing callbacks
 * - **Connection Management**: Handles client connections, disconnections, and errors
 * - **JSON Protocol**: Structured message format for client communication
 * - **Thread Safety**: Fully thread-safe for concurrent access
 *
 * @par Message Protocol:
 *
 * **Outgoing (Server to Client) - Transcription Results:**
 * ```json
 * {
 *   "type": "transcribe",
 *   "content": "transcribed text here",
 *   "session_id": "optional_session_id",
 *   "confidence": 0.95
 * }
 * ```
 *
 * **Incoming (Client to Server) - Audio Data:**
 * ```json
 * {
 *   "type": "audio",
 *   "audio": [1234, -567, 890, ...],
 *   "sample_rate": 16000,
 *   "channels": 1,
 *   "session_id": "client_session_123",
 *   "language": "en",
 *   "timestamp": 1640995200000
 * }
 * ```
 *
 * **Incoming (Client to Server) - Control Messages:**
 * ```json
 * {
 *   "type": "reset",
 *   "content": "reason_for_reset"
 * }
 * ```
 *
 * @par Performance Characteristics:
 * - Lock-free message queuing for minimal latency
 * - Efficient broadcast to multiple clients
 * - Pre-allocated JSON templates to reduce overhead
 * - Configurable connection limits and timeouts
 * - Optimized for real-time streaming applications
 * - Concurrent audio processing with callback system
 *
 * @par Thread Safety:
 * Fully thread-safe. Multiple threads can safely:
 * - Queue transcriptions for broadcast
 * - Process incoming audio data
 * - Check client connection status
 * - Start/stop the server
 */
class websocket_server {
public:
    /// Default WebSocket server port
    static constexpr uint16_t DEFAULT_PORT = 8080;

    /// Maximum WebSocket message size (1MB for audio data)
    static constexpr size_t MAX_MESSAGE_SIZE = 1024 * 1024;

    /// Queue timeout for waiting for new transcriptions (milliseconds)
    static constexpr int QUEUE_TIMEOUT_MS = 100;

    /// TCP no-delay option for lower latency
    static constexpr bool TCP_NO_DELAY = true;

    /// WebSocket auto-fragment setting
    static constexpr bool AUTO_FRAGMENT = false;

    /// Maximum audio samples per message (prevents excessive memory usage)
    static constexpr size_t MAX_AUDIO_SAMPLES = 160000; // 10 seconds at 16kHz

    /**
     * @typedef audio_callback_t
     * @brief Callback function type for processing received audio data
     * @param audio_data The audio data received from client
     * @param client_ws Pointer to the client WebSocket stream for sending responses
     */
    using audio_callback_t = std::function<void(const audio_data&,
                                                websocket::stream<tcp::socket>*)>;

    /**
     * @typedef error_callback_t
     * @brief Callback function type for handling errors
     * @param error_message Description of the error
     * @param client_ws Pointer to the client WebSocket stream (may be nullptr)
     */
    using error_callback_t = std::function<void(const std::string&,
                                                websocket::stream<tcp::socket>*)>;

    /**
     * @class transcription_queue
     * @brief Lock-free concurrent queue for transcription messages
     *
     * High-performance queue implementation using Moodycamel's concurrent queue
     * for efficient producer-consumer communication between transcription and
     * broadcasting threads.
     *
     * @par Features:
     * - Lock-free enqueue/dequeue operations
     * - Blocking wait with timeout support
     * - Signal-based notification for efficiency
     * - Thread-safe for multiple producers and consumers
     */
    class transcription_queue {
    private:
        /// Lock-free concurrent queue for messages
        moodycamel::ConcurrentQueue<std::string> m_queue;

        /// Atomic flag indicating new data availability
        std::atomic<bool> m_new_data{false};

        /// Condition variable for blocking waits
        std::condition_variable m_cond;

        /// Mutex for condition variable synchronization
        std::mutex m_mutex;

    public:
        /**
         * @brief Adds a transcription to the queue
         * @param transcription Text to queue for broadcasting
         *
         * Thread-safe enqueue operation that notifies waiting consumers.
         * Uses lock-free queue for high performance.
         */
        void push(const std::string& transcription);

        /**
         * @brief Attempts to retrieve a transcription (non-blocking)
         * @param transcription Output parameter to receive the message
         * @return true if message retrieved, false if queue empty
         *
         * Non-blocking dequeue operation for polling-based consumption.
         */
        bool pop(std::string& transcription);

        /**
         * @brief Waits for and retrieves a transcription with timeout
         * @param transcription Output parameter to receive the message
         * @param timeout_ms Maximum time to wait in milliseconds
         * @param is_running Reference to running flag for early termination
         * @return true if message retrieved, false if timeout or shutdown
         *
         * Blocking wait operation with timeout support. Efficiently waits
         * for new messages using condition variables, avoiding busy polling.
         */
        bool wait_and_pop(std::string& transcription, int timeout_ms,
                          const std::atomic<bool>& is_running);
    };

    /**
     * @brief Constructs WebSocket server on specified port
     * @param port TCP port number for the server (default: DEFAULT_PORT)
     *
     * Initializes the server infrastructure including:
     * - Shared state for connection management
     * - Transcription queue for message passing
     * - Thread synchronization primitives
     * - Default callback handlers
     */
    explicit websocket_server(uint16_t port = DEFAULT_PORT);

    /**
     * @brief Destructor - ensures clean server shutdown
     *
     * Automatically stops the server and waits for all threads to complete
     * before destroying resources.
     */
    ~websocket_server();

    /**
     * @brief Starts the WebSocket server
     *
     * Launches two background threads:
     * - **Server Thread**: Accepts new WebSocket connections
     * - **Broadcast Thread**: Processes transcription queue and broadcasts
     *
     * The server begins accepting connections immediately and continues
     * until stop() is called.
     */
    void start();

    /**
     * @brief Stops the WebSocket server
     *
     * Gracefully shuts down the server:
     * - Sets shutdown flag to stop accepting new connections
     * - Waits for existing connections to close
     * - Joins background threads
     * - Cleans up resources
     */
    void stop();

    /**
     * @brief Immediately broadcasts a message to all connected clients
     * @param message Text message to broadcast
     *
     * Synchronously sends the message to all currently connected WebSocket
     * clients. This bypasses the queue system for immediate delivery.
     *
     * @note For high-frequency messages, prefer queue_transcription()
     */
    void broadcast(const std::string& message);

    /**
     * @brief Queues a transcription for asynchronous broadcasting
     * @param transcription Transcribed text to broadcast
     *
     * Adds the transcription to the internal queue for broadcasting by
     * the background thread. This is the preferred method for regular
     * transcription results as it provides better performance and
     * doesn't block the calling thread.
     */
    void queue_transcription(const std::string& transcription);

    /**
     * @brief Queues a transcription with metadata for asynchronous broadcasting
     * @param transcription Transcribed text to broadcast
     * @param session_id Session identifier for the transcription
     * @param confidence Confidence score (0.0 to 1.0)
     *
     * Enhanced version of queue_transcription that includes metadata
     * in the broadcasted message.
     */
    void queue_transcription(const std::string& transcription,
                             const std::string& session_id,
                             float confidence = 0.0f);

    /**
     * @brief Sets callback for processing received audio data
     * @param callback Function to call when audio data is received
     *
     * The callback will be invoked on a background thread whenever
     * a client sends audio data. The callback should process the
     * audio quickly to avoid blocking other clients.
     */
    void set_audio_callback(audio_callback_t callback);

    /**
     * @brief Sets callback for handling errors
     * @param callback Function to call when errors occur
     *
     * The callback will be invoked when various errors occur,
     * such as malformed messages, connection issues, or processing errors.
     */
    void set_error_callback(error_callback_t callback);

    /**
     * @brief Checks if any clients are currently connected
     * @return true if at least one client is connected, false otherwise
     *
     * Thread-safe check for client connectivity. Useful for optimizing
     * processing when no clients are listening.
     */
    bool has_clients() const;

    /**
     * @brief Gets the current number of connected clients
     * @return Number of active WebSocket connections
     */
    size_t get_client_count() const;

    /**
     * @brief Sends a message to a specific client
     * @param client_ws Pointer to the client WebSocket stream
     * @param message JSON message to send
     * @return true if message sent successfully, false otherwise
     */
    bool send_to_client(websocket::stream<tcp::socket>* client_ws, const nlohmann::json& message);

    /**
     * @typedef command_handler_t
     * @brief Callback type for handling command messages
     * @param command The command string (e.g., "set_audio_source")
     * @param params JSON object containing command parameters
     * @param client_ws Pointer to the client WebSocket stream
     * @return JSON response object
     */
    using command_handler_t = std::function<nlohmann::json(const std::string&,
                                                           const nlohmann::json&,
                                                           websocket::stream<tcp::socket>*)>;

    /**
     * @brief Sets the command handler for processing client commands
     * @param handler Function to handle command messages
     */
    void set_command_handler(command_handler_t handler);

private:
    class shared_state; ///< Forward declaration for shared connection state

    /// Server port number
    uint16_t m_port;

    /// Atomic flag controlling server operation
    std::atomic<bool> m_is_running{true};

    /// Shared state for managing client connections
    std::unique_ptr<shared_state> m_state;

    /// Queue for transcription messages
    std::unique_ptr<transcription_queue> m_queue;

    /// Thread handling incoming connections
    std::thread m_server_thread;

    /// Thread handling message broadcasting
    std::thread m_broadcast_thread;

    /// Callback for processing audio data
    audio_callback_t m_audio_callback;

    /// Callback for handling errors
    error_callback_t m_error_callback;

    /// Mutex for callback access
    std::mutex m_callback_mutex;

    /// Command handler callback
    command_handler_t m_command_handler;

    /// Mutex for command handler access
    std::mutex m_command_handler_mutex;

    /**
     * @brief Main server loop for accepting connections
     *
     * Runs in a separate thread, continuously accepting new WebSocket
     * connections and spawning session handlers for each client.
     */
    void server_loop();

    /**
     * @brief Main broadcast loop for processing transcription queue
     *
     * Runs in a separate thread, continuously processing the transcription
     * queue and broadcasting messages to all connected clients.
     */
    void broadcast_loop();

    /**
     * @brief Handles individual WebSocket client session
     * @param socket TCP socket for the client connection
     * @param state Shared state for connection management
     *
     * Manages the complete lifecycle of a WebSocket client connection:
     * - Performs WebSocket handshake
     * - Handles incoming messages (audio data, control messages)
     * - Manages connection cleanup
     * - Processes client commands and audio data
     */
    void do_session(tcp::socket socket, std::shared_ptr<shared_state> state);

    /**
     * @brief Processes incoming JSON messages from clients
     * @param message_view String view of the received message
     * @param client_ws Pointer to the client WebSocket stream
     * @param state Shared state for connection management
     *
     * Parses and routes different types of messages:
     * - Audio data messages
     * - Control messages (reset, etc.)
     * - Configuration messages
     */
    void process_message(std::string_view message_view,
                         websocket::stream<tcp::socket>* client_ws,
                         std::shared_ptr<shared_state> state);

    /**
     * @brief Parses audio data from JSON message
     * @param json_message Parsed JSON message containing audio data
     * @return audio_data structure with parsed audio samples and metadata
     * @throws std::invalid_argument if message format is invalid
     */
    audio_data parse_audio_data(const nlohmann::json& json_message);

    /**
     * @brief Validates audio data parameters
     * @param audio Audio data to validate
     * @return true if audio data is valid, false otherwise
     */
    bool validate_audio_data(const audio_data& audio);

    /**
     * @brief Handles errors by logging and optionally calling error callback
     * @param error_message Description of the error
     * @param client_ws Pointer to the client WebSocket stream (may be nullptr)
     */
    void handle_error(const std::string& error_message, websocket::stream<tcp::socket>* client_ws = nullptr);
};
