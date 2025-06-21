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

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = net::ip::tcp;

/**
 * @file websocket_server.h
 * @brief WebSocket server for real-time transcription broadcasting
 * @author WStream Development Team
 * @version 1.0
 * @date 2024
 */

/**
 * @class websocket_server
 * @brief High-performance WebSocket server for broadcasting transcriptions
 *
 * This class implements a multi-threaded WebSocket server optimized for
 * real-time broadcasting of speech transcription results. Key features:
 *
 * - **Multi-client Support**: Handles multiple concurrent WebSocket connections
 * - **Lock-free Queuing**: Uses concurrent queue for high-performance message passing
 * - **Automatic Broadcasting**: Efficiently broadcasts messages to all connected clients
 * - **Connection Management**: Handles client connections, disconnections, and errors
 * - **JSON Protocol**: Structured message format for client communication
 * - **Thread Safety**: Fully thread-safe for concurrent access
 *
 * @par Message Protocol:
 * The server uses JSON messages with the following structure:
 * ```json
 * {
 *   "type": "transcribe",
 *   "content": "transcribed text here"
 * }
 * ```
 *
 * @par Performance Characteristics:
 * - Lock-free message queuing for minimal latency
 * - Efficient broadcast to multiple clients
 * - Pre-allocated JSON templates to reduce overhead
 * - Configurable connection limits and timeouts
 * - Optimized for real-time streaming applications
 *
 * @par Thread Safety:
 * Fully thread-safe. Multiple threads can safely:
 * - Queue transcriptions for broadcast
 * - Check client connection status
 * - Start/stop the server
 */
class websocket_server {
public:
    /// Default WebSocket server port
    static constexpr uint16_t DEFAULT_PORT = 8080;

    /// Maximum WebSocket message size (64KB)
    static constexpr size_t MAX_MESSAGE_SIZE = 64 * 1024;

    /// Queue timeout for waiting for new transcriptions (milliseconds)
    static constexpr int QUEUE_TIMEOUT_MS = 100;

    /// TCP no-delay option for lower latency
    static constexpr bool TCP_NO_DELAY = true;

    /// WebSocket auto-fragment setting
    static constexpr bool AUTO_FRAGMENT = false;

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
     * @brief Checks if any clients are currently connected
     * @return true if at least one client is connected, false otherwise
     *
     * Thread-safe check for client connectivity. Useful for optimizing
     * processing when no clients are listening.
     */
    bool has_clients() const;

private:
    class shared_state; ///< Forward declaration for shared connection state

    /// Server port number
    uint16_t m_port;

    /// Atomic flag controlling server operation
    std::atomic<bool>m_is_running{true};

    /// Shared state for managing client connections
    std::unique_ptr<shared_state> m_state;

    /// Queue for transcription messages
    std::unique_ptr<transcription_queue> m_queue;

    /// Thread handling incoming connections
    std::thread m_server_thread;

    /// Thread handling message broadcasting
    std::thread m_broadcast_thread;

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
     * - Handles incoming messages
     * - Manages connection cleanup
     * - Processes client commands (e.g., reset)
     */
    void do_session(tcp::socket socket, std::shared_ptr<shared_state> state);
};
