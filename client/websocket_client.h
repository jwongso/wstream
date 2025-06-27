#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <memory>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

/**
 * @class websocket_client
 * @brief WebSocket client for connecting to wstream server (Boost.Beast implementation)
 */
class websocket_client {
public:
    /// Transcription callback type
    using transcription_callback_t = std::function<void(const std::string&)>;

    /// Error callback type
    using error_callback_t = std::function<void(const std::string&)>;

    /// Response callback type
    using response_callback_t = std::function<void(const json&)>;

    /**
     * @brief Constructor
     */
    websocket_client();

    /**
     * @brief Destructor
     */
    ~websocket_client();

    /**
     * @brief Connect to WebSocket server
     * @param uri Server URI (e.g., "ws://localhost:8080")
     * @return true if connection successful, false otherwise
     */
    bool connect(const std::string& uri);

    /**
     * @brief Disconnect from server
     */
    void disconnect();

    /**
     * @brief Send audio data to server
     * @param pcm_data 16-bit PCM audio samples
     * @param session_id Session identifier
     * @param language Language code
     * @return true if sent successfully, false otherwise
     */
    bool send_audio_data(const std::vector<int16_t>& pcm_data,
                         const std::string& session_id = "client-session",
                         const std::string& language = "en");

    /**
     * @brief Send command to set audio source on server
     * @param source_type Audio source type ("websocket", "microphone", "auto")
     * @return true if sent successfully, false otherwise
     */
    bool set_audio_source(const std::string& source_type);

    /**
     * @brief Send command to get server status
     * @return true if sent successfully, false otherwise
     */
    bool get_server_status();

    /**
     * @brief Set transcription callback
     * @param callback Function to call when transcription is received
     */
    void set_transcription_callback(transcription_callback_t callback);

    /**
     * @brief Set error callback
     * @param callback Function to call when error occurs
     */
    void set_error_callback(error_callback_t callback);

    /**
     * @brief Set response callback for command responses
     * @param callback Function to call when command response is received
     */
    void set_response_callback(response_callback_t callback);

    /**
     * @brief Check if connected
     * @return true if connected, false otherwise
     */
    bool is_connected() const { return m_connected.load(); }

    /**
     * @brief Toggle verbose mode for debugging
     * @param verbose Whether to enable verbose output
     */
    void set_verbose(bool verbose) { m_verbose = verbose; }

    /**
     * @brief Sets whether to use Base64 encoding for audio data
     * @param use_base64 Whether to use Base64 encoding
     */
    void set_use_base64(bool use_base64) { m_use_base64 = use_base64; }

private:
    /// IO context
    net::io_context m_ioc;

    /// WebSocket stream
    std::unique_ptr<websocket::stream<tcp::socket>> m_ws;

    /// Client thread
    std::unique_ptr<std::thread> m_client_thread;

    /// Connection state
    std::atomic<bool> m_connected{false};

    /// Running state
    std::atomic<bool> m_running{false};

    /// Callbacks
    transcription_callback_t m_transcription_callback;
    error_callback_t m_error_callback;
    response_callback_t m_response_callback;

    /// Callback mutex
    std::mutex m_callback_mutex;

    /// Write mutex
    std::mutex m_write_mutex;

    /// Verbose mode flag
    bool m_verbose = false;

    /// Base64 encoding flag
    bool m_use_base64 = true;

    /**
     * @brief Client thread function
     */
    void client_thread_func();

    /**
     * @brief Read messages from server
     */
    void read_messages();

    /**
     * @brief Parse and handle incoming message
     * @param message Received message
     */
    void handle_message(const std::string& message);

    /**
     * @brief Log debug message if verbose mode is enabled
     * @param message Message to log
     */
    void debug_log(const std::string& message);
};
