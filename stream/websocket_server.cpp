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

#include "websocket_server.h"
#include "wstream_app.h"
#include "base64.h"
#include <iostream>
#include <algorithm>

void websocket_server::transcription_queue::push(const std::string& transcription) {
    m_queue.enqueue(transcription);
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_new_data.store(true);
    }
    m_cond.notify_one();
}

bool websocket_server::transcription_queue::pop(std::string& transcription) {
    return m_queue.try_dequeue(transcription);
}

bool websocket_server::transcription_queue::wait_and_pop(std::string& transcription,
                                                         int timeout_ms,
                                                         const std::atomic<bool>& is_running) {
    if (m_queue.try_dequeue(transcription)) {
        return true;
    }

    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_cond.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [this, &is_running] {
                            return m_new_data.load() || !is_running.load();
                        })) {
        m_new_data.store(false);
        if (!is_running.load()) {
            return false;
        }
        return m_queue.try_dequeue(transcription);
    }
    return false;
}

/**
 * @class websocket_server::shared_state
 * @brief Manages shared state between WebSocket connections
 *
 * Thread-safe container for managing active WebSocket connections
 * and providing broadcasting capabilities.
 */
class websocket_server::shared_state {
private:
    /// Set of active WebSocket connections
    std::set<websocket::stream<tcp::socket>*> m_connections;

    /// Mutex protecting the connections set
    std::mutex m_mutex;

    /// Pre-built JSON template for transcription messages
    nlohmann::json m_json_template;

public:
    /**
     * @brief Constructor - initializes JSON template
     */
    shared_state() {
        m_json_template["type"] = "transcribe";
    }

    /**
     * @brief Adds a WebSocket connection to the active set
     * @param ws Pointer to the WebSocket stream
     */
    void join(websocket::stream<tcp::socket>* ws) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_connections.insert(ws);
        std::cout << "Client connected. Total clients: " << m_connections.size() << std::endl;
    }

    /**
     * @brief Removes a WebSocket connection from the active set
     * @param ws Pointer to the WebSocket stream
     */
    void leave(websocket::stream<tcp::socket>* ws) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_connections.erase(ws);
        std::cout << "Client disconnected. Remaining clients: " << m_connections.size() << std::endl;
    }

    /**
     * @brief Checks if any clients are connected
     * @return true if at least one client is connected
     */
    bool has_clients() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return !m_connections.empty();
    }

    /**
     * @brief Gets the number of connected clients
     * @return Number of active connections
     */
    size_t get_client_count() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_connections.size();
    }

    /**
     * @brief Broadcasts a transcription to all connected clients
     * @param transcription The transcribed text to broadcast
     */
    void broadcast(const std::string& transcription) {
        if (transcription.empty()) return;

        auto json_message = m_json_template;
        json_message["content"] = transcription;
        std::string message = json_message.dump();

        broadcast_json_string(message);
    }

    /**
     * @brief Broadcasts a transcription with metadata to all connected clients
     * @param transcription The transcribed text to broadcast
     * @param session_id Session identifier
     * @param confidence Confidence score
     */
    void broadcast(const std::string& transcription,
                   const std::string& session_id,
                   float confidence) {
        if (transcription.empty()) return;

        auto json_message = m_json_template;
        json_message["content"] = transcription;

        if (!session_id.empty()) {
            json_message["session_id"] = session_id;
        }

        if (confidence > 0.0f) {
            json_message["confidence"] = confidence;
        }

        json_message["timestamp"] =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch()).count();

        std::string message = json_message.dump();
        broadcast_json_string(message);
    }

    /**
     * @brief Sends a JSON message to a specific client
     * @param client_ws Pointer to the client WebSocket stream
     * @param json_message JSON message to send
     * @return true if message sent successfully
     */
    bool send_to_client(websocket::stream<tcp::socket>* client_ws,
                        const nlohmann::json& json_message) {
        if (!client_ws) return false;

        std::string message = json_message.dump();

        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_connections.find(client_ws) == m_connections.end()) {
            return false; // Client not in active connections
        }

        try {
            client_ws->text(true);
            client_ws->write(net::buffer(message));
            return true;
        } catch (const std::exception& e) {
            std::cerr << "WebSocket Send Error: " << e.what() << std::endl;
            return false;
        }
    }

private:
    /**
     * @brief Internal method to broadcast a JSON string to all clients
     * @param message JSON message string to broadcast
     */
    void broadcast_json_string(const std::string& message) {
        std::vector<websocket::stream<tcp::socket>*> clients;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_connections.empty()) return;
            clients.assign(m_connections.begin(), m_connections.end());
        }

        for (auto ws : clients) {
            try {
                ws->text(true);
                ws->write(net::buffer(message));
            } catch (const std::exception& e) {
                std::cerr << "WebSocket Broadcast Error: " << e.what() << std::endl;
                // Note: Connection will be removed when the session detects the error
            }
        }
    }
};

websocket_server::websocket_server(uint16_t port)
    : m_port(port) {
    m_state = std::make_unique<shared_state>();
    m_queue = std::make_unique<transcription_queue>();

    // Set default error callback
    m_error_callback = [](const std::string& error, websocket::stream<tcp::socket>* client) {
        std::cerr << "WebSocket Error: " << error << std::endl;
    };
}

websocket_server::~websocket_server() {
    stop();
}

void websocket_server::start() {
    m_is_running = true;
    m_server_thread = std::thread(&websocket_server::server_loop, this);
    m_broadcast_thread = std::thread(&websocket_server::broadcast_loop, this);
}

void websocket_server::stop() {
    m_is_running = false;

    // Just detach - don't wait
    if (m_server_thread.joinable()) {
        m_server_thread.detach();
    }

    if (m_broadcast_thread.joinable()) {
        m_broadcast_thread.detach();
    }
}

void websocket_server::broadcast(const std::string& message) {
    if (m_state) {
        m_state->broadcast(message);
    }
}

void websocket_server::queue_transcription(const std::string& transcription) {
    if (m_queue) {
        m_queue->push(transcription);
    }
}

void websocket_server::queue_transcription(const std::string& transcription,
                                           const std::string& session_id,
                                           float confidence) {
    if (!m_queue) return;

    // Create enhanced JSON message
    nlohmann::json json_message;
    json_message["type"] = "transcribe";
    json_message["content"] = transcription;

    if (!session_id.empty()) {
        json_message["session_id"] = session_id;
    }

    if (confidence > 0.0f) {
        json_message["confidence"] = confidence;
    }

    json_message["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count();

    m_queue->push(json_message.dump());
}

void websocket_server::set_audio_callback(audio_callback_t callback) {
    std::lock_guard<std::mutex> lock(m_callback_mutex);
    m_audio_callback = callback;
}

void websocket_server::set_error_callback(error_callback_t callback) {
    std::lock_guard<std::mutex> lock(m_callback_mutex);
    m_error_callback = callback;
}

bool websocket_server::has_clients() const {
    return m_state ? m_state->has_clients() : false;
}

size_t websocket_server::get_client_count() const {
    return m_state ? m_state->get_client_count() : 0;
}

bool websocket_server::send_to_client(websocket::stream<tcp::socket>* client_ws,
                                      const nlohmann::json& message) {
    return m_state ? m_state->send_to_client(client_ws, message) : false;
}

void websocket_server::set_command_handler(command_handler_t handler) {
    std::lock_guard<std::mutex> lock(m_command_handler_mutex);
    m_command_handler = handler;
}

void websocket_server::server_loop() {
    try {
        net::io_context ioc;
        tcp::acceptor acceptor{ioc, {tcp::v4(), m_port}};
        acceptor.set_option(tcp::acceptor::reuse_address(true));

        std::cout << "WebSocket server is running on port " << m_port << "..." << std::endl;

        while (m_is_running) {
            tcp::socket socket{ioc};
            acceptor.accept(socket);
            if (!m_is_running) break;

            auto state_shared = std::shared_ptr<shared_state>(m_state.get(), [](shared_state*){});
            std::thread([this, socket = std::move(socket), state_shared]() mutable {
                do_session(std::move(socket), state_shared);
            }).detach();
        }

        ioc.stop();
    } catch (std::exception const& e) {
        if (m_is_running) {
            handle_error(std::string("Server loop error: ") + e.what());
        }
    }
}

void websocket_server::broadcast_loop() {
    std::string message_data;
    while (m_is_running) {
        if (m_queue->wait_and_pop(message_data, QUEUE_TIMEOUT_MS, m_is_running)) {
            // Check if it's a pre-formatted JSON string or plain text
            try {
                nlohmann::json test_parse = nlohmann::json::parse(message_data);
                // It's already JSON, broadcast as-is
                if (m_state) {
                    std::vector<websocket::stream<tcp::socket>*> clients;
                    // We need to manually broadcast the JSON string
                    // since our state->broadcast expects plain text
                    // Let's add a new method or modify the existing one

                    // For now, we'll extract the content if it's a simple transcription
                    if (test_parse.contains("content")) {
                        m_state->broadcast(test_parse["content"].get<std::string>());
                    }
                }
            } catch (const nlohmann::json::exception&) {
                // It's plain text, use regular broadcast
                if (m_state) {
                    m_state->broadcast(message_data);
                }
            }
        }
    }
}

void websocket_server::do_session(tcp::socket socket, std::shared_ptr<shared_state> state) {
    websocket::stream<tcp::socket> ws{std::move(socket)};

    // Add a unique ID for this session to track it
    static std::atomic<int> session_counter{0};
    int session_id = session_counter++;

    std::cout << "[Session " << session_id << "] Starting..." << std::endl;

    try {
        ws.auto_fragment(AUTO_FRAGMENT);
        ws.read_message_max(MAX_MESSAGE_SIZE);
        beast::get_lowest_layer(ws).set_option(tcp::no_delay(TCP_NO_DELAY));
        ws.set_option(websocket::stream_base::timeout::suggested(
            beast::role_type::server));

        ws.accept();
        std::cout << "[Session " << session_id << "] WebSocket handshake completed" << std::endl;

        state->join(&ws);

        while (m_is_running) {
            beast::flat_buffer buffer;
            beast::error_code ec;
            ws.read(buffer, ec);

            if (ec) {
                std::cout << "[Session " << session_id << "] Read error: "
                          << ec.message() << std::endl;
                break;
            }

            std::string_view message(
                static_cast<const char*>(buffer.data().data()),
                buffer.data().size());

            process_message(message, &ws, state);
        }

        std::cout << "[Session " << session_id << "] Exited read loop" << std::endl;

    } catch (beast::system_error const& se) {
        std::cout << "[Session " << session_id << "] Beast system error: "
                  << se.code().message() << std::endl;
        if (se.code() != websocket::error::closed && m_is_running) {
            handle_error(std::string("WebSocket system error: ") + se.code().message(), &ws);
        }
    } catch (std::exception const& e) {
        std::cout << "[Session " << session_id << "] Exception: " << e.what() << std::endl;
        if (m_is_running) {
            handle_error(std::string("WebSocket session error: ") + e.what(), &ws);
        }
    }

    std::cout << "[Session " << session_id << "] Calling state->leave()" << std::endl;
    state->leave(&ws);
    std::cout << "[Session " << session_id << "] Session ended" << std::endl;
}

void websocket_server::process_message(std::string_view message_view,
                                       websocket::stream<tcp::socket>* client_ws,
                                       std::shared_ptr<shared_state> state) {
    try {
        nlohmann::json json_message = nlohmann::json::parse(message_view);

        if (!json_message.contains("type") || !json_message["type"].is_string()) {
            handle_error("Message missing 'type' field", client_ws);
            return;
        }

        std::string message_type = json_message["type"].get<std::string>();

        if (message_type == "command") {
            // Handle command messages through callback
            if (!json_message.contains("action") || !json_message["action"].is_string()) {
                handle_error("Command message missing 'action' field", client_ws);
                return;
            }

            std::string action = json_message["action"].get<std::string>();

            // Call command handler if set
            std::lock_guard<std::mutex> lock(m_command_handler_mutex);
            if (m_command_handler) {
                try {
                    nlohmann::json response = m_command_handler(action, json_message, client_ws);
                    state->send_to_client(client_ws, response);
                } catch (const std::exception& e) {
                    nlohmann::json error_response = {
                        {"type", "response"},
                        {"action", action},
                        {"status", "error"},
                        {"message", std::string("Command handler error: ") + e.what()}
                    };
                    state->send_to_client(client_ws, error_response);
                }
            } else {
                handle_error("No command handler registered", client_ws);
            }
        } else if (message_type == "audio") {
            // Process audio data
            try {
                audio_data audio = parse_audio_data(json_message);

                if (!validate_audio_data(audio)) {
                    handle_error("Invalid audio data parameters", client_ws);
                    return;
                }

                // Call audio callback if set
                std::lock_guard<std::mutex> lock(m_callback_mutex);
                if (m_audio_callback) {
                    // Run callback in a separate thread to avoid blocking the session
                    std::thread([this, audio, client_ws]() {
                        try {
                            m_audio_callback(audio, client_ws);
                        } catch (const std::exception& e) {
                            handle_error(std::string("Audio callback error: ") + e.what(),
                                         client_ws);
                        }
                    }).detach();
                }
            } catch (const std::exception& e) {
                handle_error(std::string("Audio processing error: ") + e.what(), client_ws);
            }

        } else if (message_type == "reset") {
            // Handle reset message (existing functionality)
            std::string content;
            if (json_message.contains("content")) {
                content = json_message["content"].get<std::string>();
            }

            // Send acknowledgment
            nlohmann::json reset_response;
            reset_response["type"] = "reset_ack";
            reset_response["status"] = "completed";
            state->send_to_client(client_ws, reset_response);

        } else if (message_type == "ping") {
            // Handle ping message for keep-alive
            nlohmann::json pong_response;
            pong_response["type"] = "pong";
            pong_response["timestamp"] =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count();
            state->send_to_client(client_ws, pong_response);

        } else {
            handle_error("Unknown message type: " + message_type, client_ws);
        }

    } catch (const nlohmann::json::exception& e) {
        handle_error(std::string("JSON parsing error: ") + e.what(), client_ws);
    }
}

audio_data websocket_server::parse_audio_data(const nlohmann::json& json_message) {
    audio_data result;

    // Parse required audio samples
    if (!json_message.contains("audio")) {
        throw std::invalid_argument("Missing 'audio' field");
    }

    // Check if audio is Base64 encoded
    bool is_base64 = false;
    if (json_message.contains("encoding") && json_message["encoding"].is_string()) {
        is_base64 = json_message["encoding"].get<std::string>() == "base64";
    }

    if (is_base64) {
        if (!json_message["audio"].is_string()) {
            throw std::invalid_argument("Base64 audio data must be a string");
        }

        std::string base64_data = json_message["audio"].get<std::string>();

        try {
            // Use optimized fast decode for audio data
            if (!base64::decode_audio_fast(base64_data, result.samples)) {
                throw std::invalid_argument("Failed to decode Base64 audio data");
            }

        } catch (const std::exception& e) {
            throw std::invalid_argument(std::string("Base64 decoding error: ") + e.what());
        }
    } else {
        // Parse raw audio data
        if (!json_message["audio"].is_array()) {
            throw std::invalid_argument("Raw audio data must be an array");
        }

        const auto& audio_array = json_message["audio"];
        if (audio_array.size() > MAX_AUDIO_SAMPLES) {
            throw std::invalid_argument("Audio data exceeds maximum allowed size");
        }

        result.samples.reserve(audio_array.size());
        for (const auto& sample : audio_array) {
            if (!sample.is_number()) {
                throw std::invalid_argument("Audio sample is not a number");
            }
            result.samples.push_back(sample.get<int16_t>());
        }
    }

    // Parse optional fields
    if (json_message.contains("sample_rate") && json_message["sample_rate"].is_number()) {
        result.sample_rate = json_message["sample_rate"].get<uint32_t>();
    }

    if (json_message.contains("channels") && json_message["channels"].is_number()) {
        result.channels = json_message["channels"].get<uint16_t>();
    }

    if (json_message.contains("session_id") && json_message["session_id"].is_string()) {
        result.session_id = json_message["session_id"].get<std::string>();
    }

    if (json_message.contains("language") && json_message["language"].is_string()) {
        result.language = json_message["language"].get<std::string>();
    }

    if (json_message.contains("timestamp") && json_message["timestamp"].is_number()) {
        result.timestamp = json_message["timestamp"].get<uint64_t>();
    }

    return result;
}

bool websocket_server::validate_audio_data(const audio_data& audio) {
    if (!audio.is_valid()) {
        return false;
    }

    // Check reasonable sample rate range (8kHz to 48kHz)
    if (audio.sample_rate < 8000 || audio.sample_rate > 48000) {
        return false;
    }

    // Check reasonable channel count (1-2 channels)
    if (audio.channels < 1 || audio.channels > 2) {
        return false;
    }

    // Check audio duration (max 60 seconds)
    if (audio.duration_ms() > 60000) {
        return false;
    }

    return true;
}

void websocket_server::handle_error(const std::string& error_message,
                                    websocket::stream<tcp::socket>* client_ws) {
    std::lock_guard<std::mutex> lock(m_callback_mutex);
    if (m_error_callback) {
        m_error_callback(error_message, client_ws);
    } else {
        std::cerr << "WebSocket Error: " << error_message << std::endl;
    }
}
