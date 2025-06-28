#include "websocket_client.h"
#include "base64.h"
#include <iostream>
#include <chrono>

websocket_client::websocket_client() : m_ioc(1) {
    // Initialize WebSocket
    m_ws = std::make_unique<websocket::stream<tcp::socket>>(m_ioc);
}

websocket_client::~websocket_client() {
    disconnect();
}

bool websocket_client::connect(const std::string& uri) {
    if (m_connected) {
        std::cerr << "Already connected" << std::endl;
        return false;
    }

    try {
        // Parse URI
        std::string host;
        std::string port = "80";
        std::string path = "/";

        // Simple URI parsing
        size_t protocol_end = uri.find("://");
        if (protocol_end == std::string::npos) {
            throw std::runtime_error("Invalid URI format");
        }

        size_t host_start = protocol_end + 3;
        size_t port_sep = uri.find(':', host_start);
        size_t path_sep = uri.find('/', host_start);

        if (port_sep != std::string::npos && (path_sep == std::string::npos || port_sep < path_sep)) {
            host = uri.substr(host_start, port_sep - host_start);
            port = uri.substr(port_sep + 1, path_sep - port_sep - 1);
        } else {
            host = uri.substr(host_start, path_sep - host_start);
        }

        if (path_sep != std::string::npos) {
            path = uri.substr(path_sep);
        }

        // Resolve host
        tcp::resolver resolver(m_ioc);
        auto const results = resolver.resolve(host, port);

        // Connect
        auto& socket = m_ws->next_layer();
        auto ep = net::connect(socket, results);

        // Update the host for the handshake if needed
        host += ':' + std::to_string(ep.port());

        // WebSocket handshake
        beast::error_code ec;
        m_ws->handshake(host, path, ec);
        if (ec) {
            throw boost::system::system_error(ec);
        }

        // Success
        m_connected = true;
        m_running = true;
        m_client_thread = std::make_unique<std::thread>(&websocket_client::client_thread_func, this);

        std::cout << "Connected to server" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Connection error: " << e.what() << std::endl;
        m_connected = false;
        return false;
    }
}

void websocket_client::disconnect() {
    // Use atomic flag to ensure single execution
    bool expected = true;
    if (!m_running.compare_exchange_strong(expected, false)) {
        // Already disconnecting
        return;
    }

    std::cout << "Disconnecting from server..." << std::endl;
    m_connected.store(false);

    // Stop the IO context first to unblock any pending reads
    m_ioc.stop();

    // Wait briefly for the client thread to notice
    if (m_client_thread && m_client_thread->joinable()) {
        // Give it a moment to exit cleanly
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // If still running, detach it
        if (m_client_thread->joinable()) {
            m_client_thread->detach();
        }
    }

    // Now try to close the WebSocket if it's still open
    if (m_ws && m_ws->is_open()) {
        try {
            beast::error_code ec;
            // Use the non-blocking close with error code
            m_ws->close(websocket::close_code::normal, ec);
            // We don't care about the error - we're shutting down anyway
        } catch (...) {
            // Ignore any exceptions
        }
    }

    // Force close the underlying socket
    if (m_ws) {
        try {
            beast::error_code ec;
            m_ws->next_layer().cancel(ec);
            m_ws->next_layer().close(ec);
        } catch (...) {
            // Ignore errors
        }
    }

    m_client_thread.reset();
    m_ws.reset();

    std::cout << "Disconnected from server" << std::endl;
}

bool websocket_client::send_audio_data(const std::vector<int16_t>& pcm_data,
                                       const std::string& session_id,
                                       const std::string& language) {
    if (!m_connected.load() || !m_running.load()) {
        // Silently return false during shutdown
        return false;
    }

    try {
        json audio_message;
        audio_message["type"] = "audio";
        audio_message["sample_rate"] = 16000;
        audio_message["channels"] = 1;
        audio_message["session_id"] = session_id;
        audio_message["language"] = language;
        audio_message["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::system_clock::now().time_since_epoch()).count();

        if (m_use_base64) {
            audio_message["encoding"] = "base64";
            audio_message["audio"] = base64::encode(pcm_data);

            if (m_verbose) {
                size_t original_size = pcm_data.size() * sizeof(int16_t);
                size_t encoded_size = audio_message["audio"].get<std::string>().size();
                debug_log("Sending " + std::to_string(pcm_data.size()) +
                          " audio samples (" + std::to_string(original_size) +
                          " bytes, Base64 size: " + std::to_string(encoded_size) + " bytes)");
            }
        } else {
            audio_message["encoding"] = "raw";
            audio_message["audio"] = pcm_data;

            if (m_verbose) {
                debug_log("Sending " + std::to_string(pcm_data.size()) + " audio samples (raw)");
            }
        }

        std::string message_str = audio_message.dump();

        if (m_verbose) {
            json log_message = audio_message;
            log_message["audio"] = "<" + std::to_string(pcm_data.size()) + " samples>";
            debug_log("Sending message: " + log_message.dump());
        }

        // Check again before sending
        if (!m_connected.load() || !m_running.load()) {
            return false;
        }

        std::lock_guard<std::mutex> lock(m_write_mutex);
        m_ws->write(net::buffer(message_str));

        return true;

    } catch (const std::exception& e) {
        std::string error_msg = e.what();

        // Check if it's a shutdown-related error
        if (error_msg.find("Operation canceled") != std::string::npos ||
            error_msg.find("Bad file descriptor") != std::string::npos ||
            error_msg.find("Connection reset") != std::string::npos) {
            // Expected during shutdown, don't log as error
            m_connected.store(false);
            return false;
        }

        // Log other errors
        std::cerr << "Error sending audio data: " << error_msg << std::endl;
        return false;
    }
}

void websocket_client::set_transcription_callback(transcription_callback_t callback) {
    std::lock_guard<std::mutex> lock(m_callback_mutex);
    m_transcription_callback = callback;
}

void websocket_client::set_error_callback(error_callback_t callback) {
    std::lock_guard<std::mutex> lock(m_callback_mutex);
    m_error_callback = callback;
}

bool websocket_client::set_audio_source(const std::string& source_type) {
    if (!m_connected) {
        std::cerr << "Not connected to server" << std::endl;
        return false;
    }

    try {
        json command = {
            {"type", "command"},
            {"action", "set_audio_source"},
            {"source", source_type}
        };

        std::string message_str = command.dump();

        std::lock_guard<std::mutex> lock(m_write_mutex);
        m_ws->write(net::buffer(message_str));

        std::cout << "Sent command to set audio source to: " << source_type << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error sending set_audio_source command: " << e.what() << std::endl;
        return false;
    }
}

bool websocket_client::get_server_status() {
    if (!m_connected) {
        std::cerr << "Not connected to server" << std::endl;
        return false;
    }

    try {
        json command = {
            {"type", "command"},
            {"action", "get_status"}
        };

        std::string message_str = command.dump();

        std::lock_guard<std::mutex> lock(m_write_mutex);
        m_ws->write(net::buffer(message_str));

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error sending get_status command: " << e.what() << std::endl;
        return false;
    }
}

void websocket_client::set_response_callback(response_callback_t callback) {
    std::lock_guard<std::mutex> lock(m_callback_mutex);
    m_response_callback = callback;
}

void websocket_client::client_thread_func() {
    try {
        read_messages();
    } catch (const std::exception& e) {
        std::string error_msg = e.what();

        // Only log non-shutdown errors
        if (m_running.load() &&
            error_msg.find("Operation canceled") == std::string::npos &&
            error_msg.find("Bad file descriptor") == std::string::npos) {
            std::cerr << "Client thread error: " << error_msg << std::endl;

            std::lock_guard<std::mutex> lock(m_callback_mutex);
            if (m_error_callback) {
                m_error_callback(error_msg);
            }
        }

        m_connected.store(false);
    }
}

void websocket_client::read_messages() {
    beast::flat_buffer buffer;

    while (m_running) {
        try {
            // Read a message into our buffer
            m_ws->read(buffer);

            // Handle the message
            std::string message = beast::buffers_to_string(buffer.data());
            handle_message(message);

            // Clear the buffer for next message
            buffer.consume(buffer.size());
        } catch (const beast::system_error& se) {
            if (se.code() != websocket::error::closed) {
                throw;
            }
            break;
        }
    }
}

void websocket_client::handle_message(const std::string& message) {
    try {
        if (m_verbose) {
            debug_log("Received message: " + message);
        }

        json response = json::parse(message);

        if (response.contains("type")) {
            std::string type = response["type"];

            if (type == "transcribe" && response.contains("content")) {
                std::string transcription = response["content"];

                if (m_verbose) {
                    debug_log("Transcription received: " + transcription);
                }

                std::lock_guard<std::mutex> lock(m_callback_mutex);
                if (m_transcription_callback) {
                    m_transcription_callback(transcription);
                }
            } else if (type == "audio_ack") {
                if (m_verbose) {
                    debug_log("Audio acknowledgment: " + response.dump());
                }
            } else if (type == "response") {
                if (m_verbose) {
                    debug_log("Command response: " + response.dump());
                }

                std::lock_guard<std::mutex> lock(m_callback_mutex);
                if (m_response_callback) {
                    m_response_callback(response);
                } else {
                    if (response.contains("action") && response.contains("status")) {
                        std::string action = response["action"];
                        std::string status = response["status"];

                        std::cout << "Server response for '" << action << "': " << status;

                        if (response.contains("message")) {
                            std::cout << " - " << response["message"].get<std::string>();
                        }

                        std::cout << std::endl;
                    } else {
                        std::cout << "Server response: " + response.dump() << std::endl;
                    }
                }
            }
        }
    } catch (const json::exception& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        if (m_verbose) {
            debug_log("Failed to parse message: " + message);
        }
    } catch (const std::exception& e) {
        std::cerr << "Message handling error: " << e.what() << std::endl;
    }
}

void websocket_client::debug_log(const std::string& message) {
    if (m_verbose) {
        std::cout << "[DEBUG] " << message << std::endl;
    }
}
