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

        // Set timeout and connect - Compatible with older Boost versions
        auto& socket = m_ws->next_layer();

        // Simple connect without timeout for maximum compatibility
        // If you need timeout, you can implement it using async operations
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
    if (!m_running) {
        return;
    }

    std::cout << "Disconnecting from server..." << std::endl;

    // Set flags
    m_running = false;
    m_connected = false;

    // Force close the socket to unblock read operations
    if (m_ws) {
        try {
            beast::error_code ec;
            // Force close the underlying socket
            m_ws->next_layer().cancel(ec);
            m_ws->next_layer().shutdown(tcp::socket::shutdown_both, ec);
            m_ws->next_layer().close(ec);
        } catch (...) {
            // Ignore errors
        }
    }

    // Stop IO context
    m_ioc.stop();

    // Give thread a moment to finish
    if (m_client_thread && m_client_thread->joinable()) {
        // Wait briefly
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // If still running, detach it
        if (m_client_thread->joinable()) {
            std::cout << "Force detaching client thread" << std::endl;
            m_client_thread->detach();
        }
    }

    m_client_thread.reset();
    m_ws.reset();

    std::cout << "Disconnected from server" << std::endl;
}

bool websocket_client::send_audio_data(const std::vector<int16_t>& pcm_data,
                                       const std::string& session_id,
                                       const std::string& language) {
    if (!m_connected) {
        std::cerr << "Not connected to server" << std::endl;
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(m_write_mutex);
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

        m_ws->write(net::buffer(message_str));

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error sending audio data: " << e.what() << std::endl;
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
        std::cerr << "Client thread error: " << e.what() << std::endl;
        m_connected = false;

        std::lock_guard<std::mutex> lock(m_callback_mutex);
        if (m_error_callback) {
            m_error_callback(e.what());
        }
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
