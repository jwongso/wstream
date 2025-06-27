#include "websocket_client.h"
#include "base64.h"
#include <iostream>
#include <chrono>

websocket_client::websocket_client() {
    // Set logging to be quiet
    m_client.set_access_channels(websocketpp::log::alevel::none);
    m_client.set_error_channels(websocketpp::log::elevel::warn);

    // Initialize ASIO
    m_client.init_asio();

    // Set handlers
    m_client.set_open_handler([this](websocketpp::connection_hdl hdl) {
        on_open(hdl);
    });

    m_client.set_close_handler([this](websocketpp::connection_hdl hdl) {
        on_close(hdl);
    });

    m_client.set_message_handler([this](websocketpp::connection_hdl hdl, message_ptr msg) {
        on_message(hdl, msg);
    });

    m_client.set_fail_handler([this](websocketpp::connection_hdl hdl) {
        on_fail(hdl);
    });
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
        websocketpp::lib::error_code ec;
        connection_ptr con = m_client.get_connection(uri, ec);

        if (ec) {
            std::cerr << "Could not create connection: " << ec.message() << std::endl;
            return false;
        }

        m_connection_hdl = con->get_handle();
        m_client.connect(con);

        // Start client thread
        m_running = true;
        m_client_thread = std::make_unique<std::thread>(&websocket_client::client_thread_func, this);

        // Wait for connection with timeout
        auto start_time = std::chrono::steady_clock::now();
        while (!m_connected && m_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed > std::chrono::seconds(10)) {
                std::cerr << "Connection timeout" << std::endl;
                disconnect();
                return false;
            }
        }

        return m_connected;

    } catch (const std::exception& e) {
        std::cerr << "Connection error: " << e.what() << std::endl;
        return false;
    }
}

void websocket_client::disconnect() {
    if (!m_running) {
        return;
    }

    m_running = false;
    m_connected = false;

    try {
        if (m_client.get_alog().static_test(websocketpp::log::alevel::devel)) {
            m_client.get_alog().write(websocketpp::log::alevel::devel, "Closing connection");
        }

        websocketpp::lib::error_code ec;
        m_client.close(m_connection_hdl, websocketpp::close::status::going_away, "Client disconnecting", ec);

        if (ec) {
            std::cerr << "Error closing connection: " << ec.message() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during disconnect: " << e.what() << std::endl;
    }

    if (m_client_thread && m_client_thread->joinable()) {
        m_client_thread->join();
    }

    m_client_thread.reset();

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
        json audio_message;
        audio_message["type"] = "audio";
        audio_message["sample_rate"] = 16000;
        audio_message["channels"] = 1;
        audio_message["session_id"] = session_id;
        audio_message["language"] = language;
        audio_message["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::system_clock::now().time_since_epoch()).count();

        // Use optimized Base64 encoding if enabled
        if (m_use_base64) {
            audio_message["encoding"] = "base64";

            // Use optimized template function
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
            // Log message without audio data for readability
            json log_message = audio_message;
            log_message["audio"] = "<" + std::to_string(pcm_data.size()) + " samples>";
            debug_log("Sending message: " + log_message.dump());
        }

        websocketpp::lib::error_code ec;
        m_client.send(m_connection_hdl, message_str, websocketpp::frame::opcode::text, ec);

        if (ec) {
            std::cerr << "Send error: " << ec.message() << std::endl;
            return false;
        }

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

        websocketpp::lib::error_code ec;
        m_client.send(m_connection_hdl, message_str, websocketpp::frame::opcode::text, ec);

        if (ec) {
            std::cerr << "Send error: " << ec.message() << std::endl;
            return false;
        }

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

        websocketpp::lib::error_code ec;
        m_client.send(m_connection_hdl, message_str, websocketpp::frame::opcode::text, ec);

        if (ec) {
            std::cerr << "Send error: " << ec.message() << std::endl;
            return false;
        }

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

void websocket_client::on_open(websocketpp::connection_hdl /*hdl*/) {
    m_connected = true;
    std::cout << "Connected to server" << std::endl;
}

void websocket_client::on_close(websocketpp::connection_hdl /*hdl*/) {
    m_connected = false;
    std::cout << "Connection closed" << std::endl;
}

void websocket_client::on_message(websocketpp::connection_hdl hdl, message_ptr msg) {
    try {
        std::string payload = msg->get_payload();

        if (m_verbose) {
            debug_log("Received message: " + payload);
        }

        json response = json::parse(payload);

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
                // Audio acknowledgment - could be used for flow control
            } else if (type == "response") {
                if (m_verbose) {
                    debug_log("Command response: " + response.dump());
                }

                std::lock_guard<std::mutex> lock(m_callback_mutex);
                if (m_response_callback) {
                    m_response_callback(response);
                } else {
                    // Default response handling
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
            debug_log("Failed to parse message: " + msg->get_payload());
        }
    } catch (const std::exception& e) {
        std::cerr << "Message handling error: " << e.what() << std::endl;
    }
}

void websocket_client::on_fail(websocketpp::connection_hdl /*hdl*/) {
    std::cerr << "Connection failed" << std::endl;
    m_connected = false;

    std::lock_guard<std::mutex> lock(m_callback_mutex);
    if (m_error_callback) {
        m_error_callback("Connection failed");
    }
}

void websocket_client::client_thread_func() {
    try {
        m_client.run();
    } catch (const std::exception& e) {
        std::cerr << "Client thread error: " << e.what() << std::endl;
    }

    m_running = false;
    m_connected = false;
}

void websocket_client::debug_log(const std::string& message) {
    if (m_verbose) {
        std::cout << "[DEBUG] " << message << std::endl;
    }
}
