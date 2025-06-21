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
#include <iostream>

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

class websocket_server::shared_state {
private:
    std::set<websocket::stream<tcp::socket>*> m_connections;
    std::mutex m_mutex;
    nlohmann::json m_json_template;

public:
    shared_state() {
        m_json_template["type"] = "transcribe";
    }

    void join(websocket::stream<tcp::socket>* ws) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_connections.insert(ws);
    }

    void leave(websocket::stream<tcp::socket>* ws) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_connections.erase(ws);
    }

    bool has_clients() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return !m_connections.empty();
    }

    void broadcast(const std::string& transcription) {
        if (transcription.empty()) return;

        auto json_message = m_json_template;
        json_message["content"] = transcription;
        std::string message = json_message.dump();

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
            }
        }
    }
};

websocket_server::websocket_server(uint16_t port)
    : m_port(port) {
    m_state = std::make_unique<shared_state>();
    m_queue = std::make_unique<transcription_queue>();
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

bool websocket_server::has_clients() const {
    return m_state ? m_state->has_clients() : false;
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
            std::cerr << "WebSocket Server Error: " << e.what() << std::endl;
        }
    }
}

void websocket_server::broadcast_loop() {
    std::string transcription;
    while (m_is_running) {
        if (m_queue->wait_and_pop(transcription, QUEUE_TIMEOUT_MS, m_is_running)) {
            m_state->broadcast(transcription);
        }
    }
}

void websocket_server::do_session(tcp::socket socket, std::shared_ptr<shared_state> state) {
    websocket::stream<tcp::socket> ws{std::move(socket)};
    try {
        ws.auto_fragment(AUTO_FRAGMENT);
        ws.read_message_max(MAX_MESSAGE_SIZE);
        beast::get_lowest_layer(ws).set_option(tcp::no_delay(TCP_NO_DELAY));

        ws.accept();
        state->join(&ws);

        while (m_is_running) {
            beast::flat_buffer buffer;
            ws.read(buffer);

            std::string_view message(
                static_cast<const char*>(buffer.data().data()),
                buffer.data().size());

            try {
                nlohmann::json json_message = nlohmann::json::parse(message);
                if (json_message["type"] == "reset") {
                    std::string content = json_message["content"];
                }
            } catch (const nlohmann::json::exception& e) {
                std::cerr << "JSON parsing error: " << e.what() << std::endl;
            }
        }
    } catch (beast::system_error const& se) {
        if (se.code() != websocket::error::closed) {
            if (m_is_running) {
                std::cerr << "WebSocket Error: " << se.code().message() << std::endl;
            }
        }
    } catch (std::exception const& e) {
        if (m_is_running) {
            std::cerr << "WebSocket Error: " << e.what() << std::endl;
        }
    }

    state->leave(&ws);
}
