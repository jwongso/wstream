#include "common.h"
#include "whisper.h"
#include "common-sdl.h"
#include <iostream>
#include <set>
#include <termios.h>
#include <thread>
#include <atomic>
#include <string>
#include <unistd.h>
#include <vector>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>

namespace fs = std::filesystem;
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = net::ip::tcp;

// Global flag for pause/resume
std::atomic<bool> is_running(true);
std::atomic<bool> is_cleared(false);

// Global vector to store transcriptions
std::vector<std::string> transcriptions;

// Shared state for WebSocket server
class shared_state {
    std::set<websocket::stream<tcp::socket>*> m_connections;
    std::mutex m_mutex;

public:
    void join(websocket::stream<tcp::socket>* ws) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_connections.insert(ws);
    }

    void leave(websocket::stream<tcp::socket>* ws) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_connections.erase(ws);
    }

    bool is_client_connected() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return !m_connections.empty();
    }

    void broadcast(const std::string& message) {
        std::vector<websocket::stream<tcp::socket>*> clients;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
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

// WebSocket session handler
void do_session(tcp::socket socket, std::shared_ptr<shared_state> state) {
    websocket::stream<tcp::socket> ws{std::move(socket)};
    try {
        ws.accept();

        state->join(&ws);

        while (is_running) {
            beast::flat_buffer buffer;
            ws.read(buffer);
            // Convert the message to a string
            std::string message = beast::buffers_to_string(buffer.data());

            // Parse the message as JSON
            nlohmann::json json_message = nlohmann::json::parse(message);

            // Check if the message is a prompt
            if (json_message["type"] == "reset") {
                // Handle reset command
                std::string content = json_message["content"];
                if (content == "clear") {
                    is_cleared = true;
                }
            }
        }
    } catch (beast::system_error const& se) {
        if (se.code() != websocket::error::closed) {
            std::cerr << "WebSocket Error: " << se.code().message() << std::endl;
        }
    } catch (std::exception const& e) {
        std::cerr << "WebSocket Error: " << e.what() << std::endl;
    }

    // Remove the client from the connections set
    state->leave(&ws);
}

// WebSocket server
void websocket_server(std::shared_ptr<shared_state> state) {
    try {
        net::io_context ioc;
        tcp::acceptor acceptor{ioc, {tcp::v4(), 8080}};

        std::cout << "WebSocket server is running on port 8080..." << std::endl;

        while (is_running) {
            tcp::socket socket{ioc};
            acceptor.accept(socket);
            if (!is_running) break;
            std::thread{do_session, std::move(socket), state}.detach();
        }

        ioc.stop();
    } catch (std::exception const& e) {
        std::cerr << "WebSocket Server Error: " << e.what() << std::endl;
    }
}

// Function to remove bracketed or parenthesised text
void remove_bracketed_text(std::string& text) {
    char* read = text.data();
    char* write = text.data();
    bool in_bracket = false;
    bool in_paren = false;

    while (*read) {
        if (!in_bracket && !in_paren) {
            if (*read == '[') {
                in_bracket = true;
                read++;
                continue;
            }
            if (*read == '(') {
                in_paren = true;
                read++;
                continue;
            }
            *write++ = *read++;
        } else {
            if (in_bracket && *read == ']') {
                in_bracket = false;
                read++;
                continue;
            }
            if (in_paren && *read == ')') {
                in_paren = false;
                read++;
                continue;
            }
            read++;
        }
    }
    *write = '\0';
    text.resize(write - text.data());
}

// Function to trim leading and trailing whitespace
void lrtrim(std::string &s) {
    const char* whitespace = " \t\n\r\f\v";
    size_t start = s.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        s.clear();
        return;
    }
    size_t end = s.find_last_not_of(whitespace);
    s.erase(end + 1);
    s.erase(0, start);
}

int main(int argc, char* argv[]) {
    // Initialize whisper context
    std::string model_path = "models/ggml-medium.en-q5_0.bin";

    // Check if a custom path was provided
    if (argc > 1) {
        std::string user_path = argv[1];

        // Validate the path
        if (fs::exists(user_path)) {
            model_path = user_path;
        } else {
            std::cerr << "Warning: Provided model path '" << user_path << "' does not exist. Falling back to default.\n";
        }
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Failed to initialize Whisper context.\n";
        return 1;
    }

    // Initialize audio capture
    audio_async audio(30000); // 10-second buffer (to accommodate 30-second chunks)
    if (!audio.init(-1, WHISPER_SAMPLE_RATE)) {
        std::cerr << "Failed to initialize audio capture.\n";
        return 1;
    }
    audio.resume();

    // Start the WebSocket server
    auto state = std::make_shared<shared_state>();
    std::thread ws_thread(websocket_server, state);

    // Main loop
    std::vector<float> pcmf32;
    pcmf32.reserve(WHISPER_SAMPLE_RATE * 15);

    // VAD parameters
    float vad_thold = 0.85f; // Increase to reduce sensitivity
    float freq_thold = 100.0f; // Frequency threshold for VAD

    while (is_running) {
        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }

        if (is_cleared) {
            pcmf32.clear();
            is_cleared = false;
        }

        // Capture audio
        audio.get(15000, pcmf32); // Get 30 seconds of audio

        // Run VAD to check for speech activity
        if (::vad_simple(pcmf32, WHISPER_SAMPLE_RATE, 1000, vad_thold, freq_thold, false)) {
            // Run inference only if speech is detected
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            wparams.print_progress = false;
            wparams.print_realtime = false;
            wparams.no_context = true; // Disable context carryover
            wparams.language = "en";
            wparams.max_tokens = 0;
            wparams.no_timestamps = true;
            wparams.n_threads = std::thread::hardware_concurrency();
            wparams.temperature = 0.1;
            wparams.greedy.best_of = 1;
            wparams.single_segment = true;

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                std::cerr << "Failed to process audio.\n";
                break;
            }

            // Get the latest transcription
            const int n_segments = whisper_full_n_segments(ctx);
            if (n_segments > 0) {
                std::string current_transcription;
                for (int i = 0; i < n_segments; ++i) {
                    const char* text = whisper_full_get_segment_text(ctx, i);
                    if (text) {
                        current_transcription += text;
                    }
                }

                // Remove partial bracketed text (e.g., [inaudible], [ Background Conversations ])
                remove_bracketed_text(current_transcription);

                // Trim leading and trailing whitespace
                lrtrim(current_transcription);

                // Skip if the transcription is empty after cleaning
                if (current_transcription.empty()) {
                    continue;
                }

                // Extract new content by comparing with the previous transcription
                const std::string new_content = current_transcription;

                // Add the new content to the transcriptions vector if it's not empty
                if (!new_content.empty()) {
                    transcriptions.push_back(new_content);
                    std::cout << new_content << std::endl; // Print the new content

                    nlohmann::json transcribe_message = {
                        {"type", "transcribe"},
                        {"content", new_content}
                    };

                    // Broadcast the new content to WebSocket clients
                    state->broadcast(transcribe_message.dump());
                }
            }
        }
    }

    std::cout << "CTRL-C again to exit..." << std::endl;
    audio.pause();
    SDL_Quit();
    is_running = false;
    if (ws_thread.joinable()) ws_thread.join();

    whisper_free(ctx);

    return 0;
}
