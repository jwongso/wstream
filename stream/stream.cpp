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

// Store last transcription
std::string last_transcription;

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

// Calculate Levenshtein distance between two strings
float levenshtein_distance(const std::string& s1, const std::string& s2) {
    const size_t len1 = s1.size();
    const size_t len2 = s2.size();
    std::vector<std::vector<size_t>> d(len1 + 1, std::vector<size_t>(len2 + 1));

    for (size_t i = 0; i <= len1; ++i) d[i][0] = i;
    for (size_t j = 0; j <= len2; ++j) d[0][j] = j;

    for (size_t i = 1; i <= len1; ++i) {
        for (size_t j = 1; j <= len2; ++j) {
            size_t cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            d[i][j] = std::min({d[i - 1][j] + 1,      // Deletion
                                d[i][j - 1] + 1,      // Insertion
                                d[i - 1][j - 1] + cost}); // Substitution
        }
    }

    return static_cast<float>(d[len1][len2]);
}

// Calculate similarity score (0 = identical, 1 = completely different)
float string_similarity(const std::string& s1, const std::string& s2) {
    if (s1.empty() || s2.empty()) return 1.0f;
    float max_len = static_cast<float>(std::max(s1.size(), s2.size()));
    return levenshtein_distance(s1, s2) / max_len;
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
    cparams.use_gpu = true;
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Failed to initialize Whisper context.\n";
        return 1;
    }

    // Initialize audio capture
    audio_async audio(10000); // 10-second buffer (to accommodate 30-second chunks)

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
    pcmf32.reserve(WHISPER_SAMPLE_RATE * 30);

    // VAD parameters
    float vad_thold = 0.85f; // Increase to reduce sensitivity
    float freq_thold = 250.0f; // Frequency threshold for VAD

    while (is_running) {
        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }

        // Capture audio
        audio.get(5000, pcmf32); // Get 5 seconds of audio
        if (pcmf32.empty()) continue;

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
            wparams.temperature = 0.0f;
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

                if (last_transcription.empty() ||
                    string_similarity(last_transcription, current_transcription) > 0.1f) {

                    std::cout << current_transcription << std::endl; // Print the new content

                    nlohmann::json transcribe_message = {
                        {"type", "transcribe"},
                        {"content", current_transcription}
                    };

                    // Broadcast the new content to WebSocket clients
                    state->broadcast(transcribe_message.dump());

                    last_transcription = current_transcription;
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
