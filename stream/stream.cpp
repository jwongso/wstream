#include "config.h"
#include "whisper.h"
#include "common.h"
#include "common-sdl.h"
#include "chatapi.h"
#include <iostream>
#include <set>
#include <termios.h>
#include <thread>
#include <atomic>
#include <string>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <nlohmann/json.hpp>


namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = net::ip::tcp;
using namespace hyni;

// Global flag for pause/resume
std::atomic<bool> is_paused(false);
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
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto ws : m_connections) {
            ws->text(true);
            ws->write(net::buffer(message));
        }
    }
};

void process_prompt(const std::string& combined_transcription, bool is_star, ChatAPI& chatapi, std::shared_ptr<shared_state> state) {
    // Create a JSON message for the prompt
    nlohmann::json prompt_message = {
        {"type", "prompt"},
        {"content", combined_transcription}
    };

    // Print to console
    std::cout << " -------------------------- " << std::endl
              << "Prompt: " << combined_transcription << std::endl
              << " -------------------------- " << std::endl;

    // Send the prompt message to WebSocket clients
    state->broadcast(prompt_message.dump());

    // Send the combined transcription to the ChatAPI
    std::string response = chatapi.sendMessage(combined_transcription,
                                               is_star ? ChatAPI::QuestionType::AmazonBehavioral : ChatAPI::QuestionType::General);
    std::string reply = chatapi.getAssistantReply(response);

    // Create a JSON message for the assistant's response
    nlohmann::json response_message = {
        {"type", "response"},
        {"content", reply}
    };

    // Print to console
    std::cout << " -------------------------- " << std::endl
              << "Assistant: " << reply << std::endl
              << " -------------------------- " << std::endl;

    // Send the response message to WebSocket clients
    state->broadcast(response_message.dump());
}

// WebSocket session handler
void do_session(tcp::socket socket, std::shared_ptr<shared_state> state, ChatAPI& chatapi) {
    websocket::stream<tcp::socket> ws{std::move(socket)};
    try {
        ws.accept();

        state->join(&ws);

        for (;;) {
            beast::flat_buffer buffer;
            ws.read(buffer);
            // Convert the message to a string
            std::string message = beast::buffers_to_string(buffer.data());

            // Parse the message as JSON
            nlohmann::json json_message = nlohmann::json::parse(message);

            // Check if the message is a prompt
            if (json_message["type"] == "prompt") {
                std::string combined_transcription = json_message["content"];
                bool is_star = json_message["star"];

                process_prompt(combined_transcription, is_star, chatapi, state);
            } else if (json_message["type"] == "reset") {
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
void websocket_server(std::shared_ptr<shared_state> state, ChatAPI& chatapi) {
    try {
        net::io_context ioc;
        tcp::acceptor acceptor{ioc, {tcp::v4(), 8080}};

        std::cout << "WebSocket server is running on port 8080..." << std::endl;

        for (;;) {
            tcp::socket socket{ioc};
            acceptor.accept(socket);
            std::thread{do_session, std::move(socket), state, std::ref(chatapi)}.detach();
        }
    } catch (std::exception const& e) {
        std::cerr << "WebSocket Server Error: " << e.what() << std::endl;
    }
}

// Function to listen for key presses
void key_listener(ChatAPI& chatapi, audio_async& audio, std::shared_ptr<shared_state> state) {
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);  // Disable buffering and echoing
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    while (true) {
        char ch = getchar();
        if (ch == 's') {
            // Check if a WebSocket client is connected
            if (state->is_client_connected()) {
                std::cout << "WebSocket client is connected. Waiting for prompt from client..." << std::endl;
            } else {
                // No WebSocket client connected, proceed with keypress behavior
                if (!transcriptions.empty()) {
                    std::string combined_transcription;
                    for (const auto& text : transcriptions) {
                        combined_transcription += text + " ";
                    }

                    // Use the helper function to process the prompt
                    process_prompt(combined_transcription, false, chatapi, state);

                    // Clear the transcriptions vector and audio buffer
                    transcriptions.clear();
                    audio.clear();
                }
            }
        }
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);  // Restore terminal settings
}
// Function to extract new content from the transcription
std::string extract_new_content(const std::string& previous, const std::string& current) {
    if (previous.empty()) {
        return current; // No previous transcription, return the entire current transcription
    }

    // Find the position where the previous transcription ends in the current transcription
    auto pos = std::search(current.begin(), current.end(), previous.begin(), previous.end());
    if (pos != current.end()) {
        // Extract the new content after the previous transcription
        return std::string(pos + previous.length(), current.end());
    }

    // No match found, return the entire current transcription
    return current;
}

// Function to remove partial bracketed text (e.g., [inaudible], [ Background Conversations ])
std::string remove_bracketed_text(const std::string& text) {
    std::string cleaned_text = text;
    size_t start_pos = cleaned_text.find('[');
    while (start_pos != std::string::npos) {
        size_t end_pos = cleaned_text.find(']', start_pos);
        if (end_pos != std::string::npos) {
            // Remove the bracketed text
            cleaned_text.erase(start_pos, end_pos - start_pos + 1);
        } else {
            break; // No closing bracket found
        }
        start_pos = cleaned_text.find('[', start_pos);
    }
    return cleaned_text;
}

// Function to trim leading and trailing whitespace
std::string lrtrim(const std::string& text) {
    const char* whitespace = " \t\n\r";
    size_t start = text.find_first_not_of(whitespace);
    size_t end = text.find_last_not_of(whitespace);

    if (start == std::string::npos) {
        return ""; // The string is all whitespace
    }

    return text.substr(start, end - start + 1);
}

int main() {
    // Initialize whisper context
    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* ctx = whisper_init_from_file_with_params("models/ggml-medium.en.bin", cparams);
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

    // Initialize ChatAPI
    ChatAPI chatapi(DS_API_URL);

    // Start the WebSocket server
    auto state = std::make_shared<shared_state>();
    std::thread ws_thread(websocket_server, state, std::ref(chatapi));
    ws_thread.detach();

    // Start the key listener thread
    std::thread key_thread(key_listener, std::ref(chatapi), std::ref(audio), state);
    key_thread.detach();

    // Main loop
    std::vector<float> pcmf32(WHISPER_SAMPLE_RATE * 15, 0.0f); // 15 seconds of audio
    std::string previous_transcription;

    // VAD parameters
    float vad_thold = 0.85f; // Increase to reduce sensitivity
    float freq_thold = 100.0f; // Frequency threshold for VAD
    bool is_running = true;

    while (is_running) {
        is_running = sdl_poll_events();

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
                current_transcription = remove_bracketed_text(current_transcription);

                // Trim leading and trailing whitespace
                current_transcription = lrtrim(current_transcription);

                // Skip if the transcription is empty after cleaning
                if (current_transcription.empty()) {
                    continue;
                }

                // Extract new content by comparing with the previous transcription
                std::string new_content = extract_new_content(previous_transcription, current_transcription);

                // Add the new content to the transcriptions vector if it's not empty
                if (!new_content.empty()) {
                    transcriptions.push_back(new_content);
                    std::cout << new_content << std::endl; // Print the new content
                    previous_transcription = current_transcription; // Update the previous transcription

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

    // Clean up
    audio.pause();
    whisper_free(ctx);
    return 0;
}
