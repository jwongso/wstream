#include "audio_recorder.h"
#include "websocket_client.h"
#include <iostream>
#include <signal.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <string>

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

void show_help() {
    std::cout << "\nAvailable commands:" << std::endl;
    std::cout << "  start    - Start audio recording" << std::endl;
    std::cout << "  stop     - Stop audio recording" << std::endl;
    std::cout << "  status   - Get server status" << std::endl;
    std::cout << "  source <type> - Set audio source (websocket/microphone)" << std::endl;
    std::cout << "  verbose  - Toggle verbose mode" << std::endl;
    std::cout << "  help     - Show this help" << std::endl;
    std::cout << "  quit     - Quit application" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // Install signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Parse command line arguments
    std::string server_uri = "ws://localhost:8080";
    std::string audio_device = "default";

    if (argc > 1) {
        server_uri = argv[1];
    }
    if (argc > 2) {
        audio_device = argv[2];
    }

    std::cout << "WStream Interactive Client" << std::endl;
    std::cout << "Server: " << server_uri << std::endl;
    std::cout << "Audio device: " << audio_device << std::endl;
    std::cout << "Type 'help' for available commands" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Create audio recorder
    audio_recorder recorder;
    if (!recorder.initialize(audio_device)) {
        std::cerr << "Failed to initialize audio recorder" << std::endl;
        return 1;
    }

    // Create WebSocket client
    websocket_client client;

    // Set up transcription callback
    client.set_transcription_callback([](const std::string& text) {
        std::cout << "[TRANSCRIPTION] " << text << std::endl;
        std::cout << "> " << std::flush;  // Show prompt again
    });

    // Set up error callback
    client.set_error_callback([](const std::string& error) {
        std::cerr << "[ERROR] " << error << std::endl;
        std::cout << "> " << std::flush;  // Show prompt again
    });

    // Set up response callback
    client.set_response_callback([](const json& response) {
        if (response.contains("action") && response.contains("status")) {
            std::string action = response["action"];
            std::string status = response["status"];

            std::cout << "[RESPONSE] " << action << " -> " << status;

            if (response.contains("message")) {
                std::cout << ": " << response["message"].get<std::string>();
            }

            if (response.contains("audio_source")) {
                std::cout << " (current: " << response["audio_source"].get<std::string>() << ")";
            }

            std::cout << std::endl;
        }
        std::cout << "> " << std::flush;  // Show prompt again
    });

    // Connect to server
    std::cout << "Connecting to server..." << std::endl;
    if (!client.connect(server_uri)) {
        std::cerr << "Failed to connect to server" << std::endl;
        return 1;
    }

    // Set up audio callback
    auto audio_callback = [&client](const std::vector<int16_t>& pcm_data) {
        client.send_audio_data(pcm_data);
    };

    // Automatically set audio source to websocket
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    client.set_audio_source("websocket");

    // Interactive command loop
    std::string command;
    std::cout << "\n> " << std::flush;

    while (g_running && std::getline(std::cin, command)) {
        if (command.empty()) {
            std::cout << "> " << std::flush;
            continue;
        }

        if (command == "quit" || command == "exit") {
            break;
        } else if (command == "help") {
            show_help();
        } else if (command == "start") {
            if (recorder.is_recording()) {
                std::cout << "Already recording" << std::endl;
            } else {
                if (recorder.start_recording(audio_callback)) {
                    std::cout << "Started recording" << std::endl;
                } else {
                    std::cout << "Failed to start recording" << std::endl;
                }
            }
        } else if (command == "stop") {
            if (!recorder.is_recording()) {
                std::cout << "Not recording" << std::endl;
            } else {
                recorder.stop_recording();
                std::cout << "Stopped recording" << std::endl;
            }
        } else if (command == "status") {
            client.get_server_status();
        } else if (command.substr(0, 6) == "source") {
            if (command.length() > 7) {
                std::string source_type = command.substr(7);
                client.set_audio_source(source_type);
            } else {
                std::cout << "Usage: source <websocket|microphone>" << std::endl;
            }
        } else if (command == "verbose") {
            client.set_verbose();
        } else {
            std::cout << "Unknown command: " << command << std::endl;
            std::cout << "Type 'help' for available commands" << std::endl;
        }

        std::cout << "> " << std::flush;
    }

    // Cleanup
    if (recorder.is_recording()) {
        recorder.stop_recording();
    }

    client.disconnect();

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
