#include "audio_recorder.h"
#include "websocket_client.h"
#include "base64.h"
#include <iostream>
#include <signal.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <string>
#include <algorithm>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

#ifdef __APPLE__
bool check_audio_permission() {
    return true;
}
#endif

void show_help() {
    std::cout << "\nAvailable commands:" << std::endl;
    std::cout << "  start            - Start audio recording" << std::endl;
    std::cout << "  stop             - Stop audio recording" << std::endl;
    std::cout << "  status           - Get server status" << std::endl;
    std::cout << "  source <type>    - Set audio source (websocket/microphone)" << std::endl;
    std::cout << "  devices          - List available audio devices" << std::endl;
    std::cout << "  device <name>    - Select audio device by name" << std::endl;
    std::cout << "  verbose <on|off> - Toggle verbose mode" << std::endl;
    std::cout << "  help             - Show this help" << std::endl;
    std::cout << "  quit             - Quit application" << std::endl;
    std::cout << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options] [server_uri] [audio_device]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -v, --verbose      Enable verbose debug output\n";
    std::cout << "  -r, --raw          Use raw PCM data instead of Base64 encoding\n";
    std::cout << "  -l, --list-devices List available audio devices\n";
    std::cout << "  -h, --help         Show this help message\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  server_uri         WebSocket server URI (default: ws://localhost:8080)\n";
    std::cout << "  audio_device       Audio device name (default: system default)\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " --verbose ws://localhost:8080 \"Built-in Microphone\"\n";
}

int main(int argc, char* argv[]) {
    // Install signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Parse command line arguments
    std::string server_uri = "ws://localhost:8080";
    std::string audio_device = "";  // Empty for default
    bool verbose = false;
    bool use_base64 = true;
    bool list_devices = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-r" || arg == "--raw") {
            use_base64 = false;
        } else if (arg == "-l" || arg == "--list-devices") {
            list_devices = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        } else if (server_uri == "ws://localhost:8080") {
            server_uri = arg;
        } else if (audio_device.empty()) {
            audio_device = arg;
        } else {
            std::cerr << "Too many arguments" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Just list devices if requested
    if (list_devices) {
        audio_recorder recorder;
        std::cout << "Available audio devices:" << std::endl;
        auto devices = recorder.list_devices();
        for (size_t i = 0; i < devices.size(); ++i) {
            std::cout << "  " << i << ": " << devices[i] << std::endl;
        }
        return 0;
    }

#ifdef __APPLE__
    check_audio_permission();
#endif

    std::cout << "WStream Interactive Client" << std::endl;
    std::cout << "Server: " << server_uri << std::endl;
    std::cout << "Audio device: " << (audio_device.empty() ? "default" : audio_device) << std::endl;
    std::cout << "Verbose mode: " << (verbose ? "enabled" : "disabled") << std::endl;
    std::cout << "Encoding: " << (use_base64 ? "Base64" : "Raw") << std::endl;
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
    client.set_verbose(verbose);
    client.set_use_base64(use_base64);

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
            std::string action = response["action"].get<std::string>();
            std::string status = response["status"].get<std::string>();

            std::cout << "[RESPONSE] " << action << " -> " << status;

            if (response.contains("message")) {
                std::cout << ": " << response["message"].get<std::string>();
            }

            if (response.contains("current_source")) {
                std::cout << " (current: " << response["current_source"].get<std::string>() << ")";
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
    auto audio_callback = [&client, &verbose](const std::vector<int16_t>& pcm_data) {
        static int packet_count = 0;
        packet_count++;

        if (!client.send_audio_data(pcm_data)) {
            std::cerr << "Failed to send audio data" << std::endl;
        } else if (verbose) {
            if (packet_count % 10 == 0) {  // Less frequent in verbose mode
                std::cout << "[AUDIO] Sent packet #" << packet_count << " (" << pcm_data.size() << " samples)" << std::endl;
                std::cout << "> " << std::flush;  // Show prompt again
            }
        }
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
                // Stop recording with timeout
                std::thread stop_thread([&recorder]() {
                    recorder.stop_recording();
                });

                // Wait up to 3 seconds
                auto start = std::chrono::steady_clock::now();
                while (stop_thread.joinable() &&
                       std::chrono::steady_clock::now() - start < std::chrono::seconds(3)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }

                if (stop_thread.joinable()) {
                    std::cout << "Warning: Stop operation timed out" << std::endl;
                    stop_thread.detach();
                } else {
                    std::cout << "Stopped recording" << std::endl;
                }
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
        } else if (command == "devices") {
            auto devices = recorder.list_devices();
            std::cout << "Available audio devices:" << std::endl;
            for (size_t i = 0; i < devices.size(); ++i) {
                std::cout << "  " << i << ": " << devices[i] << std::endl;
            }
        } else if (command.substr(0, 6) == "device") {
            if (command.length() > 7) {
                std::string device_name = command.substr(7);

                // Stop recording if active
                bool was_recording = recorder.is_recording();
                if (was_recording) {
                    recorder.stop_recording();
                }

                // Try to initialize with new device
                if (recorder.initialize(device_name)) {
                    std::cout << "Switched to audio device: " << device_name << std::endl;

                    // Restart recording if it was active
                    if (was_recording) {
                        if (recorder.start_recording(audio_callback)) {
                            std::cout << "Recording restarted" << std::endl;
                        } else {
                            std::cout << "Failed to restart recording" << std::endl;
                        }
                    }
                } else {
                    std::cout << "Failed to switch audio device" << std::endl;
                }
            } else {
                std::cout << "Usage: device <name>" << std::endl;
            }
        } else if (command.substr(0, 7) == "verbose") {
            if (command.length() > 8) {
                std::string state = command.substr(8);
                std::transform(state.begin(), state.end(), state.begin(), ::tolower);

                if (state == "on" || state == "true" || state == "1") {
                    verbose = true;
                    client.set_verbose(true);
                    std::cout << "Verbose mode enabled" << std::endl;
                } else if (state == "off" || state == "false" || state == "0") {
                    verbose = false;
                    client.set_verbose(false);
                    std::cout << "Verbose mode disabled" << std::endl;
                } else {
                    std::cout << "Usage: verbose <on|off>" << std::endl;
                }
            } else {
                std::cout << "Usage: verbose <on|off>" << std::endl;
            }
        } else if (command == "test audio") {
            // Check if already recording
            bool was_recording = recorder.is_recording();

            // Start recording if not already active
            if (!was_recording) {
                std::cout << "Starting recording for test..." << std::endl;
                if (!recorder.start_recording(audio_callback)) {
                    std::cout << "Failed to start recording for test" << std::endl;
                    continue;
                }
                // Give it a moment to start
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            // Enable dump
            recorder.enable_audio_dump("test_audio.raw");
            std::cout << "Recording 5 seconds of audio to test_audio.raw..." << std::endl;
            std::cout << "Speak now..." << std::endl;

            // Record for 5 seconds
            std::this_thread::sleep_for(std::chrono::seconds(5));

            // Disable dump
            recorder.disable_audio_dump();

            // Stop recording if we started it just for the test
            if (!was_recording) {
                recorder.stop_recording();
                std::cout << "Stopped test recording" << std::endl;
            }

            // Check file size
            std::ifstream test_file("test_audio.raw", std::ios::binary | std::ios::ate);
            if (test_file.is_open()) {
                auto file_size = test_file.tellg();
                test_file.close();
                std::cout << "Test complete. File size: " << file_size << " bytes" << std::endl;

                if (file_size > 0) {
                    std::cout << "You can play the file with:" << std::endl;
                    std::cout << "  ffplay -f s16le -ar 16000 -ac 1 test_audio.raw" << std::endl;
                    std::cout << "Or convert to WAV:" << std::endl;
                    std::cout << "  ffmpeg -f s16le -ar 16000 -ac 1 -i test_audio.raw test_audio.wav" << std::endl;
                } else {
                    std::cout << "WARNING: File is empty. Microphone may not be working or accessible." << std::endl;
                }
            } else {
                std::cout << "ERROR: Could not open test file" << std::endl;
            }
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
