#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "wstream_app_wasm.h"
#include <vector>
#include <memory>
#include <iostream>
#include <string>

using namespace emscripten;

// Global app instance
std::unique_ptr<wstream_app_wasm> g_app;

// Global transcription buffer
std::string g_transcribed;
std::string g_status = "not started";

// Initialize the app (returns handle like original)
int init(const std::string& model_path) {
    g_app = std::make_unique<wstream_app_wasm>(model_path);
    if (!g_app->initialize(model_path)) {
        std::cerr << "[ERROR] Failed to initialize Whisper" << std::endl;
        return 0;
    }

    // Set callback to capture transcriptions
    g_app->set_transcription_callback([](const std::string& text) {
        g_transcribed = text;
    });

    g_app->start();
    g_status = "ready";

    return 1; // Return non-zero handle
}

// Set audio data (following original pattern exactly)
void set_audio(int handle, const val& audio_array) {
    if (handle == 0 || !g_app) {
        return;
    }

    // Convert JavaScript Float32Array to C++ vector
    unsigned int length = audio_array["length"].as<unsigned int>();

    if (length == 0) {
        return;
    }

    std::vector<float> audio_data;
    audio_data.reserve(length);

    for (unsigned int i = 0; i < length; ++i) {
        audio_data.push_back(audio_array[i].as<float>());
    }

    // Process the accumulated audio
    g_app->process_audio_buffer(audio_data);
}

// Get transcribed text (returns latest transcription)
std::string get_transcribed() {
    if (g_transcribed.empty()) {
        return "";
    }

    // Return and clear
    std::string result = g_transcribed;
    g_transcribed.clear();
    return result;
}

// Get status
std::string get_status() {
    return g_status;
}

// Set status
void set_status(const std::string& status) {
    g_status = status;
}

EMSCRIPTEN_BINDINGS(wstream_module) {
    function("init", &init);
    function("set_audio", &set_audio);
    function("get_transcribed", &get_transcribed);
    function("get_status", &get_status);
    function("set_status", &set_status);
}
