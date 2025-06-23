#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "wstream_app_wasm.h"
#include <vector>
#include <memory>
#include <iostream>
#include <map>

using namespace emscripten;

// Global app instances (support multiple contexts like original)
std::map<size_t, std::unique_ptr<wstream_app_wasm>> g_apps;
size_t g_next_handle = 1;

// Initialize the app
size_t init(const std::string& model_path) {
    auto app = std::make_unique<wstream_app_wasm>(model_path);

    if (!app->initialize(model_path)) {
        std::cerr << "[ERROR] Failed to initialize Whisper" << std::endl;
        return 0;
    }

    app->start();

    size_t handle = g_next_handle++;
    g_apps[handle] = std::move(app);

    std::cout << "[Bindings] Created instance with handle: " << handle << std::endl;
    return handle;
}

// Free an instance
void free_instance(size_t handle) {
    auto it = g_apps.find(handle);
    if (it != g_apps.end()) {
        it->second->stop();
        g_apps.erase(it);
        std::cout << "[Bindings] Freed instance with handle: " << handle << std::endl;
    }
}

// Set audio data
int set_audio(size_t handle, const val& audio) {
    auto it = g_apps.find(handle);
    if (it == g_apps.end()) {
        return -1;
    }

    // Get the length of the audio array
    const int n = audio["length"].as<int>();
    if (n == 0) {
        return -2;
    }

    // Create a vector and copy element by element
    std::vector<float> audio_data;
    audio_data.reserve(n);

    for (int i = 0; i < n; ++i) {
        audio_data.push_back(audio[i].as<float>());
    }

    // Push to processing queue
    it->second->push_audio(audio_data);

    return 0;
}

// Get transcribed text
std::string get_transcribed(size_t handle) {
    auto it = g_apps.find(handle);
    if (it == g_apps.end()) {
        return "";
    }

    return it->second->get_transcribed();
}

// Get status
std::string get_status(size_t handle) {
    auto it = g_apps.find(handle);
    if (it == g_apps.end()) {
        return "invalid handle";
    }

    return it->second->get_status();
}

// Set status
void set_status(size_t handle, const std::string& status) {
    auto it = g_apps.find(handle);
    if (it != g_apps.end()) {
        it->second->set_status(status);
    }
}

// Get confidence metrics
std::string get_confidence_metrics(size_t handle) {
    auto it = g_apps.find(handle);
    if (it == g_apps.end()) {
        return "{\"error\":\"invalid handle\"}";
    }

    return it->second->get_confidence_metrics();
}

EMSCRIPTEN_BINDINGS(wstream_module) {
    function("init", &init);
    function("free_instance", &free_instance);
    function("set_audio", &set_audio);
    function("get_transcribed", &get_transcribed);
    function("get_status", &get_status);
    function("set_status", &set_status);
    function("get_confidence_metrics", &get_confidence_metrics);
}
