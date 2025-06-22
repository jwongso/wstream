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

// Free an instance - renamed to avoid conflict with stdlib free()
void free_instance(size_t handle) {
    auto it = g_apps.find(handle);
    if (it != g_apps.end()) {
        it->second->stop();
        g_apps.erase(it);
        std::cout << "[Bindings] Freed instance with handle: " << handle << std::endl;
    }
}

// Set audio data - matching the original Whisper.cpp pattern
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

    // Create a vector to hold the audio data
    std::vector<float> audio_data;
    audio_data.resize(n);

    // Copy data from JavaScript Float32Array to C++ vector
    // This approach matches the original Whisper.cpp implementation
    val heap = val::module_property("HEAPU8");
    val memory = heap["buffer"];

    // Create a view of the C++ vector in JavaScript memory space
    val memoryView = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(audio_data.data()), n);

    // Copy the audio data
    memoryView.call<void>("set", audio);

    // Push to processing queue
    it->second->push_audio(audio_data);

    return 0;
}

// Alternative simpler approach if the above doesn't work
int set_audio_simple(size_t handle, const val& audio) {
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

EMSCRIPTEN_BINDINGS(wstream_module) {
    function("init", &init);
    function("free_instance", &free_instance);
    function("set_audio", &set_audio_simple);
    function("get_transcribed", &get_transcribed);
    function("get_status", &get_status);
    function("set_status", &set_status);
}
