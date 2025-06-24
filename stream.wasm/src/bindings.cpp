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

std::string merge_transcription(const std::string& existing, const std::string& new_text, int max_lookback_words = 8) {
    int best_match_index = -1;

    auto merge_strings = [](std::string_view a, std::string_view b, int& best_match_index, int max_lookback_words) -> std::string {
        if (a.empty()) return std::string(b);
        if (b.empty()) return std::string(a);

        auto split = [](std::string_view str) {
            std::vector<std::string_view> words;
            size_t start = 0;
            while (start < str.size()) {
                size_t end = str.find(' ', start);
                if (end == std::string_view::npos) end = str.size();
                if (end > start) words.push_back(str.substr(start, end - start));
                start = end + 1;
            }
            return words;
        };

        const auto base = split(a);
        const auto tail = split(b);

        const int recent_start = std::max(0, static_cast<int>(base.size()) - max_lookback_words);
        best_match_index = -1;

        // Phase 1: Bigram matching
        if (tail.size() >= 2) {
            for (int i = static_cast<int>(base.size()) - 2; i >= recent_start; --i) {
                if (i + 1 >= static_cast<int>(base.size())) continue;
                if (base[i] == tail[0] && base[i+1] == tail[1]) {
                    best_match_index = i;
                    break;
                }
            }
        }

        // Phase 2: Unigram fallback
        if (best_match_index == -1 && !tail.empty()) {
            for (int i = static_cast<int>(base.size()) - 1; i >= recent_start; --i) {
                if (base[i] == tail[0]) {
                    best_match_index = i;
                    break;
                }
            }
        }

        // Merge logic
        std::string result;
        if (best_match_index >= 0) {
            size_t pos = 0;
            for (int i = 0; i < best_match_index; ++i) {
                pos += base[i].size();
                if (i < best_match_index - 1) {
                    pos += 1;
                }
            }
            result = std::string(a.substr(0, pos));
            if (!b.empty()) {
                if (!result.empty() && result.back() != ' ') {
                    result += ' ';
                }
                result += b;
            }
        } else {
            result = std::string(a);
            if (!result.empty() && !b.empty()) {
                if (result.back() != ' ') result += ' ';
                result += b;
            }
        }

        return result;
    };

    return merge_strings(existing, new_text, best_match_index, max_lookback_words);
}

EMSCRIPTEN_BINDINGS(wstream_module) {
    function("init", &init);
    function("free_instance", &free_instance);
    function("set_audio", &set_audio);
    function("get_transcribed", &get_transcribed);
    function("get_status", &get_status);
    function("set_status", &set_status);
    function("get_confidence_metrics", &get_confidence_metrics);
    function("merge_transcription", &merge_transcription);
}
