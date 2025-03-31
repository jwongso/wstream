#include "ggml.h"
#include "whisper.h"
#include "hyni_merge.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <atomic>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Constants
constexpr int N_THREAD = 8;
constexpr size_t MAX_CONTEXTS = 4;
constexpr size_t MIN_AUDIO_SAMPLES = 1024;
constexpr int64_t WINDOW_SAMPLES = 5*WHISPER_SAMPLE_RATE;
constexpr int STATUS_UPDATE_INTERVAL_MS = 100;

// Global state
struct WhisperState {
    std::vector<whisper_context*> contexts;
    std::mutex mutex;
    std::thread worker;
    std::atomic<bool> running{false};
    std::string status;
    std::string status_forced;
    std::string transcribed;
    std::vector<float> pcmf32;
    std::string merged_transcription;

    WhisperState() : contexts(MAX_CONTEXTS, nullptr) {}
};

static WhisperState g_state;

// Helper function to initialize whisper parameters
whisper_full_params get_whisper_params() {
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    wparams.n_threads = std::min(N_THREAD, (int)std::thread::hardware_concurrency());
    wparams.offset_ms = 0;
    wparams.translate = false;
    wparams.no_context = true;
    wparams.single_segment = true;
    wparams.print_realtime = false;
    wparams.print_progress = false;
    wparams.print_timestamps = true;
    wparams.print_special = false;
    wparams.max_tokens = 32;
    wparams.audio_ctx = 768; // Partial encoder context for better performance
    wparams.temperature_inc = -1.0f; // Disable temperature fallback
    wparams.language = "en";
    wparams.suppress_blank = true;
    wparams.suppress_nst = true;
    wparams.no_timestamps = true;
    wparams.greedy.best_of = 1;

    return wparams;
}

void stream_set_status(const std::string& status) {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    g_state.status = status;
}

void stream_main(size_t index) {
    auto wparams = get_whisper_params();
    printf("stream: using %d threads\n", wparams.n_threads);

    auto& ctx = g_state.contexts[index];
    std::vector<float> pcmf32_window;
    pcmf32_window.reserve(WINDOW_SAMPLES);

    auto last_status_update = std::chrono::steady_clock::now();

    while (g_state.running) {
        // Throttle status updates
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_status_update).count() > STATUS_UPDATE_INTERVAL_MS) {
            stream_set_status("waiting for audio ...");
            last_status_update = now;
        }

        {
            std::unique_lock<std::mutex> lock(g_state.mutex);
            if (g_state.pcmf32.size() < MIN_AUDIO_SAMPLES) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Efficient window extraction
            size_t copy_size = std::min((int64_t)g_state.pcmf32.size(), WINDOW_SAMPLES);
            pcmf32_window.assign(g_state.pcmf32.end() - copy_size, g_state.pcmf32.end());
            g_state.pcmf32.erase(g_state.pcmf32.end() - copy_size, g_state.pcmf32.end());
        }

        {
            stream_set_status("running whisper ...");

            int ret = whisper_full(ctx, wparams, pcmf32_window.data(), pcmf32_window.size());
            if (ret != 0) {
                printf("whisper_full() failed: %d\n", ret);
                break;
            }
        }

        {
            std::string current_segment;
            const int n_segments = whisper_full_n_segments(ctx);
            if (n_segments > 0) {
                current_segment = whisper_full_get_segment_text(ctx, n_segments - 1);

                std::lock_guard<std::mutex> lock(g_state.mutex);
                // Merge with previous transcription
                g_state.merged_transcription = hyni::HyniMerge::mergeStrings(
                    g_state.merged_transcription,
                    current_segment
                );

                printf("%s\n", g_state.merged_transcription.c_str());
            }
        }
    }

    if (index < g_state.contexts.size() && g_state.contexts[index]) {
        whisper_free(g_state.contexts[index]);
        g_state.contexts[index] = nullptr;
    }
}

EMSCRIPTEN_BINDINGS(stream) {
    emscripten::function("init", emscripten::optional_override([](const std::string& path_model) {
        for (size_t i = 0; i < g_state.contexts.size(); ++i) {
            if (!g_state.contexts[i]) {
                g_state.contexts[i] = whisper_init_from_file_with_params(path_model.c_str(), whisper_context_default_params());
                if (g_state.contexts[i]) {
                    g_state.running = true;
                    if (g_state.worker.joinable()) {
                        g_state.worker.join();
                    }
                    g_state.worker = std::thread([i]() { stream_main(i); });
                    return i + 1;
                }
                return (size_t)0;
            }
        }
        return (size_t)0;
    }));

    emscripten::function("free", emscripten::optional_override([](size_t index) {
        g_state.running = false;
    }));

    emscripten::function("set_audio", emscripten::optional_override([](size_t index, const emscripten::val& audio) {
        --index;
        if (index >= g_state.contexts.size() || !g_state.contexts[index]) {
            return -1;
        }

        std::lock_guard<std::mutex> lock(g_state.mutex);
        const int n = audio["length"].as<int>();
        g_state.pcmf32.resize(n);

        emscripten::val heap = emscripten::val::module_property("HEAPU8");
        emscripten::val memoryView = audio["constructor"].new_(
            heap["buffer"],
            reinterpret_cast<uintptr_t>(g_state.pcmf32.data()),
                                                               n
        );
        memoryView.call<void>("set", audio);
        return 0;
    }));

    emscripten::function("get_transcribed", emscripten::optional_override([]() {
        std::lock_guard<std::mutex> lock(g_state.mutex);
        return g_state.merged_transcription;
    }));

    emscripten::function("get_status", emscripten::optional_override([]() {
        std::lock_guard<std::mutex> lock(g_state.mutex);
        return g_state.status_forced.empty() ? g_state.status : g_state.status_forced;
    }));

    emscripten::function("set_status", emscripten::optional_override([](const std::string& status) {
        std::lock_guard<std::mutex> lock(g_state.mutex);
        g_state.status_forced = status;
    }));

    emscripten::function("reset_transcription", emscripten::optional_override([]() {
        std::lock_guard<std::mutex> lock(g_state.mutex);
        g_state.merged_transcription.clear();
        g_state.transcribed.clear();
    }));
}
