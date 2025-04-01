#include <vector>
#include <thread>
#include <emscripten.h>
#include <emscripten/bind.h>
#include "whisper.h"
#include "common.h"

static const int G_NUM_THREAD = 8;
static whisper_context* g_ctx;
static whisper_full_params g_params;
static std::string g_status = "";
static std::string g_transcription = "";
std::atomic<bool> is_worker_running(false);
std::thread audioProcessingThread;
std::shared_ptr<std::vector<float>> sharedAudioBuffer;
std::mutex transcription_mutex;

void process_audio_thread() {
    while (true) {  // Infinite loop to continuously process audio
        if (sharedAudioBuffer) {
            std::vector<float>& audioData = *sharedAudioBuffer;

            if (audioData.empty()) {
                // Sleep for a short time to avoid busy-waiting when there's no new audio
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;  // No new audio, just continue looping
            }

            // Parameters for the whisper_full function
            int start_sample = 0;
            int end_sample = audioData.size();

            // Call the whisper_full function (assuming g_ctx and g_params are already set up)
            if (whisper_full(g_ctx, g_params, audioData.data() + start_sample, end_sample - start_sample) != 0) {
                fprintf(stderr, "Whisper processing failed!\n");
                continue;  // If whisper processing fails, continue to the next iteration
            }

            std::lock_guard<std::mutex> lock(transcription_mutex);

            // Retrieve and print the transcription segments
            int num_segments = whisper_full_n_segments(g_ctx);
            for (int i = 0; i < num_segments; ++i) {
                g_transcription = whisper_full_get_segment_text(g_ctx, i);
                printf("%s\n", g_transcription.c_str());  // Print each segment with its index
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Sleep to avoid busy-waiting
        }
    }
}

EMSCRIPTEN_BINDINGS(stream) {
    emscripten::register_vector<float>("vector<float>");
    emscripten::function("init", emscripten::optional_override([](const std::string& path_model) {
        if (g_ctx) {
            whisper_free(g_ctx);
        }
        g_ctx = whisper_init_from_file_with_params(path_model.c_str(), whisper_context_default_params());
        if (!g_ctx) {
            fprintf(stderr, "Failed to initialize Whisper!");
            return false;
        }

        g_params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        g_params.print_realtime   = false;
        g_params.print_progress   = false;
        g_params.print_timestamps = false;
        g_params.print_special    = false;
        g_params.translate        = false;
        g_params.single_segment   = true;
        g_params.no_context       = true;
        g_params.max_tokens       = 0;
        g_params.audio_ctx        = 0;
        g_params.temperature      = 0.1f;
        g_params.temperature_inc  = 0.2f;
        g_params.greedy.best_of   = 1;
        g_params.prompt_n_tokens  = 0;
        g_params.language         = "en";
        g_params.n_threads        = std::min(G_NUM_THREAD, (int)std::thread::hardware_concurrency());
        g_params.offset_ms        = 0;
        g_params.duration_ms      = 0;

        return true;
    }));

    emscripten::function("set_audio", emscripten::optional_override([](size_t /*index*/, const emscripten::val& audio) {
        if (!g_ctx) {
            fprintf(stderr, "Whisper not initialized!\n");
            return;
        }

        const int n = audio["length"].as<int>();
        fprintf(stderr, "Audio length: %d\n", n);

        std::lock_guard<std::mutex> lock(transcription_mutex);
        if (!sharedAudioBuffer) {
            sharedAudioBuffer = std::make_shared<std::vector<float>>(n);
        } else {
            sharedAudioBuffer->resize(n);
        }

        // Copy data from audio to the shared buffer
        emscripten::val heap = emscripten::val::module_property("HEAPU8");
        emscripten::val memoryView = audio["constructor"].new_(
            heap["buffer"], reinterpret_cast<uintptr_t>(sharedAudioBuffer->data()), n);
        memoryView.call<void>("set", audio);

        if (!is_worker_running.load()) {
            is_worker_running.store(true);
            std::thread worker_thread([]() {
                process_audio_thread();
                is_worker_running.store(false);
            });
            worker_thread.detach();
        }
    }));

    emscripten::function("get_transcribed", emscripten::optional_override([]() {
        return g_transcription;
    }));

    emscripten::function("cleanup", emscripten::optional_override([]() {
        if (g_ctx) {
            whisper_free(g_ctx);
            g_ctx = nullptr;
        }
    }));

    emscripten::function("get_status", emscripten::optional_override([]() {
        return g_status;
    }));

    emscripten::function("set_status", emscripten::optional_override([](const std::string& status) {
        g_status = status;
    }));

    emscripten::function("reset_transcription", emscripten::optional_override([]() {
        g_transcription.clear();
        g_status.clear();
    }));
}
