#pragma once

#include "whisper.h"
#include <memory>
#include <atomic>
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>

// Forward declarations
class whisper_engine;
class text_processor;

class wstream_app_wasm {
public:
    using TranscriptionCallback = std::function<void(const std::string&)>;

    static constexpr const char* DEFAULT_MODEL_PATH = "models/ggml-tiny.en-q5_1.bin";
    static constexpr int AUDIO_WINDOW_SAMPLES = 5 * 16000; // 5 seconds

    explicit wstream_app_wasm(const std::string& model_path = DEFAULT_MODEL_PATH);
    ~wstream_app_wasm();

    // Non-copyable, non-movable
    wstream_app_wasm(const wstream_app_wasm&) = delete;
    wstream_app_wasm& operator=(const wstream_app_wasm&) = delete;
    wstream_app_wasm(wstream_app_wasm&&) = delete;
    wstream_app_wasm& operator=(wstream_app_wasm&&) = delete;

    bool initialize(const std::string& model_path);
    void push_audio(const std::vector<float>& audio_data);
    void set_transcription_callback(TranscriptionCallback callback);
    void start();
    void stop();
    bool is_running() const;

    // Get latest transcription (for polling from JS)
    std::string get_transcribed();
    std::string get_status() const;
    void set_status(const std::string& status);

private:
    // Thread management
    std::atomic<bool> m_is_running{false};
    std::thread m_worker_thread;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;

    // Audio buffer (protected by mutex)
    std::deque<float> m_audio_buffer;

    // Status and transcription (protected by mutex)
    std::string m_status{"not started"};
    std::string m_status_forced;
    std::string m_latest_transcription;

    // Processing components
    std::unique_ptr<whisper_engine> m_whisper_engine;
    std::unique_ptr<text_processor> m_text_processor;
    std::string m_model_path;
    TranscriptionCallback m_transcription_callback;

    // Worker thread function
    void worker_main();
    void set_status_internal(const std::string& status);
};
