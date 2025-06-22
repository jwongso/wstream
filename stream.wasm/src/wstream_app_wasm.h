#pragma once

#include "whisper.h"
#include "common.h"
#include "common-whisper.h"
#include <memory>
#include <atomic>
#include <string>
#include <vector>
#include <functional>

// Forward declarations
class whisper_engine;
class text_processor;

class wstream_app_wasm {
public:
    using TranscriptionCallback = std::function<void(const std::string&)>;

    static constexpr const char* DEFAULT_MODEL_PATH = "models/ggml-small.en-q5_1.bin";

    explicit wstream_app_wasm(const std::string& model_path = DEFAULT_MODEL_PATH);
    ~wstream_app_wasm();

    // Non-copyable, non-movable
    wstream_app_wasm(const wstream_app_wasm&) = delete;
    wstream_app_wasm& operator=(const wstream_app_wasm&) = delete;
    wstream_app_wasm(wstream_app_wasm&&) = delete;
    wstream_app_wasm& operator=(wstream_app_wasm&&) = delete;

    bool initialize(const std::string& model_path);
    void process_audio_buffer(const std::vector<float>& audio_data);
    void set_transcription_callback(TranscriptionCallback callback);
    void start();
    void stop();
    bool is_running() const;

private:
    std::atomic<bool> m_is_running{false};
    std::unique_ptr<whisper_engine> m_whisper_engine;
    std::unique_ptr<text_processor> m_text_processor;
    std::string m_model_path;
    TranscriptionCallback m_transcription_callback;

    std::string extract_new_transcription(const std::string& full_text);
};
