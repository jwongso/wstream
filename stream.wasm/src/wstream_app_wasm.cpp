#include "wstream_app_wasm.h"
#include "whisper_engine.h"
#include "text_processor.h"
#include <iostream>
#include <algorithm>
#include <emscripten.h>

wstream_app_wasm::wstream_app_wasm(const std::string& model_path)
    : m_model_path(model_path) {
}

wstream_app_wasm::~wstream_app_wasm() {
    stop();
}

bool wstream_app_wasm::initialize(const std::string& model_path) {
    m_whisper_engine = std::make_unique<whisper_engine>(model_path);

    if (!m_whisper_engine->initialize(true)) {
        std::cerr << "[WASM] Failed to initialize Whisper engine" << std::endl;
        return false;
    }

    m_text_processor = std::make_unique<text_processor>();
    return true;
}

void wstream_app_wasm::process_audio_buffer(const std::vector<float>& audio_data) {
    if (!m_is_running || audio_data.empty()) {
        return;
    }

    // Store the last processed position
    static size_t s_last_processed_size = 0;

    // Only process if we have new audio
    if (audio_data.size() <= s_last_processed_size) {
        return;
    }

    // Transcribe the entire audio
    std::string full_transcription = m_whisper_engine->transcribe(audio_data);

    if (!full_transcription.empty()) {
        // Extract only the new part of transcription
        std::string new_text = extract_new_transcription(full_transcription);

        if (!new_text.empty()) {
            // Process through text processor
            std::string processed_text = m_text_processor->process(new_text);

            if (!processed_text.empty() && m_transcription_callback) {
                m_transcription_callback(processed_text);
            }
        }
    }

    // Update last processed size
    s_last_processed_size = audio_data.size();
}

std::string wstream_app_wasm::extract_new_transcription(const std::string& full_text) {
    // Keep track of last transcription to extract only new parts
    static std::string s_last_transcription;

    std::string new_text;

    if (full_text.length() > s_last_transcription.length()) {
        // Check if the full text starts with the last transcription
        if (full_text.substr(0, s_last_transcription.length()) == s_last_transcription) {
            // Extract only the new part
            new_text = full_text.substr(s_last_transcription.length());
        } else {
            // Full text has changed completely
            new_text = full_text;
        }
    }

    // Update last transcription
    s_last_transcription = full_text;

    // Trim whitespace
    if (!new_text.empty()) {
        new_text.erase(0, new_text.find_first_not_of(" \t\n\r"));
        new_text.erase(new_text.find_last_not_of(" \t\n\r") + 1);
    }

    return new_text;
}

void wstream_app_wasm::set_transcription_callback(TranscriptionCallback callback) {
    m_transcription_callback = callback;
}

void wstream_app_wasm::start() {
    m_is_running = true;
}

void wstream_app_wasm::stop() {
    m_is_running = false;
}

bool wstream_app_wasm::is_running() const {
    return m_is_running;
}
