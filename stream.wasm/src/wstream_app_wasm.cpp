#include "wstream_app_wasm.h"
#include "whisper_engine.h"
#include "text_processor.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <emscripten.h>

class simple_vad {
private:
    static constexpr float ENERGY_THRESHOLD = 0.01f;
    static constexpr float FREQ_THRESHOLD = 0.02f;
    static constexpr int MIN_VOICE_FRAMES = 800; // 50ms at 16kHz

public:
    static bool has_voice_activity(const std::vector<float>& audio) {
        if (audio.size() < MIN_VOICE_FRAMES) {
            return false;
        }

        // Calculate RMS energy
        float energy = 0.0f;
        float max_amplitude = 0.0f;

        for (const auto& sample : audio) {
            energy += sample * sample;
            max_amplitude = std::max(max_amplitude, std::abs(sample));
        }

        energy = std::sqrt(energy / audio.size());

        // Check if audio has sufficient energy and peak amplitude
        return energy > ENERGY_THRESHOLD && max_amplitude > FREQ_THRESHOLD;
    }

    // Find voice segments in audio
    static std::pair<size_t, size_t> find_voice_bounds(const std::vector<float>& audio) {
        const size_t window_size = 1600; // 100ms windows
        size_t start = 0;
        size_t end = audio.size();

        // Find start of voice
        for (size_t i = 0; i < audio.size() - window_size; i += window_size/2) {
            std::vector<float> window(audio.begin() + i, audio.begin() + i + window_size);
            if (has_voice_activity(window)) {
                start = i;
                break;
            }
        }

        // Find end of voice
        for (size_t i = audio.size(); i > window_size; i -= window_size/2) {
            std::vector<float> window(audio.begin() + i - window_size, audio.begin() + i);
            if (has_voice_activity(window)) {
                end = i;
                break;
            }
        }

        return {start, end};
    }
};

wstream_app_wasm::wstream_app_wasm(const std::string& model_path)
    : m_model_path(model_path) {
}

wstream_app_wasm::~wstream_app_wasm() {
    stop();
}

bool wstream_app_wasm::initialize(const std::string& model_path) {
    set_status_internal("loading model...");

    m_whisper_engine = std::make_unique<whisper_engine>(model_path);

    if (!m_whisper_engine->initialize(true)) {
        std::cerr << "[WASM] Failed to initialize Whisper engine" << std::endl;
        set_status_internal("error: failed to load model");
        return false;
    }

    m_text_processor = std::make_unique<text_processor>();
    set_status_internal("ready");
    return true;
}

void wstream_app_wasm::start() {
    if (m_is_running.exchange(true)) {
        return; // Already running
    }

    // Start worker thread
    m_worker_thread = std::thread([this]() {
        worker_main();
    });

    set_status_internal("started");
}

void wstream_app_wasm::stop() {
    if (!m_is_running.exchange(false)) {
        return; // Already stopped
    }

    // Wake up worker thread
    m_cv.notify_all();

    // Wait for thread to finish
    if (m_worker_thread.joinable()) {
        m_worker_thread.join();
    }

    set_status_internal("stopped");
}

void wstream_app_wasm::push_audio(const std::vector<float>& audio_data) {
    if (!m_is_running || audio_data.empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Append new audio data
        m_audio_buffer.insert(m_audio_buffer.end(), audio_data.begin(), audio_data.end());

        // Keep buffer size reasonable (e.g., 2 minutes max)
        const size_t max_samples = 120 * 16000; // 2 minutes
        if (m_audio_buffer.size() > max_samples) {
            size_t to_remove = m_audio_buffer.size() - max_samples;
            m_audio_buffer.erase(m_audio_buffer.begin(), m_audio_buffer.begin() + to_remove);
        }
    }

    // Notify worker thread
    m_cv.notify_one();
}

void wstream_app_wasm::worker_main() {
    std::cout << "[Worker] Thread started" << std::endl;

    std::vector<float> processing_buffer;
    processing_buffer.reserve(AUDIO_WINDOW_SAMPLES);

    while (m_is_running) {
        {
            std::unique_lock<std::mutex> lock(m_mutex);

            // Wait for audio data
            m_cv.wait(lock, [this] {
                return !m_is_running || m_audio_buffer.size() >= 16000; // At least 1 second
            });

            if (!m_is_running) {
                break;
            }

            // Take the last window_samples for processing
            size_t samples_to_take = std::min(
                static_cast<size_t>(AUDIO_WINDOW_SAMPLES),
                m_audio_buffer.size()
            );

            if (samples_to_take < 16000) { // Less than 1 second
                continue;
            }

            // Copy audio data for processing
            processing_buffer.clear();
            processing_buffer.assign(
                m_audio_buffer.end() - samples_to_take,
                m_audio_buffer.end()
            );

            // Clear old audio to prevent unbounded growth
            m_audio_buffer.clear();
        }

        // Skip if no voice activity
        if (!simple_vad::has_voice_activity(processing_buffer)) {
            set_status_internal("waiting for speech...");
            continue;
        }

        // Find voice boundaries to process only speech
        auto [start, end] = simple_vad::find_voice_bounds(processing_buffer);
        if (end - start < 8000) { // Less than 0.5 seconds
            continue;
        }

        // Process only the part with voice
        std::vector<float> voice_segment(
            processing_buffer.begin() + start,
            processing_buffer.begin() + end
        );

        // Process audio outside of lock
        set_status_internal("processing...");

        auto t_start = std::chrono::high_resolution_clock::now();

        // Transcribe
        std::string transcription = m_whisper_engine->transcribe(voice_segment);

        auto t_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "[Worker] Processed " << processing_buffer.size()
                  << " samples in " << duration << " seconds" << std::endl;

        if (!transcription.empty()) {
            // Process through text processor
            std::string processed_text = m_text_processor->process(transcription);

            if (!processed_text.empty()) {
                // Store transcription
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_latest_transcription = processed_text;
                }

                // Call callback if set
                if (m_transcription_callback) {
                    m_transcription_callback(processed_text);
                }
            }
        }

        set_status_internal("waiting for audio...");
    }

    std::cout << "[Worker] Thread stopped" << std::endl;
}

std::string wstream_app_wasm::get_transcribed() {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string result = std::move(m_latest_transcription);
    m_latest_transcription.clear();
    return result;
}

void wstream_app_wasm::set_status_internal(const std::string& status) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_status = status;
}

std::string wstream_app_wasm::get_status() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_status_forced.empty() ? m_status : m_status_forced;
}

void wstream_app_wasm::set_status(const std::string& status) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_status_forced = status;
}

void wstream_app_wasm::set_transcription_callback(TranscriptionCallback callback) {
    m_transcription_callback = callback;
}

bool wstream_app_wasm::is_running() const {
    return m_is_running;
}
