#include "wstream_app_wasm.h"
#include "whisper_engine.h"
#include "text_processor.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <emscripten.h>

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

    // Smart buffering parameters
    const size_t MIN_PROCESS_SAMPLES = 16000;   // 1 second minimum
    const size_t IDEAL_PROCESS_SAMPLES = 48000; // 3 seconds ideal
    const size_t MAX_PROCESS_SAMPLES = 80000;   // 5 seconds maximum
    const size_t OVERLAP_SAMPLES = 4800;        // 300ms overlap for context

    auto last_process_time = std::chrono::steady_clock::now();
    const auto MIN_PROCESS_INTERVAL = std::chrono::milliseconds(1500); // Don't process too frequently

    while (m_is_running) {
        size_t buffer_size = 0;

        {
            std::unique_lock<std::mutex> lock(m_mutex);

            // Wait for audio with timeout
            m_cv.wait_for(lock, std::chrono::milliseconds(500), [this, &buffer_size] {
                buffer_size = m_audio_buffer.size();
                return !m_is_running || buffer_size >= MIN_PROCESS_SAMPLES;
            });

            if (!m_is_running) {
                break;
            }

            buffer_size = m_audio_buffer.size();

            // Determine if we should process
            auto now = std::chrono::steady_clock::now();
            auto time_since_last = now - last_process_time;

            bool should_process = false;
            size_t samples_to_process = 0;

            if (buffer_size >= MAX_PROCESS_SAMPLES) {
                // Buffer is getting full, must process
                should_process = true;
                samples_to_process = MAX_PROCESS_SAMPLES;
            } else if (buffer_size >= IDEAL_PROCESS_SAMPLES &&
                      time_since_last >= MIN_PROCESS_INTERVAL) {
                // Ideal amount of audio and enough time has passed
                should_process = true;
                samples_to_process = IDEAL_PROCESS_SAMPLES;
            } else if (buffer_size >= MIN_PROCESS_SAMPLES &&
                      time_since_last >= std::chrono::seconds(3)) {
                // Been waiting too long, process what we have
                should_process = true;
                samples_to_process = buffer_size;
            }

            if (!should_process) {
                continue;
            }

            // Extract audio for processing
            samples_to_process = std::min(samples_to_process, buffer_size);
            processing_buffer.assign(
                m_audio_buffer.end() - samples_to_process,
                m_audio_buffer.end()
            );

            // Smart buffer management: keep some overlap for context
            if (buffer_size > samples_to_process && samples_to_process > OVERLAP_SAMPLES) {
                // Keep the overlap at the beginning
                m_audio_buffer.erase(
                    m_audio_buffer.begin(),
                    m_audio_buffer.end() - OVERLAP_SAMPLES
                );
            } else {
                // Not enough for overlap, clear everything
                m_audio_buffer.clear();
            }

            last_process_time = now;
        }

        // Process audio outside of lock
        set_status_internal("processing...");

        auto t_start = std::chrono::high_resolution_clock::now();

        // Transcribe the audio
        //std::string transcription = m_whisper_engine->transcribe(processing_buffer);
        auto result = m_whisper_engine->transcribe_with_confidence(processing_buffer);

        if (!result.text.empty() && result.n_tokens > 0) {
            float confidence = calculate_confidence_score(result.avg_logprob, result.entropy);

            // Skip very low confidence results
            if (confidence < 30.0f) {
                std::cout << "[Worker] Skipping low confidence result: " << confidence << "%" << std::endl;
                continue;
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double>(t_end - t_start).count();

            // Calculate real-time factor
            double audio_duration = processing_buffer.size() / 16000.0;
            double rtf = duration / audio_duration;

            // Update stats
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_last_metrics.confidence = confidence;
                m_last_metrics.avg_logprob = result.avg_logprob;
                m_last_metrics.entropy = result.entropy;
                m_last_metrics.n_tokens = result.n_tokens;
                m_last_metrics.rtf = rtf;
                m_last_metrics.audio_duration = audio_duration;
            }

            // Process text
            std::string processed_text = m_text_processor->process(result.text);

            if (!processed_text.empty()) {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_latest_transcription = processed_text;

                // Call callback if set
                if (m_transcription_callback) {
                    m_transcription_callback(processed_text);
                }
            }

            set_status_internal("waiting for audio...");
        }
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

float wstream_app_wasm::calculate_confidence_score(float avg_logprob, float entropy) {
    std::cout << "[Confidence] Input - avg_logprob: " << avg_logprob
              << ", entropy: " << entropy << std::endl;

    float confidence = 0.0f;

    // Since we're getting positive values (0.17 to 0.69),
    // let's treat them as quality scores from 0 to 1
    if (avg_logprob > 0) {
        // Map positive values directly to confidence
        // Values seem to range from 0.17 to 0.69
        // Let's map: 0.0-0.2 = poor, 0.2-0.4 = fair, 0.4-0.6 = good, 0.6+ = excellent

        if (avg_logprob >= 0.6) {
            confidence = 85.0f + (avg_logprob - 0.6f) * 37.5f; // 85-100%
        } else if (avg_logprob >= 0.4) {
            confidence = 70.0f + ((avg_logprob - 0.4f) / 0.2f) * 15.0f; // 70-85%
        } else if (avg_logprob >= 0.2) {
            confidence = 50.0f + ((avg_logprob - 0.2f) / 0.2f) * 20.0f; // 50-70%
        } else {
            confidence = avg_logprob * 250.0f; // 0-50%
        }
    } else {
        // Proper negative log probabilities
        if (avg_logprob >= -0.1f) {
            confidence = 95.0f;
        } else if (avg_logprob >= -0.5f) {
            confidence = 80.0f + ((avg_logprob + 0.5f) / 0.4f) * 15.0f;
        } else if (avg_logprob >= -1.0f) {
            confidence = 60.0f + ((avg_logprob + 1.0f) / 0.5f) * 20.0f;
        } else if (avg_logprob >= -2.0f) {
            confidence = 30.0f + ((avg_logprob + 2.0f) / 1.0f) * 30.0f;
        } else {
            confidence = std::max(0.0f, 30.0f + (avg_logprob + 2.0f) * 10.0f);
        }
    }

    // Clamp to 0-100 range
    confidence = std::min(100.0f, std::max(0.0f, confidence));

    std::cout << "[Confidence] Output - confidence: " << confidence << "%" << std::endl;

    return confidence;
}

std::string wstream_app_wasm::get_confidence_metrics() const {
    std::lock_guard<std::mutex> lock(m_mutex);

    std::ostringstream json;
    json << "{";
    json << "\"confidence\":" << m_last_metrics.confidence << ",";
    json << "\"avg_logprob\":" << m_last_metrics.avg_logprob << ",";
    json << "\"entropy\":" << m_last_metrics.entropy << ",";
    json << "\"n_tokens\":" << m_last_metrics.n_tokens;
    json << "}";

    return json.str();
}
