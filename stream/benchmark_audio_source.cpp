// -------------------------------------------------------------------------------------------------
//
// Copyright (C) all of the contributors. All rights reserved.
//
// This software, including documentation, is protected by copyright controlled by
// contributors. All rights are reserved. Copying, including reproducing, storing,
// adapting or translating, any or all of this material requires the prior written
// consent of all contributors.
//
// -------------------------------------------------------------------------------------------------

#include "benchmark_audio_source.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <thread>
#include <algorithm>

benchmark_audio_source::benchmark_audio_source(const config& cfg)
    : m_config(cfg) {

    // Pre-calculate sample counts for efficiency
    m_n_samples_30s = static_cast<int>((MS_TO_SECONDS * BUFFER_30S_DURATION) * m_config.sample_rate);
    m_n_samples_step = static_cast<int>((MS_TO_SECONDS * m_config.step_ms) * m_config.sample_rate);
    m_n_samples_keep = static_cast<int>((MS_TO_SECONDS * m_config.keep_ms) * m_config.sample_rate);

    // Pre-allocate vectors with capacity to avoid reallocations
    m_pcmf32_new.reserve(m_n_samples_30s);
    m_pcmf32_old.reserve(m_n_samples_keep);

    // Initialize to empty
    m_pcmf32_new.clear();
    m_pcmf32_old.clear();
}

benchmark_audio_source::~benchmark_audio_source() {
    stop();
}

bool benchmark_audio_source::initialize() {
    std::cout << "[Benchmark] Initializing benchmark audio source..." << std::endl;
    std::cout << "[Benchmark] Configuration:" << std::endl;
    std::cout << "  - Step size: " << m_config.step_ms << " ms" << std::endl;
    std::cout << "  - Buffer length: " << m_config.length_ms << " ms" << std::endl;
    std::cout << "  - Keep/overlap: " << m_config.keep_ms << " ms" << std::endl;
    std::cout << "  - VAD mode: " << (m_config.use_vad ? "enabled" : "disabled") << std::endl;

    // Load WAV file
    if (!load_wav_file(m_config.wav_file_path)) {
        std::cerr << "[Benchmark] Failed to load WAV file: " << m_config.wav_file_path << std::endl;
        return false;
    }

    // Load reference text if available
    if (!m_config.reference_text_path.empty()) {
        if (!load_reference_text(m_config.reference_text_path)) {
            std::cerr << "[Benchmark] Warning: Failed to load reference text: "
                      << m_config.reference_text_path << std::endl;
            // Non-fatal - continue without reference
        }
    }

    std::cout << "[Benchmark] Loaded " << m_audio_buffer.size() << " samples ("
              << get_audio_duration_ms() / 1000.0 << " seconds)" << std::endl;

    return true;
}

bool benchmark_audio_source::start() {
    if (m_audio_buffer.empty()) {
        std::cerr << "[Benchmark] No audio data loaded" << std::endl;
        return false;
    }

    m_active = true;
    m_current_position = 0;
    m_total_samples_processed = 0;
    m_total_chunks_processed = 0;
    m_start_time = std::chrono::steady_clock::now();
    m_last_chunk_time = m_start_time;
    m_last_vad_time = m_start_time;
    m_end_of_file_reported = false;
    m_last_processed_end = 0;

    // Clear buffers for fresh start
    m_pcmf32_new.clear();
    m_pcmf32_old.clear();

    std::cout << "[Benchmark] Started benchmark audio source (simulating audio_processor behavior)" << std::endl;
    return true;
}

void benchmark_audio_source::stop() {
    m_active = false;

    if (m_total_samples_processed > 0) {
        auto duration = get_processing_duration_ms();
        auto audio_duration = get_audio_duration_ms();
        double rtf = duration / audio_duration;  // Real-time factor

        std::cout << "[Benchmark] Stopped. Statistics:" << std::endl;
        std::cout << "  Total samples: " << m_total_samples_processed << std::endl;
        std::cout << "  Total chunks: " << m_total_chunks_processed << std::endl;
        std::cout << "  Processing time: " << duration / 1000.0 << " seconds" << std::endl;
        std::cout << "  Audio duration: " << audio_duration / 1000.0 << " seconds" << std::endl;
        std::cout << "  Real-time factor: " << rtf << "x" << std::endl;
    }
}

bool benchmark_audio_source::get_audio_samples(std::vector<float>& samples) {
    if (!m_active) {
        return false;
    }

    samples.clear();

    bool got_samples = false;

    try {
        if (!m_config.use_vad) {
            got_samples = process_non_vad(samples);
        } else {
            got_samples = process_vad(samples);
        }
    } catch (const std::exception& e) {
        std::cerr << "[Benchmark] Exception during audio processing: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "[Benchmark] Unknown exception during audio processing" << std::endl;
        return false;
    }

    if (!got_samples) {
        // Check if we've reached the end of the file
        if (m_current_position >= m_audio_buffer.size()) {
            if (m_config.loop_audio) {
                // Reset for looping
                m_current_position = 0;
                m_pcmf32_old.clear();
                m_last_processed_end = 0;  // Reset tracking for VAD mode
                std::cout << "[Benchmark] Looping audio..." << std::endl;
                return get_audio_samples(samples); // Recursive call to get samples from start
            } else {
                if (!m_end_of_file_reported) {
                    std::cout << "[Benchmark] End of audio file reached" << std::endl;
                    m_end_of_file_reported = true;

                    if (m_completion_callback) {
                        m_completion_callback();
                    }
                }
                m_active = false;
                return false;
            }
        }
        return false;
    }

    // Update statistics
    m_total_samples_processed += samples.size();
    m_total_chunks_processed++;

    // Simulate real-time delay if enabled
    if (m_config.real_time_simulation) {
        simulate_real_time_delay();
    } else {
        // Small delay to prevent CPU spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return true;
}

bool benchmark_audio_source::process_non_vad(std::vector<float>& samples) {
    // Step 1: Read new audio data from buffer
    m_pcmf32_new.clear();

    // Calculate how many samples we need
    size_t samples_needed = m_n_samples_step;
    size_t samples_available = m_audio_buffer.size() - m_current_position;

    if (samples_available == 0) {
        return false; // No more audio
    }

    // Read up to step_ms of audio
    size_t samples_to_read = std::min(samples_needed, samples_available);

    // Copy samples from main buffer to new buffer
    m_pcmf32_new.reserve(samples_to_read);
    m_pcmf32_new.insert(m_pcmf32_new.end(),
                        m_audio_buffer.begin() + m_current_position,
                        m_audio_buffer.begin() + m_current_position + samples_to_read);

    // Update position
    m_current_position += samples_to_read;

    // Step 2: Determine overlap amount
    const int n_samples_new = m_pcmf32_new.size();
    const int n_samples_take = std::min(static_cast<int>(m_pcmf32_old.size()), m_n_samples_keep);

    // Step 3: Resize the processing buffer to fit overlap + new samples
    samples.resize(n_samples_new + n_samples_take);

    // Step 4: Copy overlap samples first
    if (n_samples_take > 0) {
        std::copy(m_pcmf32_old.end() - n_samples_take, m_pcmf32_old.end(), samples.begin());
    }

    // Step 5: Copy new samples
    std::copy(m_pcmf32_new.begin(), m_pcmf32_new.end(), samples.begin() + n_samples_take);

    // Step 6: Keep only overlap amount for next iteration
    if (static_cast<int>(samples.size()) >= m_n_samples_keep) {
        m_pcmf32_old.assign(samples.end() - m_n_samples_keep, samples.end());
    } else {
        m_pcmf32_old = samples;
    }

    return true;
}

bool benchmark_audio_source::process_vad(std::vector<float>& samples) {
    const auto t_now = std::chrono::steady_clock::now();
    const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                            t_now - m_last_vad_time).count();

    if (t_diff < 100) {  // Short interval to prevent duplicate processing
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        samples.clear();
        return false;
    }

    // Calculate samples for detection and processing
    const size_t detection_samples = (VAD_DETECTION_WINDOW_MS * m_config.sample_rate) / 1000;
    const size_t segment_samples = (VAD_SEGMENT_SIZE_MS * m_config.sample_rate) / 1000;
    size_t samples_available = m_audio_buffer.size() - m_current_position;

    if (samples_available < detection_samples) {
        return false;
    }

    // Check energy in detection window
    float energy = 0.0f;
    const size_t end_pos = std::min(m_current_position + detection_samples, m_audio_buffer.size());
    for (size_t i = m_current_position; i < end_pos; i++) {
        float sample = m_audio_buffer[i];
        energy += sample * sample;
    }
    energy /= detection_samples;

    // Determine if speech is detected
    bool speech_detected = (energy > VAD_ENERGY_THRESHOLD);
    if (m_config.force_vad_detection && energy > VAD_MIN_ENERGY) {
        speech_detected = true;
    }

    if (speech_detected) {
        // Get audio segment for processing
        size_t samples_to_read = std::min(segment_samples, samples_available);

        // Avoid duplicating content
        if (m_current_position < m_last_processed_end) {
            if (m_last_processed_end >= m_audio_buffer.size()) {
                return false;
            }
            m_current_position = m_last_processed_end;
            samples_available = m_audio_buffer.size() - m_current_position;
            if (samples_available < detection_samples) {
                return false;
            }
            samples_to_read = std::min(segment_samples, samples_available);
        }

        // Prepare output buffer
        samples.clear();
        samples.reserve(samples_to_read);

        // Copy samples with bounds checking
        const size_t safe_end = std::min(m_current_position + samples_to_read, m_audio_buffer.size());
        samples.insert(samples.end(),
                       m_audio_buffer.begin() + m_current_position,
                       m_audio_buffer.begin() + safe_end);

        // Update last processed position
        m_last_processed_end = m_current_position + samples.size();

        // Advance position for next check
        const size_t advance_samples = (VAD_ADVANCE_MS * m_config.sample_rate) / 1000;
        m_current_position = std::min(m_current_position + advance_samples, m_audio_buffer.size());
    } else {
        // No speech detected - advance by smaller step
        const size_t skip_samples = (VAD_SKIP_MS * m_config.sample_rate) / 1000;
        m_current_position = std::min(m_current_position + skip_samples, m_audio_buffer.size());

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        samples.clear();
        return false;
    }

    m_last_vad_time = t_now;
    return true;
}

bool benchmark_audio_source::is_active() const {
    return m_active;
}

bool benchmark_audio_source::load_wav_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Benchmark] Cannot open WAV file: " << file_path << std::endl;
        return false;
    }

    // Read RIFF header
    char riff[4];
    uint32_t file_size;
    char wave[4];

    file.read(riff, 4);
    file.read(reinterpret_cast<char*>(&file_size), 4);
    file.read(wave, 4);

    if (!file.good() || std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0) {
        std::cerr << "[Benchmark] Invalid WAV file format" << std::endl;
        return false;
    }

    // Parse chunks
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    bool found_fmt = false;
    bool found_data = false;

    while (!file.eof() && file.good()) {
        char chunk_id[4];
        uint32_t chunk_size;

        // Read chunk header
        file.read(chunk_id, 4);
        if (!file.good()) break;

        file.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (!file.good()) break;

        // Debug output
        std::cout << "[Benchmark] Found chunk: '" << std::string(chunk_id, 4)
                  << "' size: " << chunk_size << " bytes" << std::endl;

        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            found_fmt = true;

            // Read format data
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&channels), 2);
            file.read(reinterpret_cast<char*>(&sample_rate), 4);
            uint32_t byte_rate;
            file.read(reinterpret_cast<char*>(&byte_rate), 4);
            uint16_t block_align;
            file.read(reinterpret_cast<char*>(&block_align), 2);
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);

            // Skip any extra format bytes
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }

            std::cout << "[Benchmark] Format: " << audio_format
                      << ", Channels: " << channels
                      << ", Sample Rate: " << sample_rate
                      << ", Bits: " << bits_per_sample << std::endl;

        } else if (std::strncmp(chunk_id, "data", 4) == 0) {
            if (!found_fmt) {
                std::cerr << "[Benchmark] Found data chunk before format chunk" << std::endl;
                return false;
            }

            found_data = true;

            // Validate format
            if (audio_format != 1) {  // PCM
                std::cerr << "[Benchmark] Only PCM format is supported (found format "
                          << audio_format << ")" << std::endl;
                return false;
            }

            if (bits_per_sample != 16) {
                std::cerr << "[Benchmark] Only 16-bit PCM is supported (found "
                          << bits_per_sample << " bits)" << std::endl;
                return false;
            }

            // Check sample rate
            if (sample_rate != static_cast<uint32_t>(m_config.sample_rate)) {
                std::cerr << "[Benchmark] Warning: Sample rate mismatch. Expected "
                          << m_config.sample_rate << " Hz, got " << sample_rate << " Hz" << std::endl;

                if (m_config.strict_format_validation) {
                    return false;
                }
            }

            // Read PCM data
            size_t num_samples = chunk_size / (bits_per_sample / 8);
            size_t num_frames = num_samples / channels;

            std::cout << "[Benchmark] Reading " << num_samples << " samples ("
                      << num_frames << " frames)" << std::endl;

            std::vector<int16_t> pcm_data(num_samples);
            file.read(reinterpret_cast<char*>(pcm_data.data()), chunk_size);

            if (!file.good() && !file.eof()) {
                std::cerr << "[Benchmark] Failed to read audio data" << std::endl;
                return false;
            }

            // Convert to float
            m_audio_buffer.clear();
            m_audio_buffer.reserve(num_frames);

            const float scale = 1.0f / 32768.0f;
            if (channels == 1) {
                // Mono - direct conversion
                for (const auto& sample : pcm_data) {
                    m_audio_buffer.push_back(sample * scale);
                }
            } else {
                // Multi-channel - convert to mono by averaging
                for (size_t i = 0; i < num_frames; i++) {
                    float sum = 0.0f;
                    for (uint16_t c = 0; c < channels; c++) {
                        sum += pcm_data[i * channels + c] * scale;
                    }
                    m_audio_buffer.push_back(sum / channels);
                }
                std::cout << "[Benchmark] Converted " << channels << " channels to mono" << std::endl;
            }

            break;  // We found the data, we're done

        } else {
            // Unknown chunk - skip it
            std::cout << "[Benchmark] Skipping unknown chunk: '"
                      << std::string(chunk_id, 4) << "'" << std::endl;

            if (chunk_size > file_size) {
                std::cerr << "[Benchmark] Invalid chunk size: " << chunk_size << std::endl;
                return false;
            }

            // Skip this chunk
            file.seekg(chunk_size, std::ios::cur);

            // Align to word boundary if chunk size is odd
            if (chunk_size % 2 == 1) {
                file.seekg(1, std::ios::cur);
            }
        }
    }

    if (!found_fmt) {
        std::cerr << "[Benchmark] No format chunk found in WAV file" << std::endl;
        return false;
    }

    if (!found_data) {
        std::cerr << "[Benchmark] No data chunk found in WAV file" << std::endl;
        return false;
    }

    std::cout << "[Benchmark] Successfully loaded " << m_audio_buffer.size()
              << " samples (" << (m_audio_buffer.size() / static_cast<float>(m_config.sample_rate))
              << " seconds)" << std::endl;

    return true;
}

bool benchmark_audio_source::load_reference_text(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return false;
    }

    m_reference_text.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!m_reference_text.empty()) {
            m_reference_text += " ";
        }
        m_reference_text += line;
    }

    std::cout << "[Benchmark] Loaded reference text ("
              << m_reference_text.length() << " characters)" << std::endl;

    return true;
}

void benchmark_audio_source::simulate_real_time_delay() {
    // Calculate how much time should have elapsed for this chunk
    auto chunk_duration_ms = std::chrono::milliseconds(m_config.step_ms);
    auto target_time = m_last_chunk_time + chunk_duration_ms;

    // Sleep until target time
    auto now = std::chrono::steady_clock::now();
    if (now < target_time) {
        std::this_thread::sleep_until(target_time);
    }

    m_last_chunk_time = std::chrono::steady_clock::now();
}

double benchmark_audio_source::get_processing_duration_ms() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - m_start_time).count();
}

double benchmark_audio_source::get_audio_duration_ms() const {
    if (m_config.sample_rate == 0) return 0.0;
    return (m_audio_buffer.size() * 1000.0) / m_config.sample_rate;
}

void benchmark_audio_source::reset() {
    m_current_position = 0;
    m_total_samples_processed = 0;
    m_total_chunks_processed = 0;
    m_start_time = std::chrono::steady_clock::now();
    m_last_chunk_time = m_start_time;
    m_last_vad_time = m_start_time;
    m_end_of_file_reported = false;
    m_last_processed_end = 0;

    // Clear buffers
    m_pcmf32_new.clear();
    m_pcmf32_old.clear();
}

void benchmark_audio_source::set_chunk_size_ms(int ms) {
    // Clamp to valid range
    ms = std::max(MIN_CHUNK_SIZE_MS, std::min(ms, MAX_CHUNK_SIZE_MS));

    m_config.step_ms = ms;

    // Recalculate sample counts
    m_n_samples_step = static_cast<int>((MS_TO_SECONDS * m_config.step_ms) * m_config.sample_rate);

    std::cout << "[Benchmark] Chunk size changed to " << ms << " ms ("
              << m_n_samples_step << " samples)" << std::endl;
}
