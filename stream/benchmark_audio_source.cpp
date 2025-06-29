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

// WAV file header structure
struct wav_header {
    char riff[4];           // "RIFF"
    uint32_t file_size;     // File size - 8
    char wave[4];           // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmt_size;      // Format chunk size
    uint16_t audio_format;  // Audio format (1 = PCM)
    uint16_t channels;      // Number of channels
    uint32_t sample_rate;   // Sample rate
    uint32_t byte_rate;     // Byte rate
    uint16_t block_align;   // Block align
    uint16_t bits_per_sample; // Bits per sample
    char data[4];           // "data"
    uint32_t data_size;     // Data size
};

benchmark_audio_source::benchmark_audio_source(const config& cfg)
    : m_config(cfg) {
}

benchmark_audio_source::~benchmark_audio_source() {
    stop();
}

bool benchmark_audio_source::initialize() {
    std::cout << "[Benchmark] Initializing benchmark audio source..." << std::endl;

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

    std::cout << "[Benchmark] Started benchmark audio source" << std::endl;
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
    if (!m_active || m_current_position >= m_audio_buffer.size()) {
        if (m_config.loop_audio && m_current_position >= m_audio_buffer.size()) {
            m_current_position = 0;
            std::cout << "[Benchmark] Looping audio..." << std::endl;
        } else if (m_current_position >= m_audio_buffer.size()) {
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

    // Use configurable chunk size instead of hardcoded 3 seconds
    const size_t chunk_samples = (m_config.chunk_size_ms * m_config.sample_rate) / 1000;

    // Calculate how many samples to read
    size_t samples_to_read = std::min(chunk_samples, m_audio_buffer.size() - m_current_position);

    // Clear and fill output vector
    samples.clear();
    samples.reserve(samples_to_read);
    samples.insert(samples.end(),
                   m_audio_buffer.begin() + m_current_position,
                   m_audio_buffer.begin() + m_current_position + samples_to_read);

    // Update position
    m_current_position += samples_to_read;

    // Update statistics
    m_total_samples_processed += samples_to_read;
    m_total_chunks_processed++;

    // Simulate real-time delay if enabled
    if (m_config.real_time_simulation) {
        int delay_ms = static_cast<int>((samples_to_read * 1000) / m_config.sample_rate);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

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
        if (!file.good()) break;  // End of file

        file.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (!file.good()) break;

        // Debug output
        std::cout << "[Benchmark] Found chunk: '" << std::string(chunk_id, 4)
                  << "' size: " << chunk_size << " bytes" << std::endl;

        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            // Format chunk
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
            // Data chunk
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
            m_audio_buffer.reserve(num_frames);  // Reserve for mono output

            if (channels == 1) {
                // Mono - direct conversion
                const float scale = 1.0f / 32768.0f;
                for (const auto& sample : pcm_data) {
                    m_audio_buffer.push_back(sample * scale);
                }
            } else {
                // Multi-channel - convert to mono
                const float scale = 1.0f / (32768.0f * channels);
                for (size_t i = 0; i < num_frames; i++) {
                    float sum = 0.0f;
                    for (uint16_t c = 0; c < channels; c++) {
                        sum += pcm_data[i * channels + c];
                    }
                    m_audio_buffer.push_back(sum * scale);
                }
                std::cout << "[Benchmark] Converted " << channels << " channels to mono" << std::endl;
            }

            break;  // We found the data, we're done

        } else {
            // Unknown chunk - skip it
            std::cout << "[Benchmark] Skipping unknown chunk: '"
                      << std::string(chunk_id, 4) << "'" << std::endl;

            // Make sure chunk_size is reasonable
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

void benchmark_audio_source::convert_pcm_to_float(const std::vector<int16_t>& pcm,
                                                  std::vector<float>& output) {
    output.clear();
    output.reserve(pcm.size());

    const float scale = 1.0f / 32768.0f;
    for (const auto& sample : pcm) {
        output.push_back(sample * scale);
    }
}

void benchmark_audio_source::simulate_real_time_delay() {
    // Calculate how much time should have elapsed for this chunk
    auto chunk_duration_ms = std::chrono::milliseconds(m_config.chunk_size_ms);
    auto target_time = m_last_chunk_time + chunk_duration_ms;

    // Sleep until target time
    auto now = std::chrono::steady_clock::now();
    if (now < target_time) {
        std::this_thread::sleep_until(target_time);
    }

    m_last_chunk_time = std::chrono::steady_clock::now();
}

double benchmark_audio_source::get_processing_duration_ms()const {
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
}

void benchmark_audio_source::set_chunk_size_ms(int ms) {
    if (ms > 0 && ms <= 1000) {
        m_config.chunk_size_ms = ms;
    }
}
