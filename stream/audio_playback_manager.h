// audio_playback_manager.h
#pragma once

#include <SDL.h>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <iostream>

class audio_playback_manager {
private:
    struct audio_chunk {
        std::vector<float> samples;
        size_t position = 0;
    };

    std::queue<audio_chunk> m_playback_queue;
    std::mutex m_queue_mutex;
    std::condition_variable m_queue_cv;

    SDL_AudioDeviceID m_audio_device = 0;
    SDL_AudioSpec m_audio_spec;

    std::atomic<bool> m_running{false};
    bool m_sdl_audio_initialized = false;

    // Current chunk being played
    audio_chunk m_current_chunk;
    std::mutex m_current_mutex;

    static constexpr size_t MAX_QUEUE_SIZE = 10;  // Prevent memory overflow

public:
    audio_playback_manager() = default;
    ~audio_playback_manager() {
        stop();
    }

    bool initialize(int sample_rate = 16000) {
        // Initialize SDL audio subsystem if needed
        if (SDL_WasInit(SDL_INIT_AUDIO) == 0) {
            if (SDL_InitSubSystem(SDL_INIT_AUDIO) < 0) {
                std::cerr << "[Playback] Failed to initialize SDL audio subsystem: "
                          << SDL_GetError() << std::endl;
                return false;
            }
            m_sdl_audio_initialized = true;
            std::cout << "[Playback] SDL audio subsystem initialized" << std::endl;
        }

        // List available audio devices (optional, for debugging)
        int num_devices = SDL_GetNumAudioDevices(0);
        if (num_devices > 0) {
            std::cout << "[Playback] Available audio devices:" << std::endl;
            for (int i = 0; i < num_devices; i++) {
                std::cout << "  " << i << ": " << SDL_GetAudioDeviceName(i, 0) << std::endl;
            }
        }

        SDL_AudioSpec desired_spec;
        SDL_zero(desired_spec);
        desired_spec.freq = sample_rate;
        desired_spec.format = AUDIO_F32SYS;
        desired_spec.channels = 1;
        desired_spec.samples = 512;
        desired_spec.callback = audio_callback;
        desired_spec.userdata = this;

        // Open the default audio device
        m_audio_device = SDL_OpenAudioDevice(nullptr, 0, &desired_spec, &m_audio_spec,
                                             SDL_AUDIO_ALLOW_FORMAT_CHANGE);
        if (m_audio_device == 0) {
            std::cerr << "[Playback] Failed to open audio device: " << SDL_GetError() << std::endl;
            return false;
        }

        // Start playback
        SDL_PauseAudioDevice(m_audio_device, 0);

        m_running = true;
        std::cout << "[Playback] Audio playback initialized" << std::endl;
        std::cout << "[Playback] Format: " << m_audio_spec.freq << " Hz, "
                  << (int)m_audio_spec.channels << " channel(s)" << std::endl;
        return true;
    }

    void stop() {
        m_running = false;
        m_queue_cv.notify_all();

        if (m_audio_device != 0) {
            SDL_CloseAudioDevice(m_audio_device);
            m_audio_device = 0;
        }

        // Quit SDL audio if we initialized it
        if (m_sdl_audio_initialized) {
            SDL_QuitSubSystem(SDL_INIT_AUDIO);
            m_sdl_audio_initialized = false;
        }
    }

    void queue_audio(const std::vector<float>& samples) {
        if (!m_running || samples.empty()) return;

        std::lock_guard<std::mutex> lock(m_queue_mutex);

        // Drop old audio if queue is full
        if (m_playback_queue.size() >= MAX_QUEUE_SIZE) {
            m_playback_queue.pop();
        }

        audio_chunk chunk;
        chunk.samples = samples;
        chunk.position = 0;
        m_playback_queue.push(std::move(chunk));

        m_queue_cv.notify_one();
    }

    void clear_queue() {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        std::queue<audio_chunk> empty;
        std::swap(m_playback_queue, empty);
    }

private:
    static void audio_callback(void* userdata, Uint8* stream, int len) {
        auto* manager = static_cast<audio_playback_manager*>(userdata);
        manager->fill_audio_buffer(reinterpret_cast<float*>(stream), len / sizeof(float));
    }

    void fill_audio_buffer(float* buffer, size_t samples_needed) {
        size_t samples_written = 0;

        while (samples_written < samples_needed) {
            // Check if we have a current chunk
            {
                std::lock_guard<std::mutex> lock(m_current_mutex);
                if (m_current_chunk.position < m_current_chunk.samples.size()) {
                    // Copy samples from current chunk
                    size_t samples_to_copy = std::min(
                        samples_needed - samples_written,
                        m_current_chunk.samples.size() - m_current_chunk.position
                        );

                    std::copy(
                        m_current_chunk.samples.begin() + m_current_chunk.position,
                        m_current_chunk.samples.begin() + m_current_chunk.position + samples_to_copy,
                        buffer + samples_written
                        );

                    m_current_chunk.position += samples_to_copy;
                    samples_written += samples_to_copy;
                    continue;
                }
            }

            // Need new chunk
            {
                std::lock_guard<std::mutex> lock(m_queue_mutex);
                if (!m_playback_queue.empty()) {
                    std::lock_guard<std::mutex> current_lock(m_current_mutex);
                    m_current_chunk = std::move(m_playback_queue.front());
                    m_playback_queue.pop();
                } else {
                    // No more audio, fill with silence
                    std::fill(buffer + samples_written, buffer + samples_needed, 0.0f);
                    break;
                }
            }
        }
    }
};
