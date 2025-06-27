#include "audio_recorder.h"
#include <iostream>
#include <cstring>

audio_recorder::audio_recorder() = default;

audio_recorder::~audio_recorder() {
    stop_recording();

    if (m_pcm_handle) {
        snd_pcm_close(m_pcm_handle);
        m_pcm_handle = nullptr;
    }
}

bool audio_recorder::initialize(const std::string& device_name) {
    // Open PCM device for recording
    int err = snd_pcm_open(&m_pcm_handle, device_name.c_str(), SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
        std::cerr << "Cannot open audio device " << device_name << ": " << snd_strerror(err) << std::endl;
        return false;
    }

    if (!setup_alsa_params()) {
        snd_pcm_close(m_pcm_handle);
        m_pcm_handle = nullptr;
        return false;
    }

    std::cout << "Audio recorder initialized successfully" << std::endl;
    std::cout << "Sample rate: " << SAMPLE_RATE << " Hz" << std::endl;
    std::cout << "Channels: " << CHANNELS << std::endl;
    std::cout << "Buffer size: " << BUFFER_SIZE << " frames" << std::endl;

    return true;
}

bool audio_recorder::setup_alsa_params() {
    snd_pcm_hw_params_t* hw_params;
    int err;

    // Allocate hardware parameters object
    if ((err = snd_pcm_hw_params_malloc(&hw_params)) < 0) {
        std::cerr << "Cannot allocate hardware parameter structure: " << snd_strerror(err) << std::endl;
        return false;
    }

    // Initialize hardware parameters
    if ((err = snd_pcm_hw_params_any(m_pcm_handle, hw_params)) < 0) {
        std::cerr << "Cannot initialize hardware parameter structure: " << snd_strerror(err) << std::endl;
        snd_pcm_hw_params_free(hw_params);
        return false;
    }

    // Set access type
    if ((err = snd_pcm_hw_params_set_access(m_pcm_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        std::cerr << "Cannot set access type: " << snd_strerror(err) << std::endl;
        snd_pcm_hw_params_free(hw_params);
        return false;
    }

    // Set sample format
    if ((err = snd_pcm_hw_params_set_format(m_pcm_handle, hw_params, SAMPLE_FORMAT)) < 0) {
        std::cerr << "Cannot set sample format: " << snd_strerror(err) << std::endl;
        snd_pcm_hw_params_free(hw_params);
        return false;
    }

    // Set sample rate
    unsigned int rate = SAMPLE_RATE;
    if ((err = snd_pcm_hw_params_set_rate_near(m_pcm_handle, hw_params, &rate, 0)) < 0) {
        std::cerr << "Cannot set sample rate: " << snd_strerror(err) << std::endl;
        snd_pcm_hw_params_free(hw_params);
        return false;
    }

    if (rate != SAMPLE_RATE) {
        std::cerr << "Warning: Requested rate " << SAMPLE_RATE << " Hz, got " << rate << " Hz" << std::endl;
    }

    // Set number of channels
    if ((err = snd_pcm_hw_params_set_channels(m_pcm_handle, hw_params, CHANNELS)) < 0) {
        std::cerr << "Cannot set channel count: " << snd_strerror(err) << std::endl;
        snd_pcm_hw_params_free(hw_params);
        return false;
    }

    // Set buffer size
    snd_pcm_uframes_t buffer_size = BUFFER_SIZE;
    if ((err = snd_pcm_hw_params_set_buffer_size_near(m_pcm_handle, hw_params, &buffer_size)) < 0) {
        std::cerr << "Cannot set buffer size: " << snd_strerror(err) << std::endl;
        snd_pcm_hw_params_free(hw_params);
        return false;
    }

    // Apply hardware parameters
    if ((err = snd_pcm_hw_params(m_pcm_handle, hw_params)) < 0) {
        std::cerr << "Cannot set hardware parameters: " << snd_strerror(err) << std::endl;
        snd_pcm_hw_params_free(hw_params);
        return false;
    }

    snd_pcm_hw_params_free(hw_params);

    // Prepare the PCM for use
    if ((err = snd_pcm_prepare(m_pcm_handle)) < 0) {
        std::cerr << "Cannot prepare audio interface for use: " << snd_strerror(err) << std::endl;
        return false;
    }

    return true;
}

bool audio_recorder::start_recording(audio_callback_t callback) {
    if (m_recording) {
        std::cerr << "Already recording" << std::endl;
        return false;
    }

    if (!m_pcm_handle) {
        std::cerr << "Audio recorder not initialized" << std::endl;
        return false;
    }

    m_callback = callback;
    m_recording = true;

    m_record_thread = std::make_unique<std::thread>(&audio_recorder::record_thread_func, this);

    std::cout << "Started recording..." << std::endl;
    return true;
}

void audio_recorder::stop_recording() {
    if (!m_recording) {
        return;
    }

    m_recording = false;

    if (m_record_thread && m_record_thread->joinable()) {
        m_record_thread->join();
    }

    m_record_thread.reset();

    std::cout << "Stopped recording" << std::endl;
}

void audio_recorder::record_thread_func() {
    std::vector<int16_t> buffer(BUFFER_SIZE);

    while (m_recording) {
        snd_pcm_sframes_t frames = snd_pcm_readi(m_pcm_handle, buffer.data(), BUFFER_SIZE);

        if (frames < 0) {
            if (frames == -EPIPE) {
                // Buffer overrun
                std::cerr << "Buffer overrun occurred" << std::endl;
                snd_pcm_prepare(m_pcm_handle);
                continue;
            } else if (frames == -ESTRPIPE) {
                // Suspend, try to recover
                std::cerr << "Stream suspended, trying to recover" << std::endl;
                while ((frames = snd_pcm_resume(m_pcm_handle)) == -EAGAIN) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                if (frames < 0) {
                    frames = snd_pcm_prepare(m_pcm_handle);
                }
                continue;
            } else {
                std::cerr << "Read error: " << snd_strerror(frames) << std::endl;
                break;
            }
        }

        if (frames > 0 && m_callback) {
            // Resize buffer to actual frames read
            buffer.resize(frames);
            m_callback(buffer);
            buffer.resize(BUFFER_SIZE);  // Restore buffer size
        }
    }
}
