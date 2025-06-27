#include "audio_recorder.h"
#include <iostream>
#include <chrono>
#include <cstring>

audio_recorder::audio_recorder() {
    // Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio initialization failed: " << Pa_GetErrorText(err) << std::endl;
    }
}

audio_recorder::~audio_recorder() {
    stop_recording();

    // Terminate PortAudio
    PaError err = Pa_Terminate();
    if (err != paNoError) {
        std::cerr << "PortAudio termination failed: " << Pa_GetErrorText(err) << std::endl;
    }
}

bool audio_recorder::initialize(const std::string& device_name) {
    if (!device_name.empty()) {
        m_device_index = find_device_by_name(device_name);
        if (m_device_index == -1) {
            std::cerr << "Audio device not found: " << device_name << std::endl;
            std::cerr << "Available devices:" << std::endl;
            auto devices = list_devices();
            for (const auto& device : devices) {
                std::cerr << "  - " << device << std::endl;
            }
            return false;
        }
    } else {
        // Use default input device
        m_device_index = Pa_GetDefaultInputDevice();
        if (m_device_index == paNoDevice) {
            std::cerr << "No default input device found" << std::endl;
            return false;
        }
    }

    // Get device info
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(m_device_index);
    if (!deviceInfo) {
        std::cerr << "Failed to get device info" << std::endl;
        return false;
    }

    std::cout << "Audio recorder initialized successfully" << std::endl;
    std::cout << "Device: " << deviceInfo->name << std::endl;
    std::cout << "Max input channels: " << deviceInfo->maxInputChannels << std::endl;
    std::cout << "Default sample rate: " << deviceInfo->defaultSampleRate << " Hz" << std::endl;
    std::cout << "Using sample rate: " << SAMPLE_RATE << " Hz" << std::endl;
    std::cout << "Channels: " << CHANNELS << std::endl;

    return true;
}

bool audio_recorder::start_recording(audio_callback_t callback) {
    if (m_recording) {
        std::cerr << "Already recording" << std::endl;
        return false;
    }

    m_callback = callback;

    // Clear buffer
    {
        std::lock_guard<std::mutex> lock(m_buffer_mutex);
        m_buffer.clear();
    }

    // Open stream
    PaStreamParameters inputParams;
    memset(&inputParams, 0, sizeof(inputParams));
    inputParams.device = m_device_index;
    inputParams.channelCount = CHANNELS;
    inputParams.sampleFormat = paInt16;
    inputParams.suggestedLatency = Pa_GetDeviceInfo(m_device_index)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(&m_stream,
                                &inputParams,
                                nullptr, // No output
                                SAMPLE_RATE,
                                FRAMES_PER_BUFFER,
                                paClipOff,
                                pa_callback,
                                this);

    if (err != paNoError) {
        std::cerr << "Failed to open audio stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    // Start stream
    err = Pa_StartStream(m_stream);
    if (err != paNoError) {
        std::cerr << "Failed to start audio stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(m_stream);
        m_stream = nullptr;
        return false;
    }

    m_recording = true;

    // Start processing thread
    m_thread = std::make_unique<std::thread>(&audio_recorder::process_audio_thread, this);

    std::cout << "Started recording..." << std::endl;
    return true;
}

void audio_recorder::stop_recording() {
    if (!m_recording) {
        return;
    }

    m_recording = false;

    if (m_stream) {
        PaError err = Pa_StopStream(m_stream);
        if (err != paNoError) {
            std::cerr << "Failed to stop audio stream: " << Pa_GetErrorText(err) << std::endl;
        }

        err = Pa_CloseStream(m_stream);
        if (err != paNoError) {
            std::cerr << "Failed to close audio stream: " << Pa_GetErrorText(err) << std::endl;
        }

        m_stream = nullptr;
    }

    if (m_thread && m_thread->joinable()) {
        m_thread->join();
    }

    m_thread.reset();

    std::cout << "Stopped recording" << std::endl;
}

int audio_recorder::pa_callback(const void* input, void* output,
                                unsigned long frameCount,
                                const PaStreamCallbackTimeInfo* timeInfo,
                                PaStreamCallbackFlags statusFlags,
                                void* userData) {
    (void)output; // Unused
    (void)timeInfo;
    (void)statusFlags;

    audio_recorder* recorder = static_cast<audio_recorder*>(userData);
    if (!recorder->m_recording) {
        return paAbort;
    }

    const int16_t* inputBuffer = static_cast<const int16_t*>(input);
    const size_t numSamples = frameCount * CHANNELS;

    // Add samples to buffer
    {
        std::lock_guard<std::mutex> lock(recorder->m_buffer_mutex);
        for (size_t i = 0; i < numSamples; ++i) {
            recorder->m_buffer.push_back(inputBuffer[i]);
        }
    }

    return paContinue;
}

void audio_recorder::process_audio_thread() {
    const size_t BATCH_SIZE = FRAMES_PER_BUFFER * CHANNELS;
    std::vector<int16_t> batch;
    batch.reserve(BATCH_SIZE);

    while (m_recording) {
        // Get samples from buffer
        {
            std::lock_guard<std::mutex> lock(m_buffer_mutex);
            while (!m_buffer.empty() && batch.size() < BATCH_SIZE) {
                batch.push_back(m_buffer.front());
                m_buffer.pop_front();
            }
        }

        // Process batch if not empty
        if (!batch.empty() && m_callback) {
            m_callback(batch);
            batch.clear();
        } else {
            // Sleep a bit to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

std::vector<std::string> audio_recorder::list_devices() {
    std::vector<std::string> result;

    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(numDevices) << std::endl;
        return result;
    }

    for (int i = 0; i < numDevices; ++i) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo && deviceInfo->maxInputChannels > 0) {
            std::string name = deviceInfo->name;
            const PaHostApiInfo* hostInfo = Pa_GetHostApiInfo(deviceInfo->hostApi);
            if (hostInfo) {
                name += " (" + std::string(hostInfo->name) + ")";
            }
            result.push_back(name);
        }
    }

    return result;
}

int audio_recorder::find_device_by_name(const std::string& name) {
    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(numDevices) << std::endl;
        return -1;
    }

    for (int i = 0; i < numDevices; ++i) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo && deviceInfo->maxInputChannels > 0) {
            std::string deviceName = deviceInfo->name;
            if (deviceName.find(name) != std::string::npos) {
                return i;
            }
        }
    }

    return -1;
}
