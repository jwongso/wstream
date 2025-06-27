#pragma once

#include <alsa/asoundlib.h>
#include <vector>
#include <atomic>
#include <thread>
#include <functional>
#include <memory>

/**
 * @class audio_recorder
 * @brief ALSA-based audio recorder for capturing microphone input
 *
 * Records audio at 16kHz mono and provides it as 16-bit PCM samples.
 */
class audio_recorder {
public:
    /// Audio callback type: receives PCM samples
    using audio_callback_t = std::function<void(const std::vector<int16_t>&)>;

    /// Sample rate for recording
    static constexpr unsigned int SAMPLE_RATE = 16000;

    /// Number of channels (mono)
    static constexpr unsigned int CHANNELS = 1;

    /// Sample format (16-bit signed)
    static constexpr snd_pcm_format_t SAMPLE_FORMAT = SND_PCM_FORMAT_S16_LE;

    /// Buffer size in frames
    static constexpr snd_pcm_uframes_t BUFFER_SIZE = 1024;

    /**
     * @brief Constructor
     */
    audio_recorder();

    /**
     * @brief Destructor
     */
    ~audio_recorder();

    /**
     * @brief Initialize the audio recorder
     * @param device_name ALSA device name (default: "default")
     * @return true if successful, false otherwise
     */
    bool initialize(const std::string& device_name = "default");

    /**
     * @brief Start recording audio
     * @param callback Function to call with audio data
     * @return true if successful, false otherwise
     */
    bool start_recording(audio_callback_t callback);

    /**
     * @brief Stop recording audio
     */
    void stop_recording();

    /**
     * @brief Check if currently recording
     * @return true if recording, false otherwise
     */
    bool is_recording() const { return m_recording; }

private:
    /// ALSA PCM handle
    snd_pcm_t* m_pcm_handle = nullptr;

    /// Recording thread
    std::unique_ptr<std::thread> m_record_thread;

    /// Recording state
    std::atomic<bool> m_recording{false};

    /// Audio callback
    audio_callback_t m_callback;

    /**
     * @brief Recording thread function
     */
    void record_thread_func();

    /**
     * @brief Setup ALSA parameters
     * @return true if successful, false otherwise
     */
    bool setup_alsa_params();
};
