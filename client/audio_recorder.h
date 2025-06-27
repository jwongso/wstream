#pragma once

#include <portaudio.h>
#include <vector>
#include <atomic>
#include <thread>
#include <functional>
#include <memory>
#include <mutex>
#include <deque>
#include <fstream>

/**
 * @class audio_recorder_pa
 * @brief PortAudio-based audio recorder for cross-platform microphone input
 *
 * Records audio at 16kHz mono and provides it as 16-bit PCM samples.
 * Works on Linux, macOS, and Windows.
 */
class audio_recorder {
public:
    /// Audio callback type: receives PCM samples
    using audio_callback_t = std::function<void(const std::vector<int16_t>&)>;

    /// Sample rate for recording
    static constexpr unsigned int SAMPLE_RATE = 16000;

    /// Number of channels (mono)
    static constexpr unsigned int CHANNELS = 1;

    /// Frames per buffer (adjust for latency vs stability)
    static constexpr unsigned int FRAMES_PER_BUFFER = 1024;

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
     * @param device_name Device name or empty for default
     * @return true if successful, false otherwise
     */
    bool initialize(const std::string& device_name = "");

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

    /**
     * @brief List available audio devices
     * @return Vector of device names
     */
    std::vector<std::string> list_devices();

    void enable_audio_dump(const std::string& filename);
    void disable_audio_dump();

private:
    /// PortAudio stream
    PaStream* m_stream = nullptr;

    /// Recording state
    std::atomic<bool> m_recording{false};

    /// Audio callback
    audio_callback_t m_callback;

    /// Audio buffer
    std::deque<int16_t> m_buffer;

    /// Buffer mutex
    std::mutex m_buffer_mutex;

    /// Processing thread
    std::unique_ptr<std::thread> m_thread;

    /// Device index
    int m_device_index = -1;

    std::ofstream m_audio_dump_file;
    std::mutex m_dump_mutex;
    bool m_dump_enabled = false;

    /**
     * @brief PortAudio callback function
     * @param input Input buffer
     * @param output Output buffer (not used)
     * @param frameCount Number of frames in buffer
     * @param timeInfo Time information (not used)
     * @param statusFlags Status flags (not used)
     * @param userData User data (this instance)
     * @return paContinue if successful, paAbort otherwise
     */
    static int pa_callback(const void* input, void* output,
                           unsigned long frameCount,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void* userData);

    /**
     * @brief Processing thread function
     */
    void process_audio_thread();

    /**
     * @brief Find device index by name
     * @param name Device name to find
     * @return Device index or -1 if not found
     */
    int find_device_by_name(const std::string& name);
};
