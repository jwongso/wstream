#include <gtest/gtest.h>
#include "websocket_audio_source.h"
#include <thread>
#include <chrono>

class WebSocketAudioSourceTest : public ::testing::Test {
protected:
    void SetUp() override {
        audio_source = std::make_unique<websocket_audio_source>();
        audio_source->initialize();
        audio_source->start();
    }

    void TearDown() override {
        audio_source->stop();
        audio_source.reset();
    }

    std::unique_ptr<websocket_audio_source> audio_source;
};

TEST_F(WebSocketAudioSourceTest, InitializeReturnsTrue) {
    websocket_audio_source source;
    EXPECT_TRUE(source.initialize());
}

TEST_F(WebSocketAudioSourceTest, StartReturnsTrue) {
    websocket_audio_source source;
    source.initialize();
    EXPECT_TRUE(source.start());
}

TEST_F(WebSocketAudioSourceTest, GetNameReturnsWebSocketClient) {
    EXPECT_EQ(audio_source->get_name(), "WebSocket Client");
}

TEST_F(WebSocketAudioSourceTest, InitiallyInactive) {
    websocket_audio_source source;
    source.initialize();
    EXPECT_FALSE(source.is_active());
}

TEST_F(WebSocketAudioSourceTest, NoSamplesWhenEmpty) {
    std::vector<float> samples;
    EXPECT_FALSE(audio_source->get_audio_samples(samples));
    EXPECT_TRUE(samples.empty());
}

TEST_F(WebSocketAudioSourceTest, HandlesAudioData) {
    // Create test audio data
    std::vector<int16_t> test_samples(1600, 1000);  // 100ms of 1kHz sine wave
    std::string test_session = "test-session";
    std::string test_language = "en-US";

    // Send to audio source
    audio_source->handle_audio_data(test_samples, test_session, test_language);

    // Give a moment for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Should now be active
    EXPECT_TRUE(audio_source->is_active());

    // Should return samples
    std::vector<float> received_samples;
    EXPECT_TRUE(audio_source->get_audio_samples(received_samples));
    EXPECT_FALSE(received_samples.empty());
    EXPECT_EQ(received_samples.size(), test_samples.size());

    // Should have set session ID and language
    EXPECT_EQ(audio_source->get_session_id(), test_session);
    EXPECT_EQ(audio_source->get_language(), test_language);

    // Check sample conversion (int16_t to float)
    const float scale = 1.0f / 32768.0f;
    EXPECT_NEAR(received_samples[0], test_samples[0] * scale, 0.0001f);
}

TEST_F(WebSocketAudioSourceTest, BecomesInactiveAfterTimeout) {
    // Create test audio data
    std::vector<int16_t> test_samples(1600, 1000);

    // Send to audio source
    audio_source->handle_audio_data(test_samples);

    // Should now be active
    EXPECT_TRUE(audio_source->is_active());

    // Wait for activity timeout (set to 5000ms in implementation)
    std::this_thread::sleep_for(std::chrono::milliseconds(5500));

    // Should now be inactive
    EXPECT_FALSE(audio_source->is_active());
}

TEST_F(WebSocketAudioSourceTest, StopClearsQueue) {
    // Create test audio data
    std::vector<int16_t> test_samples(1600, 1000);

    // Send multiple packets
    for (int i = 0; i < 5; i++) {
        audio_source->handle_audio_data(test_samples);
    }

    // Stop the audio source
    audio_source->stop();

    // Should be inactive
    EXPECT_FALSE(audio_source->is_active());

    // Should not return any samples
    std::vector<float> received_samples;
    EXPECT_FALSE(audio_source->get_audio_samples(received_samples));
    EXPECT_TRUE(received_samples.empty());
}
