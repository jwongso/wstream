#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "audio_processor.h"
#include <cmath>

class AudioProcessorSourceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create with default config
        audio_processor::config cfg;
        processor = std::make_unique<audio_processor>(cfg);
    }

    void TearDown() override {
        if (processor) {
            processor->stop();
        }
        processor.reset();
    }

    std::unique_ptr<audio_processor> processor;
};

TEST_F(AudioProcessorSourceTest, ImplementsAudioSourceInterface) {
    // Test that audio_processor properly implements audio_source interface
    audio_source* source = dynamic_cast<audio_source*>(processor.get());
    ASSERT_NE(source, nullptr);
}

TEST_F(AudioProcessorSourceTest, GetNameReturnsMicrophone) {
    EXPECT_EQ(processor->get_name(), "SDL Microphone");
}

TEST_F(AudioProcessorSourceTest, InitiallyInactive) {
    EXPECT_FALSE(processor->is_active());
}

TEST_F(AudioProcessorSourceTest, StartAndStopToggleActiveState) {
    EXPECT_FALSE(processor->is_active());

    // Note: start() calls resume() internally, which sets active state
    processor->initialize();
    processor->start();
    EXPECT_TRUE(processor->is_active());

    processor->stop();
    EXPECT_FALSE(processor->is_active());
}

TEST_F(AudioProcessorSourceTest, GetAudioSamplesFailsWhenInactive) {
    // Don't start the processor
    EXPECT_FALSE(processor->is_active());

    std::vector<float> samples;
    EXPECT_FALSE(processor->get_audio_samples(samples));
}

TEST_F(AudioProcessorSourceTest, DefaultSessionIdIsEmpty) {
    EXPECT_EQ(processor->get_session_id(), "");
}

TEST_F(AudioProcessorSourceTest, DefaultLanguageIsEmpty) {
    EXPECT_EQ(processor->get_language(), "");
}

TEST_F(AudioProcessorSourceTest, ConfigurationIsPreserved) {
    audio_processor::config cfg;
    cfg.step_ms = 5000;
    cfg.length_ms = 15000;
    cfg.use_vad = true;

    auto custom_processor = std::make_unique<audio_processor>(cfg);

    const auto& stored_config = custom_processor->get_config();
    EXPECT_EQ(stored_config.step_ms, 5000);
    EXPECT_EQ(stored_config.length_ms, 15000);
    EXPECT_TRUE(stored_config.use_vad);
}

TEST_F(AudioProcessorSourceTest, InitializeWithInvalidDeviceHandlesGracefully) {
    // Try to initialize with an invalid device ID
    // This should return false but not crash
    bool result = processor->initialize(999);  // Very high device ID likely doesn't exist
    ASSERT_FALSE(result);
    // We don't assert the result because it depends on the system
    // But the call should not crash
    SUCCEED();
}
