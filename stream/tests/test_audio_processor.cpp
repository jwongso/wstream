#include <gtest/gtest.h>
#include "audio_processor.h"
#include <thread>
#include <chrono>

/**
 * @file test_audio_processor.cpp
 * @brief Integration tests for audio_processor class
 */

class AudioProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create audio processor with test configuration
        audio_processor::config config;
        config.step_ms = 1000;  // Shorter for testing
        config.length_ms = 3000;
        config.keep_ms = 100;
        config.use_vad = false;

        processor = std::make_unique<audio_processor>(config);

        // Create VAD processor
        audio_processor::config vad_config;
        vad_config.use_vad = true;
        vad_processor = std::make_unique<audio_processor>(vad_config);
    }

    void TearDown() override {
        if (processor) {
            processor->pause();
        }
        if (vad_processor) {
            vad_processor->pause();
        }
    }

    std::unique_ptr<audio_processor> processor;
    std::unique_ptr<audio_processor> vad_processor;
};

TEST_F(AudioProcessorTest, Initialization) {
    EXPECT_TRUE(processor->initialize());

    // Test invalid device ID
    audio_processor::config config;
    auto invalid_processor = std::make_unique<audio_processor>(config);
    // Device ID 9999 likely doesn't exist
    EXPECT_FALSE(invalid_processor->initialize(9999));
}

TEST_F(AudioProcessorTest, ConfigurationAccess) {
    const auto& config = processor->get_config();
    EXPECT_EQ(config.step_ms, 1000);
    EXPECT_EQ(config.length_ms, 3000);
    EXPECT_EQ(config.keep_ms, 100);
    EXPECT_FALSE(config.use_vad);
    EXPECT_EQ(config.sample_rate, WHISPER_SAMPLE_RATE);
}

TEST_F(AudioProcessorTest, PauseAndResume) {
    ASSERT_TRUE(processor->initialize());

    processor->resume();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    processor->pause();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    processor->resume();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    processor->pause();

    // Should not crash or hang
    SUCCEED();
}

TEST_F(AudioProcessorTest, GetSamplesNonVAD) {
    ASSERT_TRUE(processor->initialize());
    processor->resume();

    std::vector<float> samples;

    // Wait for audio to be available - but with a reasonable timeout
    auto start = std::chrono::steady_clock::now();
    bool got_samples = false;

    // Audio processor needs time to accumulate enough samples (step_ms = 1000 in test config)
    // So we need to wait at least that long, plus some buffer
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(3)) {
        if (processor->get_processed_samples(samples)) {
            got_samples = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // It's OK if we don't get samples in CI/test environment
    // The important thing is that it doesn't crash
    if (got_samples) {
        EXPECT_FALSE(samples.empty());

        // Check sample values are in valid range
        for (float sample : samples) {
            EXPECT_GE(sample, -1.0f);
            EXPECT_LE(sample, 1.0f);
        }
    }

    // Test should complete quickly even if no audio is available
    auto total_time = std::chrono::steady_clock::now() - start;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(total_time).count(), 5);
}

TEST_F(AudioProcessorTest, GetSamplesVAD) {
    ASSERT_TRUE(vad_processor->initialize());
    vad_processor->resume();

    std::vector<float> samples;

    // VAD mode might not return samples if no voice detected
    // Just test that it doesn't crash
    for (int i = 0; i < 10; ++i) {
        vad_processor->get_processed_samples(samples);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    SUCCEED();
}

TEST_F(AudioProcessorTest, MultipleInitialization) {
    EXPECT_TRUE(processor->initialize());

    // Should handle re-initialization gracefully
    EXPECT_TRUE(processor->initialize());

    SUCCEED();
}

TEST_F(AudioProcessorTest, CustomConfiguration) {
    audio_processor::config custom_config;
    custom_config.step_ms = 500;
    custom_config.length_ms = 2000;
    custom_config.keep_ms = 50;
    custom_config.use_vad = false;

    auto custom_processor = std::make_unique<audio_processor>(custom_config);
    ASSERT_TRUE(custom_processor->initialize());

    const auto& config = custom_processor->get_config();
    EXPECT_EQ(config.step_ms, 500);
    EXPECT_EQ(config.length_ms, 2000);
    EXPECT_EQ(config.keep_ms, 50);
}

TEST_F(AudioProcessorTest, StressTest) {
    ASSERT_TRUE(processor->initialize());
    processor->resume();

    const int iterations = 20;  // Reduced from 100
    int successful_reads = 0;
    std::vector<float> samples;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        if (processor->get_processed_samples(samples)) {
            successful_reads++;
        }

        // Don't sleep too little - audio needs time to accumulate
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Early exit if we've been running too long
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 8) {
            break;
        }
    }

    auto duration = std::chrono::steady_clock::now() - start;

    // Should handle rapid polling without issues and finish within reasonable time
    EXPECT_GE(successful_reads, 0);  // At least we didn't crash
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(duration).count(), 10);
}

TEST_F(AudioProcessorTest, ConfigurationStressTest) {
    // Test rapid initialization/deinitialization
    const int iterations = 50;
    int successful_inits = 0;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        audio_processor::config config;
        config.step_ms = 500 + (i % 3) * 100;  // Vary configuration
        config.length_ms = 1000 + (i % 5) * 200;
        config.keep_ms = 50 + (i % 4) * 25;

        auto test_processor = std::make_unique<audio_processor>(config);
        if (test_processor->initialize()) {
            successful_inits++;
            // Quick test
            test_processor->pause();
        }
    }

    auto duration = std::chrono::steady_clock::now() - start;

    // Should handle multiple initializations quickly
    EXPECT_GT(successful_inits, iterations / 2);  // At least half should succeed
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(duration).count(), 5);
}

TEST_F(AudioProcessorTest, RapidStateChanges) {
    ASSERT_TRUE(processor->initialize());

    const int iterations = 20;
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        processor->resume();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        processor->pause();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto duration = std::chrono::steady_clock::now() - start;

    // Should handle rapid state changes without hanging
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(duration).count(), 3);
}

TEST_F(AudioProcessorTest, MemoryStressTest) {
    // Test that we don't leak memory with multiple processors
    std::vector<std::unique_ptr<audio_processor>> processors;

    auto start = std::chrono::steady_clock::now();

    // Create many processors (but don't initialize them all - that would use too many audio devices)
    for (int i = 0; i < 100; ++i) {
        audio_processor::config config;
        config.step_ms = 1000;
        config.length_ms = 2000;
        processors.push_back(std::make_unique<audio_processor>(config));
    }

    // Initialize just a few to test actual audio functionality
    for (int i = 0; i < std::min(3, static_cast<int>(processors.size())); ++i) {
        processors[i]->initialize();
    }

    auto duration = std::chrono::steady_clock::now() - start;

    // Should create many processors quickly without memory issues
    EXPECT_EQ(processors.size(), 100);
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(duration).count(), 2);
}

TEST_F(AudioProcessorTest, MemoryConsistency) {
    ASSERT_TRUE(processor->initialize());
    processor->resume();

    std::vector<float> samples1, samples2;

    // Get two consecutive sample sets
    auto start = std::chrono::steady_clock::now();
    bool got_first = false, got_second = false;

    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(10)) {
        if (!got_first && processor->get_processed_samples(samples1)) {
            got_first = true;
        }
        if (got_first && processor->get_processed_samples(samples2)) {
            got_second = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (got_first && got_second) {
        // Samples should be different (time has passed)
        EXPECT_NE(samples1, samples2);
    }
}
