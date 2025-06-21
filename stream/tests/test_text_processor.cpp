#include <gtest/gtest.h>
#include "text_processor.h"
#include <string>
#include <vector>

/**
 * @file test_text_processor.cpp
 * @brief Integration tests for text_processor class
 */

class TextProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create processors with different configurations
        default_processor = std::make_unique<text_processor>();

        text_processor::config no_brackets_config;
        no_brackets_config.remove_brackets = false;
        no_brackets_processor = std::make_unique<text_processor>(no_brackets_config);

        text_processor::config no_trim_config;
        no_trim_config.trim_whitespace = false;
        no_trim_processor = std::make_unique<text_processor>(no_trim_config);

        text_processor::config nothing_config;
        nothing_config.remove_brackets = false;
        nothing_config.remove_parentheses = false;
        nothing_config.trim_whitespace = false;
        nothing_processor = std::make_unique<text_processor>(nothing_config);
    }

    std::unique_ptr<text_processor> default_processor;
    std::unique_ptr<text_processor> no_brackets_processor;
    std::unique_ptr<text_processor> no_trim_processor;
    std::unique_ptr<text_processor> nothing_processor;
};

TEST_F(TextProcessorTest, ProcessEmptyString) {
    EXPECT_EQ(default_processor->process(""), "");
    EXPECT_EQ(default_processor->process("   "), "");
}

TEST_F(TextProcessorTest, RemoveBracketedContent) {
    std::string input = "Hello [NOISE] world";
    EXPECT_EQ(default_processor->process(input), "Hello  world");

    input = "[START] Hello world [END]";
    EXPECT_EQ(default_processor->process(input), "Hello world");

    input = "Nested [[brackets]] test";
    EXPECT_EQ(default_processor->process(input), "Nested  test");
}

TEST_F(TextProcessorTest, RemoveParentheticalContent) {
    std::string input = "Hello (cough) world";
    EXPECT_EQ(default_processor->process(input), "Hello  world");

    input = "(um) Hello world (ah)";
    EXPECT_EQ(default_processor->process(input), "Hello world");

    input = "Nested ((parentheses)) test";
    EXPECT_EQ(default_processor->process(input), "Nested  test");
}

TEST_F(TextProcessorTest, TrimWhitespace) {
    std::string input = "  Hello world  ";
    EXPECT_EQ(default_processor->process(input), "Hello world");

    input = "\t\nHello world\r\n";
    EXPECT_EQ(default_processor->process(input), "Hello world");

    input = "   \t  \n  ";
    EXPECT_EQ(default_processor->process(input), "");
}

TEST_F(TextProcessorTest, CombinedProcessing) {
    std::string input = "  [NOISE] Hello (um) world [END]  ";
    EXPECT_EQ(default_processor->process(input), "Hello  world");

    input = "[START] The (uh) quick [SOUND] brown fox  ";
    EXPECT_EQ(default_processor->process(input), "The  quick  brown fox");
}

TEST_F(TextProcessorTest, ConfigurableProcessing) {
    std::string input = "[NOISE] Hello world";

    // With brackets removal disabled
    EXPECT_EQ(no_brackets_processor->process(input), "[NOISE] Hello world");

    // With trim disabled
    input = "  Hello world  ";
    EXPECT_EQ(no_trim_processor->process(input), "  Hello world  ");

    // With everything disabled
    input = "  [NOISE] Hello (um) world  ";
    EXPECT_EQ(nothing_processor->process(input), input);
}

TEST_F(TextProcessorTest, UnmatchedBrackets) {
    std::string input = "Hello [world";
    EXPECT_EQ(default_processor->process(input), "Hello");

    input = "Hello world]";
    EXPECT_EQ(default_processor->process(input), "Hello world]");

    input = "Hello (world";
    EXPECT_EQ(default_processor->process(input), "Hello");
}

TEST_F(TextProcessorTest, SpecialCharacters) {
    std::string input = "Hello [世界] world";
    EXPECT_EQ(default_processor->process(input), "Hello  world");

    input = "Test (émoji 😀) here";
    EXPECT_EQ(default_processor->process(input), "Test  here");
}

TEST_F(TextProcessorTest, PerformanceLargeText) {
    // Test with large text
    std::string large_text;
    for (int i = 0; i < 1000; ++i) {
        large_text += "[noise] Hello (um) world ";
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::string result = default_processor->process(large_text);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should process large text quickly (< 100ms)
    EXPECT_LT(duration.count(), 100);
    EXPECT_FALSE(result.empty());
}

TEST_F(TextProcessorTest, ThreadSafety) {
    const int num_threads = 10;
    const int iterations = 100;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, &success_count, iterations]() {
            for (int j = 0; j < iterations; ++j) {
                std::string input = "[TEST] Hello (world) " + std::to_string(j);
                std::string result = default_processor->process(input);
                if (!result.empty()) {
                    success_count++;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count, num_threads * iterations);
}
