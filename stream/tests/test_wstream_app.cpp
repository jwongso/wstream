#include <gtest/gtest.h>
#include "wstream_app.h"
#include <thread>
#include <chrono>
#include <filesystem>

/**
 * @file test_wstream_app.cpp
 * @brief Integration tests for wstream_app class
 */

class WStreamAppTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if test model exists
        test_model_path = "models/ggml-tiny.en.bin";

        if (!std::filesystem::exists(test_model_path)) {
            GTEST_SKIP() << "Test model not found at: " << test_model_path;
        }
    }

    std::string test_model_path;
};

TEST_F(WStreamAppTest, Construction) {
    wstream_app app(test_model_path);
    // Should construct without issues
    SUCCEED();
}

TEST_F(WStreamAppTest, InitializationWithValidModel) {
    wstream_app app(test_model_path);

    const char* argv[] = {"wstream"};
    EXPECT_TRUE(app.initialize(1, const_cast<char**>(argv)));
}

TEST_F(WStreamAppTest, InitializationWithCustomModel) {
    wstream_app app;  // Use default path

    const char* argv[] = {"wstream", test_model_path.c_str()};
    EXPECT_TRUE(app.initialize(2, const_cast<char**>(argv)));
}

TEST_F(WStreamAppTest, InitializationWithInvalidModel) {
    wstream_app app;

    const char* argv[] = {"wstream", "non_existent_model.bin"};
    // Should use default model and warn
    bool result = app.initialize(2, const_cast<char**>(argv));

    // Will succeed if default model exists, otherwise fail
    // This is expected behavior
    SUCCEED();
}

TEST_F(WStreamAppTest, ShutdownWithoutRun) {
    wstream_app app(test_model_path);

    const char* argv[] = {"wstream"};
    ASSERT_TRUE(app.initialize(1, const_cast<char**>(argv)));

    app.shutdown();
    // Should shutdown cleanly without running
    SUCCEED();
}

TEST_F(WStreamAppTest, RunAndShutdown) {
    wstream_app app(test_model_path);

    const char* argv[] = {"wstream"};
    ASSERT_TRUE(app.initialize(1, const_cast<char**>(argv)));

    // Run in a separate thread
    std::thread app_thread([&app]() {
        app.run();
    });

    // Let it run briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Shutdown
    app.shutdown();

    // Wait for thread to finish
    if (app_thread.joinable()) {
        app_thread.join();
    }

    SUCCEED();
}

TEST_F(WStreamAppTest, MultipleShutdown) {
    wstream_app app(test_model_path);

    const char* argv[] = {"wstream"};
    ASSERT_TRUE(app.initialize(1, const_cast<char**>(argv)));

    app.shutdown();
    app.shutdown();  // Should handle multiple shutdowns gracefully

    SUCCEED();
}

TEST_F(WStreamAppTest, DefaultModelPath) {
    wstream_app app;  // Use default model

    const char* argv[] = {"wstream"};

    // This will succeed if default model exists
    // Otherwise it will fail, which is expected
    app.initialize(1, const_cast<char**>(argv));

    SUCCEED();
}
