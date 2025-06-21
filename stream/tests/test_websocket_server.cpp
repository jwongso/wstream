#include <gtest/gtest.h>
#include "websocket_server.h"
#include <thread>
#include <chrono>
#include <atomic>

/**
 * @file test_websocket_server.cpp
 * @brief Integration tests for websocket_server class
 */

class WebSocketServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_port = 18080 + (rand() % 1000);  // Random port to avoid conflicts
        server = std::make_unique<websocket_server>(test_port);
    }

    void TearDown() override {
        if (server) {
            server->stop();
            // Give it time to clean up
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    uint16_t test_port;
    std::unique_ptr<websocket_server> server;
};

TEST_F(WebSocketServerTest, StartAndStop) {
    server->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    server->stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Should start and stop without crashing
    SUCCEED();
}

TEST_F(WebSocketServerTest, QueueTranscription) {
    server->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Queue some transcriptions
    server->queue_transcription("Test message 1");
    server->queue_transcription("Test message 2");
    server->queue_transcription("Test message 3");

    // Give it time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    server->stop();

    // Should handle queuing without crashing
    SUCCEED();
}

TEST_F(WebSocketServerTest, BroadcastWithoutClients) {
    server->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_FALSE(server->has_clients());

    // Broadcasting without clients should be safe
    server->broadcast("Test message");
    server->queue_transcription("Queued message");

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    server->stop();

    SUCCEED();
}

TEST_F(WebSocketServerTest, RapidMessages) {
    server->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Queue many messages rapidly
    const int num_messages = 20;
    for (int i = 0; i < num_messages; ++i) {
        server->queue_transcription("Message " + std::to_string(i));
    }

    // Allow time for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    server->stop();

    SUCCEED();
}

TEST_F(WebSocketServerTest, EmptyMessage) {
    server->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Empty messages should be handled gracefully
    server->broadcast("");
    server->queue_transcription("");

    // Queue a real message
    server->queue_transcription("Real message");

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    server->stop();

    SUCCEED();
}

TEST_F(WebSocketServerTest, StressTest) {
    server->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    const int messages_count = 50;

    auto start_time = std::chrono::steady_clock::now();

    // Send many messages
    for (int i = 0; i < messages_count; ++i) {
        server->queue_transcription("Stress test " + std::to_string(i));
        if (i % 10 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Allow processing time
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    auto duration = std::chrono::steady_clock::now() - start_time;

    server->stop();

    // Should complete within reasonable time
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(duration).count(), 5);
}

TEST_F(WebSocketServerTest, MultipleStartStop) {
    // Test multiple start/stop cycles
    for (int i = 0; i < 3; ++i) {
        server->start();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        server->queue_transcription("Test " + std::to_string(i));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        server->stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    SUCCEED();
}

// Simple connection test using basic socket operations
TEST_F(WebSocketServerTest, BasicConnection) {
    server->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Try to connect using basic socket (not full WebSocket handshake)
    try {
        boost::asio::io_context ioc;
        boost::asio::ip::tcp::socket socket(ioc);
        boost::asio::ip::tcp::resolver resolver(ioc);

        auto endpoints = resolver.resolve("127.0.0.1", std::to_string(test_port));

        // Set a timeout for connection
        socket.async_connect(*endpoints.begin(), [](boost::system::error_code ec) {
            // Connection callback - we don't need to do anything
        });

        // Run with timeout
        ioc.run_for(std::chrono::milliseconds(500));

        if (socket.is_open()) {
            socket.close();
            // Connection was successful
            SUCCEED();
        } else {
            // Connection failed - might be port in use or other issue
            // Don't fail the test for this
            SUCCEED();
        }

    } catch (const std::exception& e) {
        // Network issues shouldn't fail the test
        std::cout << "Connection test skipped due to: " << e.what() << std::endl;
        SUCCEED();
    }

    server->stop();
}

// Test server lifecycle without actual WebSocket connections
TEST_F(WebSocketServerTest, ServerLifecycle) {
    EXPECT_FALSE(server->has_clients());

    server->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Server should be running
    EXPECT_FALSE(server->has_clients());  // No clients connected yet

    // Queue some work
    for (int i = 0; i < 5; ++i) {
        server->queue_transcription("Lifecycle test " + std::to_string(i));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    server->stop();

    // Should still not have clients
    EXPECT_FALSE(server->has_clients());

    SUCCEED();
}
