#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "websocket_server.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <string>

using json = nlohmann::json;

class WebSocketServerCommandTest : public ::testing::Test {
protected:
    void SetUp() override {
        server = std::make_unique<websocket_server>();
        command_responses.clear();
        audio_data_received.clear();
    }

    void TearDown() override {
        if (server) {
            server->stop();
        }
        server.reset();
    }

    // Helper to capture command responses
    void setupCommandHandler() {
        server->set_command_handler(
            [this](const std::string& command, const json& params, websocket::stream<tcp::socket>* client_ws) -> json {
                json response;
                response["type"] = "response";
                response["action"] = command;

                if (command == "set_audio_source") {
                    if (!params.contains("source") || !params["source"].is_string()) {
                        response["status"] = "error";
                        response["message"] = "Missing or invalid 'source' parameter";
                    } else {
                        std::string source = params["source"].get<std::string>();
                        if (source == "websocket" || source == "microphone" || source == "auto") {
                            response["status"] = "success";
                            response["source"] = source;
                            response["message"] = "Audio source switched to " + source;
                        } else {
                            response["status"] = "error";
                            response["message"] = "Invalid audio source: " + source;
                        }
                    }
                } else if (command == "get_status") {
                    response["status"] = "success";
                    response["current_source"] = "auto";
                    response["client_count"] = 0;
                } else {
                    response["status"] = "error";
                    response["message"] = "Unknown command: " + command;
                }

                command_responses.push_back(response);
                return response;
            }
            );
    }

    // Helper to capture audio data
    void setupAudioHandler() {
        server->set_audio_callback(
            [this](const audio_data& audio, websocket::stream<tcp::socket>* client_ws) {
                audio_data_received.push_back(audio);
            }
            );
    }

    std::unique_ptr<websocket_server> server;
    std::vector<json> command_responses;
    std::vector<audio_data> audio_data_received;
};

TEST_F(WebSocketServerCommandTest, CommandHandlerCanBeSet) {
    setupCommandHandler();
    // If we get here without crashing, the command handler was set successfully
    SUCCEED();
}

TEST_F(WebSocketServerCommandTest, AudioCallbackCanBeSet) {
    setupAudioHandler();
    // If we get here without crashing, the audio callback was set successfully
    SUCCEED();
}

TEST_F(WebSocketServerCommandTest, BroadcastWithoutClients) {
    setupCommandHandler();

    // This should not crash even with no clients
    server->broadcast("Test message");
    SUCCEED();
}

TEST_F(WebSocketServerCommandTest, QueueTranscriptionWithoutClients) {
    setupCommandHandler();

    // This should not crash even with no clients
    server->queue_transcription("Test transcription");
    SUCCEED();
}

TEST_F(WebSocketServerCommandTest, HasClientsReturnsFalseInitially) {
    EXPECT_FALSE(server->has_clients());
}

TEST_F(WebSocketServerCommandTest, GetClientCountReturnsZeroInitially) {
    EXPECT_EQ(server->get_client_count(), 0);
}
