#ifndef CHATAPI_H
#define CHATAPI_H

#include <string>
#include <iostream>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

namespace hyni
{

class ChatAPI {
public:
    enum class APIProvider {
        OpenAI,
        DeepSeek,
        Unknown
    };

    enum class QuestionType {
        AmazonBehavioral,  // STAR-based Amazon behavioral question
        General            // LeetCode-style coding or system design question
    };

    ChatAPI(const std::string& url);

    std::string sendMessage(const std::string& message,
                            QuestionType type = ChatAPI::QuestionType::General,
                            int max_tokens = 1200,
                            double temperature = 0.7);

    std::string getAssistantReply(const std::string& jsonResponse);

private:
    std::string api_key;
    std::string api_url;
    std::string model;
    std::string role;
    APIProvider api_provider;

    // Callback function to handle HTTP response data
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
        size_t newLength = size * nmemb;
        try {
            s->reserve(s->size() + newLength);
            s->append(static_cast<char*>(contents), newLength);
        } catch (const std::bad_alloc& e) {
            std::cerr << "Memory allocation error in WriteCallback: " << e.what() << std::endl;
            return CURL_WRITEFUNC_PAUSE;  // Instead of 0, pause and avoid corrupting the response
        }
        return newLength;
    }

    APIProvider detectAPIProvider(const std::string& url);
};
}

#endif
