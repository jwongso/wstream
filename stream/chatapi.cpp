#include "chatapi.h"
#include "config.h"

using json = nlohmann::json;
using namespace hyni;

ChatAPI::APIProvider ChatAPI::detectAPIProvider(const std::string& url) {
    if (url.find("openai.com") != std::string::npos) {
        return APIProvider::OpenAI;
    } else if (url.find("deepseek.com") != std::string::npos) {
        return APIProvider::DeepSeek;
    } else {
        return APIProvider::Unknown;
    }
}

ChatAPI::ChatAPI(const std::string& url)
    : api_url(url) {
    api_provider = detectAPIProvider(url);

    api_key = API_KEY;
    role = "user";

    if (api_provider == APIProvider::OpenAI) {
        model = GPT_MODEL_TYPE;
    } else if (api_provider == APIProvider::DeepSeek) {
        model = DS_CODING_MODEL_TYPE;
    }
}

// Method to send a message to the API
std::string ChatAPI::sendMessage(const std::string& message,
                                 QuestionType type,
                                 int max_tokens,
                                 float temperature) {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if (curl) {
        json payload;

        if (api_provider == APIProvider::DeepSeek) {
            if (type == QuestionType::AmazonBehavioral) {
                payload["model"] = DS_GENERAL_MODEL_TYPE;
            } else {
                payload["model"] = DS_CODING_MODEL_TYPE;
            }
        }
        else if (api_provider == APIProvider::OpenAI) {
            payload["model"] = GPT_MODEL_TYPE;
        }

        payload["max_tokens"] = max_tokens;

        // Set temperature defaults
        if (temperature == -1.0f) {
            payload["temperature"] = 0.7f;
        } else {
            payload["temperature"] = temperature;
        }

        // OpenAI-specific settings
        if (api_provider == APIProvider::OpenAI) {
            payload["top_p"] = 1.0f;
        } else if (api_provider == APIProvider::DeepSeek) {
            payload["stream"] = false;
        }

        // Construct messages based on question type
        json messages;

        if (type == QuestionType::AmazonBehavioral) {
            if (api_provider == APIProvider::OpenAI) {
                // OpenAI: Use system role
                messages.push_back({{"role", "system"}, {"content", SYSTEM_CONTENT}});
                messages.push_back({{"role", "user"}, {"content", message}});
            } else {
                // DeepSeek: Merge system + user message
                std::string deepseek_prompt =
                    "[INSTRUCTIONS]: " + SYSTEM_CONTENT + "\n\n[QUESTION]: " + message;
                messages.push_back({{"role", "user"}, {"content", deepseek_prompt}});
            }
        } else {
            // Non-behavioral questions (coding/system design)
            messages.push_back({{"role", "user"}, {"content", message}});
        }

        payload["messages"] = messages;

        // Convert payload to a string
        std::string payloadStr = payload.dump();

        std::cout << payload.dump(4) << std::endl;

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, ("Authorization: Bearer " + api_key).c_str());

        curl_easy_setopt(curl, CURLOPT_URL, api_url.c_str());
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);          // Total request timeout (60s)
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payloadStr.c_str()); // Pass the string directly
        curl_easy_setopt(curl, CURLOPT_POST, true);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);  // Disable peer certificate checks
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);  // Disable hostname checks
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);  // Enable TCP keep-alive
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE, 30L);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    return readBuffer;
}

// Method to parse the API response and extract the assistant's reply
std::string ChatAPI::getAssistantReply(const std::string& jsonResponse) {
    std::cout << jsonResponse << std::endl;
    try {
        json responseJson = json::parse(jsonResponse);

        // Check if the response contains an error
        if (responseJson.contains("error")) {
            std::string errorMessage = responseJson["error"]["message"].get<std::string>();
            std::cerr << "API Error: " << errorMessage << std::endl;
            return "";
        }

        // Extract the assistant's reply
        return responseJson["choices"][0]["message"]["content"].get<std::string>();
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON response: " << e.what() << std::endl;
        return "";
    }
}
