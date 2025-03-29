#ifndef HYNI_CONFIG_H
#define HYNI_CONFIG_H

#include <string>

namespace hyni
{

const std::string SYSTEM_CONTENT = "You are an AI assistant that helps users prepare for Amazon leadership principle \
                                    behavioral interviews by structuring their answers using the STAR method. For each \
                                    leadership principle, provide a structured response: Situation, Task, Action, Result. \
                                    The user is a software engineer with 20+ years of experience in C++. He has worked \
                                    for Nokia, Here Technologies, Mercedes-Benz, SAP, and FrameCAD. He has also led a team. \
                                    Please provide a great example using the STAR method.";

// This could be for other system configurations (e.g., API key, model type)
const std::string API_KEY = "sk-e721f7e4fe834b969e1ea5113f0e3d3f";
const std::string GPT_MODEL_TYPE = "gpt-3.5-turbo";
const std::string DS_GENERAL_MODEL_TYPE = "deepseek-chat";
const std::string DS_CODING_MODEL_TYPE = "deepseek-coder";
const std::string GPT_API_URL = "https://api.openai.com/v1/chat/completions";
const std::string DS_API_URL = "https://api.deepseek.com/v1/chat/completions";

} // namespace hyni

#endif // HYNI_CONFIG_H
