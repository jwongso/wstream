// -------------------------------------------------------------------------------------------------
//
// Copyright (C) all of the contributors. All rights reserved.
//
// This software, including documentation, is protected by copyright controlled by
// contributors. All rights are reserved. Copying, including reproducing, storing,
// adapting or translating, any or all of this material requires the prior written
// consent of all contributors.
//
// -------------------------------------------------------------------------------------------------

#include "text_processor.h"
#include <algorithm>
#include <cctype>

std::string text_processor::process(std::string_view input) const {
    std::string result(input);

    if (m_config.remove_brackets || m_config.remove_parentheses) {
        remove_bracketed_text(result);
    }

    if (m_config.trim_whitespace) {
        trim_whitespace(result);
    }

    return result;
}

void text_processor::remove_bracketed_text(std::string& text) const {
    std::string result;
    result.reserve(text.size());

    bool in_bracket = false;
    bool in_paren = false;

    for (char c : text) {
        if (!in_bracket && !in_paren) {
            if (c == '[' && m_config.remove_brackets) {
                in_bracket = true;
                continue;
            }
            if (c == '(' && m_config.remove_parentheses) {
                in_paren = true;
                continue;
            }
            result.push_back(c);
        } else {
            if (in_bracket && c == ']') {
                in_bracket = false;
                continue;
            }
            if (in_paren && c == ')') {
                in_paren = false;
                continue;
            }
        }
    }

    text = std::move(result);
}

void text_processor::trim_whitespace(std::string& text) const {
    // Trim leading whitespace
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](unsigned char ch) {
                   return !std::isspace(ch);
               }));

    // Trim trailing whitespace
    text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char ch) {
                   return !std::isspace(ch);
               }).base(), text.end());
}
