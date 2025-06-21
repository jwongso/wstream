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

    int bracket_depth = 0;
    int paren_depth = 0;

    for (char c : text) {
        if (m_config.remove_brackets && c == '[') {
            bracket_depth++;
            continue;
        }
        if (m_config.remove_brackets && c == ']') {
            if (bracket_depth > 0) {
                bracket_depth--;
            } else {
                // Unmatched closing bracket - keep it
                result.push_back(c);
            }
            continue;
        }
        if (m_config.remove_parentheses && c == '(') {
            paren_depth++;
            continue;
        }
        if (m_config.remove_parentheses && c == ')') {
            if (paren_depth > 0) {
                paren_depth--;
            } else {
                // Unmatched closing parenthesis - keep it
                result.push_back(c);
            }
            continue;
        }

        // Only add character if we're not inside brackets or parentheses
        if (bracket_depth == 0 && paren_depth == 0) {
            result.push_back(c);
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
