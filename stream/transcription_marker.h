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

// Add this at the top of wstream_app.cpp
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>

class transcription_marker {
private:
    std::vector<std::string> reference_words;
    size_t current_ref_index = 0;

    // ANSI color codes
    const std::string RED = "\033[91m";      // Error words
    const std::string GREEN = "\033[92m";    // Correct words (optional)
    const std::string YELLOW = "\033[93m";   // Warning/uncertain
    const std::string RESET = "\033[0m";     // Reset to default
    const std::string BOLD = "\033[1m";      // Bold text

public:
    void load_reference(const std::string& text) {
        reference_words.clear();
        std::istringstream stream(text);
        std::string word;
        while (stream >> word) {
            // Store normalized version for comparison
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            reference_words.push_back(word);
        }
        current_ref_index = 0;
        std::cout << "[Benchmark] Loaded " << reference_words.size() << " reference words" << std::endl;
    }

    std::string mark_differences(const std::string& hypothesis) {
        if (reference_words.empty()) return hypothesis;

        // Tokenize hypothesis
        std::vector<std::string> hyp_words;
        std::istringstream stream(hypothesis);
        std::string word;
        while (stream >> word) {
            hyp_words.push_back(word);
        }

        // Build marked output
        std::stringstream result;

        for (size_t i = 0; i < hyp_words.size(); ++i) {
            if (i > 0) result << " ";

            // Normalize hypothesis word for comparison
            std::string hyp_norm = hyp_words[i];
            std::transform(hyp_norm.begin(), hyp_norm.end(), hyp_norm.begin(), ::tolower);

            // Remove punctuation for comparison
            std::string hyp_clean = hyp_norm;
            hyp_clean.erase(std::remove_if(hyp_clean.begin(), hyp_clean.end(),
                                           [](char c) { return !std::isalnum(c); }),
                            hyp_clean.end());

            // Check current position and up to 3 words ahead
            bool found = false;
            const size_t LOOK_AHEAD = 3;

            for (size_t j = 0; j <= LOOK_AHEAD && (current_ref_index + j) < reference_words.size(); ++j) {
                std::string ref_clean = reference_words[current_ref_index + j];
                ref_clean.erase(std::remove_if(ref_clean.begin(), ref_clean.end(),
                                               [](char c) { return !std::isalnum(c); }),
                                ref_clean.end());

                if (hyp_clean == ref_clean) {
                    // Found match!
                    if (j == 0) {
                        // Perfect position match - normal text
                        result << hyp_words[i];
                    } else {
                        // Found but with skipped words - show in yellow
                        result << YELLOW << hyp_words[i] << RESET;
                    }
                    current_ref_index += j + 1;  // Skip to position after matched word
                    found = true;
                    break;
                }
            }

            if (!found) {
                // No match within look-ahead window - mark as error in red
                result << RED << BOLD << hyp_words[i] << RESET;
            }
        }

        return result.str();
    }

    // Alternative marking style with brackets AND colors
    std::string mark_differences_with_brackets(const std::string& hypothesis) {
        if (reference_words.empty()) return hypothesis;

        // Tokenize hypothesis
        std::vector<std::string> hyp_words;
        std::istringstream stream(hypothesis);
        std::string word;
        while (stream >> word) {
            hyp_words.push_back(word);
        }

        // Build marked output
        std::stringstream result;

        for (size_t i = 0; i < hyp_words.size(); ++i) {
            if (i > 0) result << " ";

            // Normalize hypothesis word for comparison
            std::string hyp_norm = hyp_words[i];
            std::transform(hyp_norm.begin(), hyp_norm.end(), hyp_norm.begin(), ::tolower);

            // Remove punctuation for comparison
            std::string hyp_clean = hyp_norm;
            hyp_clean.erase(std::remove_if(hyp_clean.begin(), hyp_clean.end(),
                                           [](char c) { return !std::isalnum(c); }),
                            hyp_clean.end());

            // Check current position and up to 3 words ahead
            bool found = false;
            const size_t LOOK_AHEAD = 3;

            for (size_t j = 0; j <= LOOK_AHEAD && (current_ref_index + j) < reference_words.size(); ++j) {
                std::string ref_clean = reference_words[current_ref_index + j];
                ref_clean.erase(std::remove_if(ref_clean.begin(), ref_clean.end(),
                                               [](char c) { return !std::isalnum(c); }),
                                ref_clean.end());

                if (hyp_clean == ref_clean) {
                    // Found match!
                    result << hyp_words[i];  // Normal text
                    current_ref_index += j + 1;
                    found = true;
                    break;
                }
            }

            if (!found) {
                // Error - show in red with brackets
                result << RED << "[" << hyp_words[i] << "]" << RESET;
            }
        }

        return result.str();
    }

    // Reset to beginning of reference
    void reset() {
        current_ref_index = 0;
    }

    // Get current position (for debugging)
    size_t get_position() const {
        return current_ref_index;
    }

    // Check if color output is supported
    static bool is_color_supported() {
#ifdef _WIN32
        return false;  // Windows console needs special handling
#else
        const char* term = std::getenv("TERM");
        return term && std::string(term) != "dumb";
#endif
    }

    // Disable colors if needed
    void disable_colors() {
        const_cast<std::string&>(RED) = "";
        const_cast<std::string&>(GREEN) = "";
        const_cast<std::string&>(YELLOW) = "";
        const_cast<std::string&>(RESET) = "";
        const_cast<std::string&>(BOLD) = "";
    }
};
