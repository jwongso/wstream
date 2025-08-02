#pragma once

#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <unordered_map>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstring>

class transcription_marker {
private:
    std::vector<std::string> m_reference_words;
    std::vector<std::string> m_reference_words_normalized;

    // Configuration options
    struct Config {
        bool enable_logging = true;
        bool strict_matching = false;  // If true, don't normalize
        bool show_confidence = true;
        int context_window = 3;  // Words to show around errors
        double fuzzy_match_threshold = 0.8;  // Similarity threshold for fuzzy matching
        bool enable_fuzzy_matching = true;
        int max_search_distance = 10;  // Max distance to search for matches
    } m_config;

    // Streaming state
    size_t m_current_position = 0;  // Track where we are in the reference
    bool m_streaming_mode = false;

    struct StreamingState {
        size_t last_matched_ref_position = 0;
        std::vector<bool> used_reference_words;
        int total_matches = 0;
        int total_errors = 0;
        int total_fuzzy = 0;
        int total_chunks = 0;
    } m_streaming_state;

    // ANSI color codes
    const std::string RED = "\033[91m";      // Error words
    const std::string GREEN = "\033[92m";    // Correct words
    const std::string YELLOW = "\033[93m";   // Substitution
    const std::string CYAN = "\033[96m";     // Insertion
    const std::string MAGENTA = "\033[95m";  // Fuzzy match
    const std::string RESET = "\033[0m";     // Reset to default
    const std::string BOLD = "\033[1m";      // Bold text
    const std::string DIM = "\033[2m";       // Dim text
    const std::string UNDERLINE = "\033[4m"; // Underline

    // Alignment operations
    enum class AlignOp {
        MATCH,          // Words match exactly
        FUZZY_MATCH,    // Words are similar
        SUBSTITUTE,     // Different word (substitution error)
        INSERT,         // Extra word in hypothesis (insertion error)
        DELETE,         // Missing word from reference (deletion error)
        REORDER         // Word appears but in wrong position
    };

    struct AlignmentResult {
        std::vector<AlignOp> operations;
        std::vector<int> ref_indices;  // -1 for insertions
        std::vector<int> hyp_indices;  // -1 for deletions
        std::vector<double> confidence_scores;  // Confidence for each alignment
        int distance;
        double overall_confidence;

        // Additional tracking
        std::vector<std::pair<int, int>> reordered_pairs;  // Track reordered words
        std::unordered_map<int, std::string> error_context;  // Context around errors
    };

    // Logging
    void log(const std::string& message, const std::string& level = "INFO") const {
        if (!m_config.enable_logging) return;

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        std::cout << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S")
                  << "] [" << level << "] " << message << std::endl;
    }

    // Calculate string similarity (0.0 to 1.0)
    double calculate_similarity(const std::string& s1, const std::string& s2) const {
        if (s1 == s2) return 1.0;
        if (s1.empty() || s2.empty()) return 0.0;

        // Use Jaro-Winkler similarity
        return jaro_winkler_similarity(s1, s2);
    }

    // Jaro-Winkler similarity implementation
    double jaro_winkler_similarity(const std::string& s1, const std::string& s2) const {
        if (s1 == s2) return 1.0;

        int len1 = s1.length();
        int len2 = s2.length();

        if (len1 == 0 || len2 == 0) return 0.0;

        int match_window = std::max(len1, len2) / 2 - 1;
        if (match_window < 1) match_window = 1;

        std::vector<bool> s1_matches(len1, false);
        std::vector<bool> s2_matches(len2, false);

        int matches = 0;
        int transpositions = 0;

        // Find matches
        for (int i = 0; i < len1; i++) {
            int start = std::max(0, i - match_window);
            int end = std::min(i + match_window + 1, len2);

            for (int j = start; j < end; j++) {
                if (s2_matches[j] || s1[i] != s2[j]) continue;
                s1_matches[i] = true;
                s2_matches[j] = true;
                matches++;
                break;
            }
        }

        if (matches == 0) return 0.0;

        // Count transpositions
        int k = 0;
        for (int i = 0; i < len1; i++) {
            if (!s1_matches[i]) continue;
            while (!s2_matches[k]) k++;
            if (s1[i] != s2[k]) transpositions++;
            k++;
        }

        double jaro = (matches / (double)len1 +
                       matches / (double)len2 +
                       (matches - transpositions / 2.0) / matches) / 3.0;

        // Jaro-Winkler modification
        int prefix_len = 0;
        for (int i = 0; i < std::min(len1, len2) && i < 4; i++) {
            if (s1[i] == s2[i]) prefix_len++;
            else break;
        }

        return jaro + prefix_len * 0.1 * (1.0 - jaro);
    }

    // Normalize word for comparison
    std::string normalize_word(const std::string& word) const {
        if (m_config.strict_matching) return word;

        std::string normalized = word;
        // Convert to lowercase
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
        // Remove punctuation but keep apostrophes for contractions
        normalized.erase(std::remove_if(normalized.begin(), normalized.end(),
                                        [](char c) { return !std::isalnum(c) && c != '\''; }),
                         normalized.end());
        return normalized;
    }

    // Find best matching word within a search window
    std::pair<int, double> find_best_match(const std::string& word,
                                           int start_pos,
                                           const std::vector<bool>& used_refs) const {
        std::string norm_word = normalize_word(word);
        int best_idx = -1;
        double best_score = 0.0;

        int search_start = std::max(0, start_pos - m_config.max_search_distance);
        int search_end = std::min((int)m_reference_words_normalized.size(),
                                  start_pos + m_config.max_search_distance);

        for (int i = search_start; i < search_end; i++) {
            if (used_refs[i]) continue;

            double score = calculate_similarity(norm_word, m_reference_words_normalized[i]);
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        return {best_idx, best_score};
    }

    // Get context around an error
    std::string get_error_context(int pos, const std::vector<std::string>& words) const {
        std::stringstream context;
        int start = std::max(0, pos - m_config.context_window);
        int end = std::min((int)words.size(), pos + m_config.context_window + 1);

        for (int i = start; i < end; i++) {
            if (i == pos) context << ">>>";
            context << words[i];
            if (i == pos) context << "<<<";
            if (i < end - 1) context << " ";
        }

        return context.str();
    }

    // Original alignment method for non-streaming mode
    AlignmentResult align_sequences(const std::vector<std::string>& hyp_words) const {
        const int n = m_reference_words_normalized.size();
        const int m = hyp_words.size();

        // DP table for edit distance
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

        // Initialize base cases
        for (int i = 0; i <= n; i++) dp[i][0] = i;
        for (int j = 0; j <= m; j++) dp[0][j] = j;

        // Fill DP table
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                std::string hyp_norm = normalize_word(hyp_words[j-1]);

                if (m_reference_words_normalized[i-1] == hyp_norm) {
                    dp[i][j] = dp[i-1][j-1]; // Match
                } else {
                    dp[i][j] = 1 + std::min({
                                   dp[i-1][j],     // Deletion
                                   dp[i][j-1],     // Insertion
                                   dp[i-1][j-1]    // Substitution
                               });
                }
            }
        }

        // Backtrack to find alignment
        AlignmentResult result;
        result.distance = dp[n][m];

        int i = n, j = m;
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0) {
                std::string hyp_norm = normalize_word(hyp_words[j-1]);

                if (m_reference_words_normalized[i-1] == hyp_norm) {
                    // Match
                    result.operations.push_back(AlignOp::MATCH);
                    result.ref_indices.push_back(i-1);
                    result.hyp_indices.push_back(j-1);
                    result.confidence_scores.push_back(1.0);
                    i--; j--;
                } else {
                    // Find which operation was used
                    int sub_cost = dp[i-1][j-1];
                    int del_cost = dp[i-1][j];
                    int ins_cost = dp[i][j-1];

                    if (sub_cost <= del_cost && sub_cost <= ins_cost) {
                        // Check if it's a fuzzy match
                        double sim = calculate_similarity(hyp_norm, m_reference_words_normalized[i-1]);
                        if (sim >= m_config.fuzzy_match_threshold && m_config.enable_fuzzy_matching) {
                            result.operations.push_back(AlignOp::FUZZY_MATCH);
                            result.confidence_scores.push_back(sim);
                        } else {
                            result.operations.push_back(AlignOp::SUBSTITUTE);
                            result.confidence_scores.push_back(0.0);
                        }
                        result.ref_indices.push_back(i-1);
                        result.hyp_indices.push_back(j-1);
                        i--; j--;
                    } else if (del_cost <= ins_cost) {
                        // Deletion
                        result.operations.push_back(AlignOp::DELETE);
                        result.ref_indices.push_back(i-1);
                        result.hyp_indices.push_back(-1);
                        result.confidence_scores.push_back(0.0);
                        i--;
                    } else {
                        // Insertion
                        result.operations.push_back(AlignOp::INSERT);
                        result.ref_indices.push_back(-1);
                        result.hyp_indices.push_back(j-1);
                        result.confidence_scores.push_back(0.0);
                        j--;
                    }
                }
            } else if (i > 0) {
                // Deletion
                result.operations.push_back(AlignOp::DELETE);
                result.ref_indices.push_back(i-1);
                result.hyp_indices.push_back(-1);
                result.confidence_scores.push_back(0.0);
                i--;
            } else {
                // Insertion
                result.operations.push_back(AlignOp::INSERT);
                result.ref_indices.push_back(-1);
                result.hyp_indices.push_back(j-1);
                result.confidence_scores.push_back(0.0);
                j--;
            }
        }

        // Reverse to get correct order
        std::reverse(result.operations.begin(), result.operations.end());
        std::reverse(result.ref_indices.begin(), result.ref_indices.end());
        std::reverse(result.hyp_indices.begin(), result.hyp_indices.end());
        std::reverse(result.confidence_scores.begin(), result.confidence_scores.end());

        // Calculate overall confidence
        if (!result.confidence_scores.empty()) {
            double sum = 0.0;
            for (double score : result.confidence_scores) {
                sum += score;
            }
            result.overall_confidence = sum / result.confidence_scores.size();
        }

        return result;
    }

    // Specialized alignment for streaming chunks
    AlignmentResult align_streaming_chunk(const std::vector<std::string>& hyp_words) {
        AlignmentResult result;

        // Start searching from last matched position
        size_t search_start = m_streaming_state.last_matched_ref_position;

        log("Starting chunk alignment from reference position " + std::to_string(search_start), "DEBUG");

        // First, try to find where this chunk starts in the reference
        size_t best_start_pos = search_start;
        int best_start_score = -1;

        // Look for the best starting position within a reasonable window
        size_t max_search = std::min(m_reference_words.size(),
                                     search_start + 50); // Increased search window

        for (size_t start_pos = search_start; start_pos < max_search; start_pos++) {
            int score = 0;
            size_t matches = 0;

            // Try to match first few words to find alignment
            for (size_t i = 0; i < std::min(size_t(3), hyp_words.size()); i++) {
                if (start_pos + i < m_reference_words_normalized.size()) {
                    std::string hyp_norm = normalize_word(hyp_words[i]);
                    if (m_reference_words_normalized[start_pos + i] == hyp_norm) {
                        score += 10;  // Exact match
                        matches++;
                    } else {
                        double sim = calculate_similarity(hyp_norm,
                                                          m_reference_words_normalized[start_pos + i]);
                        score += int(sim * 5);  // Partial credit for similar words
                    }
                }
            }

            // Bonus for consecutive position
            if (start_pos == search_start) {
                score += 5;
            }

            if (score > best_start_score) {
                best_start_score = score;
                best_start_pos = start_pos;
            }

            // If we found perfect matches for all checked words, stop searching
            if (matches == std::min(size_t(3), hyp_words.size())) {
                break;
            }
        }

        // IMPORTANT: Check if we skipped any reference words
        if (best_start_pos > search_start) {
            log("Gap detected! Skipped from position " + std::to_string(search_start) +
                    " to " + std::to_string(best_start_pos), "WARN");

            // Mark all skipped words as deletions
            for (size_t skip_pos = search_start; skip_pos < best_start_pos; skip_pos++) {
                result.operations.push_back(AlignOp::DELETE);
                result.ref_indices.push_back(skip_pos);
                result.hyp_indices.push_back(-1);
                result.confidence_scores.push_back(0.0);
                m_streaming_state.used_reference_words[skip_pos] = true;

                log("Marking as deleted: '" + m_reference_words[skip_pos] +
                        "' at position " + std::to_string(skip_pos), "DEBUG");
            }
        }

        log("Best alignment found at position " + std::to_string(best_start_pos) +
                " with score " + std::to_string(best_start_score), "DEBUG");

        // Now align from the best starting position
        size_t ref_pos = best_start_pos;

        for (size_t hyp_pos = 0; hyp_pos < hyp_words.size(); hyp_pos++) {
            if (ref_pos >= m_reference_words.size()) {
                // We've run out of reference words - rest are insertions
                result.operations.push_back(AlignOp::INSERT);
                result.ref_indices.push_back(-1);
                result.hyp_indices.push_back(hyp_pos);
                result.confidence_scores.push_back(0.0);
                continue;
            }

            std::string hyp_norm = normalize_word(hyp_words[hyp_pos]);

            // Check for exact match at current position
            if (m_reference_words_normalized[ref_pos] == hyp_norm) {
                result.operations.push_back(AlignOp::MATCH);
                result.ref_indices.push_back(ref_pos);
                result.hyp_indices.push_back(hyp_pos);
                result.confidence_scores.push_back(1.0);
                m_streaming_state.used_reference_words[ref_pos] = true;
                ref_pos++;
            } else {
                // Look ahead for a match
                bool found = false;
                for (size_t look_ahead = 1; look_ahead <= 3 && ref_pos + look_ahead < m_reference_words.size(); look_ahead++) {
                    if (m_reference_words_normalized[ref_pos + look_ahead] == hyp_norm) {
                        // Mark skipped words as deletions
                        for (size_t skip = 0; skip < look_ahead; skip++) {
                            result.operations.push_back(AlignOp::DELETE);
                            result.ref_indices.push_back(ref_pos + skip);
                            result.hyp_indices.push_back(-1);
                            result.confidence_scores.push_back(0.0);
                            m_streaming_state.used_reference_words[ref_pos + skip] = true;
                        }

                        result.operations.push_back(AlignOp::MATCH);
                        result.ref_indices.push_back(ref_pos + look_ahead);
                        result.hyp_indices.push_back(hyp_pos);
                        result.confidence_scores.push_back(1.0);
                        m_streaming_state.used_reference_words[ref_pos + look_ahead] = true;
                        ref_pos = ref_pos + look_ahead + 1;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    // Check for fuzzy match
                    double best_sim = 0.0;
                    size_t best_idx = ref_pos;

                    for (size_t check = ref_pos; check < std::min(ref_pos + 3, m_reference_words.size()); check++) {
                        double sim = calculate_similarity(hyp_norm, m_reference_words_normalized[check]);
                        if (sim > best_sim) {
                            best_sim = sim;
                            best_idx = check;
                        }
                    }

                    if (best_sim >= m_config.fuzzy_match_threshold && m_config.enable_fuzzy_matching) {
                        // Mark any skipped words before the fuzzy match
                        for (size_t skip = ref_pos; skip < best_idx; skip++) {
                            result.operations.push_back(AlignOp::DELETE);
                            result.ref_indices.push_back(skip);
                            result.hyp_indices.push_back(-1);
                            result.confidence_scores.push_back(0.0);
                            m_streaming_state.used_reference_words[skip] = true;
                        }

                        result.operations.push_back(AlignOp::FUZZY_MATCH);
                        result.ref_indices.push_back(best_idx);
                        result.hyp_indices.push_back(hyp_pos);
                        result.confidence_scores.push_back(best_sim);
                        ref_pos = best_idx + 1;
                    } else {
                        // Substitution or insertion
                        result.operations.push_back(AlignOp::SUBSTITUTE);
                        result.ref_indices.push_back(ref_pos);
                        result.hyp_indices.push_back(hyp_pos);
                        result.confidence_scores.push_back(0.0);
                        ref_pos++;
                    }
                }
            }
        }

        // Update position for next chunk
        m_streaming_state.last_matched_ref_position = ref_pos;

        return result;
    }

    // Build output for streaming (only shows hypothesis words with their status)
    std::string build_streaming_output(const std::vector<std::string>& hyp_words,
                                       const AlignmentResult& alignment) {
        std::stringstream result;
        bool first_word = true;

        for (size_t i = 0; i < alignment.operations.size(); i++) {
            // Show deletions (missing words)
            if (alignment.operations[i] == AlignOp::DELETE) {
                if (!first_word) result << " ";
                result << RED << "[-" << m_reference_words[alignment.ref_indices[i]] << "]" << RESET;
                first_word = false;
                continue;
            }

            // Skip if no hypothesis index
            if (alignment.hyp_indices[i] == -1) continue;

            if (!first_word) result << " ";
            first_word = false;

            switch (alignment.operations[i]) {
            case AlignOp::MATCH:
                result << GREEN << hyp_words[alignment.hyp_indices[i]] << RESET;
                m_streaming_state.total_matches++;
                break;

            case AlignOp::FUZZY_MATCH:
                result << MAGENTA << hyp_words[alignment.hyp_indices[i]] << RESET;
                if (m_config.show_confidence) {
                    result << DIM << "[" << std::fixed << std::setprecision(2)
                    << alignment.confidence_scores[i] << "]" << RESET;
                }
                m_streaming_state.total_fuzzy++;
                break;

            case AlignOp::SUBSTITUTE:
            case AlignOp::INSERT:
                result << RED << hyp_words[alignment.hyp_indices[i]] << RESET;
                m_streaming_state.total_errors++;
                break;

            default:
                result << hyp_words[alignment.hyp_indices[i]];
                break;
            }
        }

        return result.str();
    }

    std::string get_expected_next_words(int count = 5) const {
        std::stringstream result;
        size_t start = m_streaming_state.last_matched_ref_position;
        size_t end = std::min(start + count, m_reference_words.size());

        result << "Next expected: ";
        for (size_t i = start; i < end; i++) {
            if (i > start) result << " ";
            result << "'" << m_reference_words[i] << "'";
        }
        if (end < m_reference_words.size()) {
            result << " ...";
        }

        return result.str();
    }

    std::string get_operation_description(const AlignmentResult& alignment,
                                          size_t idx,
                                          const std::vector<std::string>& hyp_words) const {
        std::stringstream desc;

        switch (alignment.operations[idx]) {
        case AlignOp::FUZZY_MATCH:
            desc << "  Fuzzy match: '" << hyp_words[alignment.hyp_indices[idx]]
                 << "' ~ '" << m_reference_words[alignment.ref_indices[idx]]
                 << "' (confidence: " << alignment.confidence_scores[idx] << ")";
            break;
        case AlignOp::SUBSTITUTE:
            desc << "  Substitution: '" << hyp_words[alignment.hyp_indices[idx]]
                 << "' instead of '" << m_reference_words[alignment.ref_indices[idx]] << "'";
            break;
        case AlignOp::INSERT:
            desc << "  Insertion: '" << hyp_words[alignment.hyp_indices[idx]] << "'";
            desc << " Context: " << get_error_context(alignment.hyp_indices[idx], hyp_words);
            break;
        case AlignOp::DELETE:
            desc << "  Deletion: '" << m_reference_words[alignment.ref_indices[idx]] << "'";
            desc << " Context: " << get_error_context(alignment.ref_indices[idx], m_reference_words);
            break;
        case AlignOp::REORDER:
            desc << "  Reordered: '" << hyp_words[alignment.hyp_indices[idx]]
                 << "' (found at wrong position)";
            break;
        default:
            break;
        }

        return desc.str();
    }

public:
    // Configuration methods
    void set_logging(bool enable) { m_config.enable_logging = enable; }
    void set_strict_matching(bool strict) { m_config.strict_matching = strict; }
    void set_fuzzy_matching(bool enable) { m_config.enable_fuzzy_matching = enable; }
    void set_fuzzy_threshold(double threshold) { m_config.fuzzy_match_threshold = threshold; }
    void set_search_distance(int distance) { m_config.max_search_distance = distance; }
    void set_show_confidence(bool show) { m_config.show_confidence = show; }

    // Enable/disable streaming mode
    void set_streaming_mode(bool enable) {
        m_streaming_mode = enable;
        if (enable) {
            reset_streaming_state();
        }
    }

    void reset_streaming_state() {
        m_streaming_state.last_matched_ref_position = 0;
        m_streaming_state.used_reference_words.clear();
        m_streaming_state.used_reference_words.resize(m_reference_words.size(), false);
        m_streaming_state.total_matches = 0;
        m_streaming_state.total_errors = 0;
        m_streaming_state.total_fuzzy = 0;
        m_streaming_state.total_chunks = 0;
    }

    void load_reference(const std::string& text) {
        m_reference_words.clear();
        m_reference_words_normalized.clear();

        std::istringstream stream(text);
        std::string word;
        while (stream >> word) {
            m_reference_words.push_back(word);
            m_reference_words_normalized.push_back(normalize_word(word));
        }

        log("Loaded " + std::to_string(m_reference_words.size()) +
            " reference words for alignment");

        // Reset streaming state when loading new reference
        if (m_streaming_mode) {
            reset_streaming_state();
        }
    }

    // New method specifically for streaming chunks
    std::string mark_streaming_chunk(const std::string& hypothesis_chunk) {
        if (m_reference_words.empty()) {
            log("No reference text loaded!", "ERROR");
            return hypothesis_chunk;
        }

        // Tokenize hypothesis chunk
        std::vector<std::string> hyp_words;
        std::istringstream stream(hypothesis_chunk);
        std::string word;
        while (stream >> word) {
            hyp_words.push_back(word);
        }

        if (hyp_words.empty()) {
            return "";
        }

        m_streaming_state.total_chunks++;

        log("Processing streaming chunk #" + std::to_string(m_streaming_state.total_chunks) +
                " with " + std::to_string(hyp_words.size()) +
                " words, starting search from position " +
                std::to_string(m_streaming_state.last_matched_ref_position), "INFO");

        // Find where this chunk aligns in the reference
        auto alignment = align_streaming_chunk(hyp_words);

        // Build output only for the hypothesis words (not the entire reference)
        return build_streaming_output(hyp_words, alignment);
    }

    std::string mark_differences(const std::string& hypothesis) {
        if (m_reference_words.empty()) {
            log("No reference text loaded!", "ERROR");
            return hypothesis;
        }

        // If in streaming mode, use streaming method
        if (m_streaming_mode) {
            return mark_streaming_chunk(hypothesis);
        }

        // Tokenize hypothesis
        std::vector<std::string> hyp_words;
        std::istringstream stream(hypothesis);
        std::string word;
        while (stream >> word) {
            hyp_words.push_back(word);
        }

        if (hyp_words.empty()) {
            log("Empty hypothesis provided!", "WARN");
            return hypothesis;
        }

        // Get alignment
        auto alignment = align_sequences(hyp_words);

        // Build marked output
        std::stringstream result;

        for (size_t i = 0; i < alignment.operations.size(); i++) {
            if (i > 0 && alignment.operations[i] != AlignOp::DELETE) {
                result << " ";
            }

            switch (alignment.operations[i]) {
            case AlignOp::MATCH:
                result << GREEN << hyp_words[alignment.hyp_indices[i]] << RESET;
                break;

            case AlignOp::FUZZY_MATCH:
                result << MAGENTA << hyp_words[alignment.hyp_indices[i]] << RESET;
                result << DIM << "~(" << m_reference_words[alignment.ref_indices[i]] << ")" << RESET;
                if (m_config.show_confidence) {
                    result << DIM << "[" << std::fixed << std::setprecision(2)
                    << alignment.confidence_scores[i] << "]" << RESET;
                }
                break;

            case AlignOp::SUBSTITUTE:
                result << YELLOW << hyp_words[alignment.hyp_indices[i]] << RESET;
                result << DIM << "≠(" << m_reference_words[alignment.ref_indices[i]] << ")" << RESET;
                break;

            case AlignOp::INSERT:
                result << CYAN << "[+" << hyp_words[alignment.hyp_indices[i]] << "]" << RESET;
                break;

            case AlignOp::DELETE:
                if (i > 0) result << " ";
                result << RED << "[-" << m_reference_words[alignment.ref_indices[i]] << "]" << RESET;
                break;

            case AlignOp::REORDER:
                result << UNDERLINE << YELLOW << hyp_words[alignment.hyp_indices[i]] << RESET;
                result << DIM << "↔(" << m_reference_words[alignment.ref_indices[i]] <<
                    "@" << alignment.ref_indices[i] << ")" << RESET;
                break;
            }
        }

        return result.str();
    }

    // Simpler marking with just colors
    std::string mark_differences_simple(const std::string& hypothesis) {
        if (m_reference_words.empty()) return hypothesis;

        // Tokenize hypothesis
        std::vector<std::string> hyp_words;
        std::istringstream stream(hypothesis);
        std::string word;
        while (stream >> word) {
            hyp_words.push_back(word);
        }

        // Get alignment
        auto alignment = align_sequences(hyp_words);

        // Build marked output
        std::stringstream result;

        for (size_t i = 0; i < alignment.operations.size(); i++) {
            if (alignment.operations[i] == AlignOp::DELETE) continue;

            if (result.tellp() > 0) result << " ";

            switch (alignment.operations[i]) {
            case AlignOp::MATCH:
                result << hyp_words[alignment.hyp_indices[i]];
                break;

            case AlignOp::SUBSTITUTE:
            case AlignOp::INSERT:
            case AlignOp::FUZZY_MATCH:
                result << RED << hyp_words[alignment.hyp_indices[i]] << RESET;
                break;

            default:
                break;
            }
        }

        return result.str();
    }

    // Debug hypothesis
    void debug_hypothesis(const std::string& hypothesis) {
        std::cout << "\n=== HYPOTHESIS DEBUG ===" << std::endl;
        std::cout << "Raw hypothesis: '" << hypothesis << "'" << std::endl;
        std::cout << "Length: " << hypothesis.length() << " characters" << std::endl;

        // Check for special characters
        std::cout << "Character codes: ";
        for (size_t i = 0; i < std::min(size_t(20), hypothesis.length()); i++) {
            std::cout << (int)hypothesis[i] << " ";
        }
        std::cout << std::endl;

        // Try tokenization
        std::vector<std::string> words;
        std::istringstream stream(hypothesis);
        std::string word;
        while (stream >> word) {
            words.push_back(word);
        }

        std::cout << "Tokenized into " << words.size() << " words" << std::endl;
        if (!words.empty()) {
            std::cout << "First word: '" << words[0] << "'" << std::endl;
            std::cout << "Last word: '" << words.back() << "'" << std::endl;
        }
        std::cout << "===================\n" << std::endl;
    }

    // Get streaming statistics
    void print_streaming_stats() {
        std::cout << "\n=== Streaming Transcription Stats ===" << std::endl;
        std::cout << "Total chunks processed: " << m_streaming_state.total_chunks << std::endl;
        std::cout << "Processed up to reference position: "
                  << m_streaming_state.last_matched_ref_position
                  << " / " << m_reference_words.size() << std::endl;

        double progress = (double)m_streaming_state.last_matched_ref_position /
                          m_reference_words.size() * 100.0;
        std::cout << "Progress: " << std::fixed << std::setprecision(1)
                  << progress << "%" << std::endl;

        std::cout << "\nAccuracy breakdown:" << std::endl;
        std::cout << "  Exact matches: " << m_streaming_state.total_matches << std::endl;
        std::cout << "  Fuzzy matches: " << m_streaming_state.total_fuzzy << std::endl;
        std::cout << "  Errors: " << m_streaming_state.total_errors << std::endl;

        int total_words = m_streaming_state.total_matches +
                          m_streaming_state.total_fuzzy +
                          m_streaming_state.total_errors;

        if (total_words > 0) {
            double accuracy = (double)(m_streaming_state.total_matches +
                                        m_streaming_state.total_fuzzy) / total_words;
            std::cout << "\nOverall accuracy: " << std::setprecision(1)
                      << accuracy * 100 << "%" << std::endl;
        }

        // Show remaining text preview
        if (m_streaming_state.last_matched_ref_position < m_reference_words.size()) {
            std::cout << "\nNext expected words: ";
            for (size_t i = m_streaming_state.last_matched_ref_position;
                 i < std::min(m_streaming_state.last_matched_ref_position + 5,
                              m_reference_words.size()); i++) {
                std::cout << "'" << m_reference_words[i] << "' ";
            }
            std::cout << "..." << std::endl;
        }

        std::cout << "==================================\n" << std::endl;
    }

    // Get detailed alignment report
    std::string get_alignment_report(const std::string& hypothesis) {
        if (m_reference_words.empty()) return "No reference text loaded!";

        std::vector<std::string> hyp_words;
        std::istringstream stream(hypothesis);
        std::string word;
        while (stream >> word) {
            hyp_words.push_back(word);
        }

        auto alignment = align_sequences(hyp_words);

        std::stringstream report;
        report << "\n=== ALIGNMENT REPORT ===\n";
        report << "Reference length: " << m_reference_words.size() << " words\n";
        report << "Hypothesis length: " << hyp_words.size() << " words\n";
        report << "Edit distance: " << alignment.distance << "\n";
        report << "Overall confidence: " << std::fixed << std::setprecision(2)
               << alignment.overall_confidence << "\n\n";

        // Count operations
        int matches = 0, fuzzy = 0, subs = 0, ins = 0, del = 0;
        for (auto op : alignment.operations) {
            switch (op) {
            case AlignOp::MATCH: matches++; break;
            case AlignOp::FUZZY_MATCH: fuzzy++; break;
            case AlignOp::SUBSTITUTE: subs++; break;
            case AlignOp::INSERT: ins++; break;
            case AlignOp::DELETE: del++; break;
            default: break;
            }
        }

        report << "--- Statistics ---\n";
        report << "Exact matches: " << matches << "\n";
        report << "Fuzzy matches: " << fuzzy << "\n";
        report << "Substitutions: " << subs << "\n";
        report << "Insertions: " << ins << "\n";
        report << "Deletions: " << del << "\n";

        double wer = (double)(subs + ins + del) / m_reference_words.size();
        report << "Word Error Rate: " << std::setprecision(1) << wer * 100 << "%\n\n";

        // Show errors with context
        report << "--- Errors with Context ---\n";
        for (size_t i = 0; i < alignment.operations.size(); i++) {
            if (alignment.operations[i] != AlignOp::MATCH) {
                report << get_operation_description(alignment, i, hyp_words) << "\n";
            }
        }

        return report.str();
    }

    // Get current position in reference (for streaming)
    size_t get_current_position() const {
        return m_streaming_state.last_matched_ref_position;
    }

    // Get total reference length
    size_t get_reference_length() const {
        return m_reference_words.size();
    }

    void reset() {
        m_reference_words.clear();
        m_reference_words_normalized.clear();
        m_current_position = 0;
        m_streaming_mode = false;
        reset_streaming_state();
        log("Marker reset");
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
};
