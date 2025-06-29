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

#include "benchmark_manager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <numeric>
#include <map>
#include <regex>
#include <filesystem>

namespace fs = std::filesystem;

benchmark_manager::benchmark_manager() : m_is_running(false), m_total_samples(0) {
    // Initialize timestamps to current time
    m_start_time = std::chrono::steady_clock::now();
    m_last_segment_time = m_start_time;
}

benchmark_manager::~benchmark_manager() {
    if (m_is_running) {
        stop();
    }
}

void benchmark_manager::set_reference_text(const std::string& text) {
    m_reference_text = normalize_text(text);
    std::cout << "[Benchmark] Reference text set ("
              << m_reference_text.length() << " characters)" << std::endl;
}

void benchmark_manager::start() {
    m_segments.clear();
    m_total_samples = 0;
    m_is_running = true;
    m_start_time = std::chrono::steady_clock::now();
    m_last_segment_time = m_start_time;

    std::cout << "[Benchmark] Benchmark manager started" << std::endl;
}

benchmark_manager::benchmark_results benchmark_manager::stop() {
    m_is_running = false;
    auto end_time = std::chrono::steady_clock::now();

    benchmark_results results = get_current_results();

    // Calculate final timing metrics
    results.total_processing_time_ms =
        std::chrono::duration<double, std::milli>(end_time - m_start_time).count();

    std::cout << "[Benchmark] Benchmark completed:" << std::endl;
    std::cout << "  WER: " << std::fixed << std::setprecision(2)
              << results.word_error_rate << "%" << std::endl;
    std::cout << "  CER: " << results.character_error_rate << "%" << std::endl;
    std::cout << "  RTF: " << results.real_time_factor << "x" << std::endl;
    std::cout << "  Avg Latency: " << results.average_latency_ms << " ms" << std::endl;

    return results;
}

void benchmark_manager::add_transcription(const std::string& text,
                                          double confidence,
                                          size_t audio_samples,
                                          double processing_latency_ms) {
    if (!m_is_running) return;

    auto now = std::chrono::steady_clock::now();

    transcription_segment segment;
    segment.text = text;
    segment.start_time = m_last_segment_time;
    segment.end_time = now;
    segment.confidence = confidence;
    segment.audio_samples_processed = audio_samples;

    if (processing_latency_ms > 0) {
        segment.processing_latency_ms = processing_latency_ms;
    } else {
        // Fall back to time between calls (less accurate)
        segment.processing_latency_ms = std::chrono::duration<double, std::milli>(
                                            segment.end_time - segment.start_time).count();
    }

    m_segments.push_back(segment);
    m_total_samples += audio_samples;
    m_last_segment_time = now;

    // Call progress callback if set
    if (m_progress_callback) {
        m_progress_callback(get_current_results());
    }
}

benchmark_manager::benchmark_results benchmark_manager::get_current_results() const {
    benchmark_results results;

    // Build hypothesis text from segments
    results.hypothesis_text.clear();
    for (const auto& segment : m_segments) {
        if (!results.hypothesis_text.empty() && !segment.text.empty()) {
            results.hypothesis_text += " ";
        }
        results.hypothesis_text += segment.text;
    }
    results.hypothesis_text = normalize_text(results.hypothesis_text);

    results.reference_text = m_reference_text;
    results.segments = m_segments;

    // Calculate accuracy metrics
    if (!m_reference_text.empty() && !results.hypothesis_text.empty()) {
        results.word_error_rate = calculate_wer(
            m_reference_text,
            results.hypothesis_text,
            &results.word_substitutions,
            &results.word_deletions,
            &results.word_insertions
            );

        results.character_error_rate = calculate_cer(
            m_reference_text,
            results.hypothesis_text
            );

        auto ref_words = tokenize(m_reference_text);
        results.total_words = ref_words.size();
        results.word_errors = results.word_substitutions +
                              results.word_deletions +
                              results.word_insertions;
    }

    // Calculate timing metrics
    if (!m_segments.empty()) {
        std::vector<double> latencies;
        double total_confidence = 0.0;

        for (const auto& segment : m_segments) {
            double latency = std::chrono::duration<double, std::milli>(
                                 segment.end_time - segment.start_time).count();
            latencies.push_back(latency);
            total_confidence += segment.confidence;
        }

        results.average_latency_ms = std::accumulate(
                                         latencies.begin(), latencies.end(), 0.0) / latencies.size();

        auto minmax = std::minmax_element(latencies.begin(), latencies.end());
        results.min_latency_ms = *minmax.first;
        results.max_latency_ms = *minmax.second;

        results.average_confidence = total_confidence / m_segments.size();
    }

    // Calculate throughput metrics
    results.total_samples_processed = m_total_samples;
    results.total_segments = m_segments.size();

    if (m_is_running) {
        auto duration = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - m_start_time).count();
        if (duration > 0) {
            results.samples_per_second = m_total_samples / duration;
        }
    }

    // Calculate audio duration (assuming 16kHz sample rate)
    const double sample_rate = 16000.0;
    results.total_audio_duration_ms = (m_total_samples / sample_rate) * 1000.0;

    // Calculate real-time factor
    if (results.total_audio_duration_ms > 0 && results.total_processing_time_ms > 0) {
        results.real_time_factor =
            results.total_processing_time_ms / results.total_audio_duration_ms;
    }

    return results;
}

double benchmark_manager::calculate_wer(const std::string& reference,
                                        const std::string& hypothesis,
                                        int* substitutions,
                                        int* deletions,
                                        int* insertions) {
    auto ref_words = tokenize(reference);
    auto hyp_words = tokenize(hypothesis);

    if (ref_words.empty()) {
        return hyp_words.empty() ? 0.0 : 100.0;
    }

    int subs = 0, dels = 0, ins = 0;
    int distance = levenshtein_distance(ref_words, hyp_words, &subs, &dels, &ins);

    if (substitutions) *substitutions = subs;
    if (deletions) *deletions = dels;
    if (insertions) *insertions = ins;

    return (distance * 100.0) / ref_words.size();
}

double benchmark_manager::calculate_cer(const std::string& reference,
                                        const std::string& hypothesis) {
    if (reference.empty()) {
        return hypothesis.empty() ? 0.0 : 100.0;
    }

    // Convert strings to character vectors
    std::vector<std::string> ref_chars, hyp_chars;
    for (char c : reference) {
        if (!std::isspace(c)) {
            ref_chars.push_back(std::string(1, c));
        }
    }
    for (char c : hypothesis) {
        if (!std::isspace(c)) {
            hyp_chars.push_back(std::string(1, c));
        }
    }

    int distance = levenshtein_distance(ref_chars, hyp_chars, nullptr, nullptr, nullptr);
    return (distance * 100.0) / ref_chars.size();
}

void benchmark_manager::export_results(const benchmark_results& results,
                                       const std::string& output_path,
                                       const std::string& model_path) const {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "[Benchmark] Failed to open output file: " << output_path << std::endl;
        return;
    }

    file << "=== WSTREAM BENCHMARK RESULTS ===" << std::endl;
    file << std::endl;

    // Add model information
    file << "MODEL INFORMATION:" << std::endl;
    file << "  Model: " << fs::path(model_path).filename().string() << std::endl;
    file << "  Full path: " << model_path << std::endl;

    // Get model file size
    try {
        std::ifstream model_file(model_path, std::ios::binary | std::ios::ate);
        if (model_file.is_open()) {
            size_t file_size = model_file.tellg();
            file << "  Size: " << std::fixed << std::setprecision(2)
                 << (file_size / (1024.0 * 1024.0)) << " MB ("
                 << file_size << " bytes)" << std::endl;
            model_file.close();
        }
    } catch (...) {
        file << "  Size: Unknown" << std::endl;
    }

    // Add timestamp
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    file << "  Timestamp: " << std::ctime(&now_time_t);
    file << std::endl;

    file << "ACCURACY METRICS:" << std::endl;
    file << "  Word Error Rate (WER): " << std::fixed << std::setprecision(2)
         << results.word_error_rate << "%" << std::endl;
    file << "  Character Error Rate (CER): " << results.character_error_rate << "%" << std::endl;
    file << "  Total Words: " << results.total_words << std::endl;
    file << "  Word Errors: " << results.word_errors << std::endl;
    file << "    Substitutions: " << results.word_substitutions << std::endl;
    file << "    Deletions: " << results.word_deletions << std::endl;
    file << "    Insertions: " << results.word_insertions << std::endl;
    file << std::endl;

    file << "TIMING METRICS:" << std::endl;
    file << "  Total Audio Duration: " << results.total_audio_duration_ms / 1000.0 << " s"
         << std::endl;
    file << "  Total Processing Time: " << results.total_processing_time_ms / 1000.0 << " s"
         << std::endl;
    file << "  Real-Time Factor: " << results.real_time_factor << "x" << std::endl;
    file << "  Average Latency: " << results.average_latency_ms << " ms" << std::endl;
    file << "  Min Latency: " << results.min_latency_ms << " ms" << std::endl;
    file << "  Max Latency: " << results.max_latency_ms << " ms" << std::endl;
    file << std::endl;

    file << "THROUGHPUT METRICS:" << std::endl;
    file << "  Total Samples: " << results.total_samples_processed << std::endl;
    file << "  Total Segments: " << results.total_segments << std::endl;
    file << "  Samples/Second: " << std::fixed << std::setprecision(0)
         << results.samples_per_second << std::endl;
    file << std::endl;

    file << "QUALITY METRICS:" << std::endl;
    file << "  Average Confidence: " << std::fixed << std::setprecision(3)
         << results.average_confidence << std::endl;
    file << std::endl;

    file << "REFERENCE TEXT:" << std::endl;
    file << results.reference_text << std::endl;
    file << std::endl;

    file << "HYPOTHESIS TEXT:" << std::endl;
    file << results.hypothesis_text << std::endl;
    file << std::endl;

    file << "\nSEGMENT ANALYSIS:" << std::endl;
    file << "  Average segment duration: " << std::fixed << std::setprecision(2)
         << (results.total_audio_duration_ms / results.total_segments) << " ms" << std::endl;
    file << "  Average processing time per segment: " << std::fixed << std::setprecision(2)
         << results.average_latency_ms << " ms" << std::endl;
    file << "  Segment RTF: " << std::fixed << std::setprecision(2)
         << (results.average_latency_ms / (results.total_audio_duration_ms / results.total_segments))
         << "x" << std::endl;

    // Analyze error patterns
    if (!results.segments.empty() && !results.reference_text.empty()) {
        file << "\nERROR PATTERN ANALYSIS:" << std::endl;

        // Common error patterns
        std::map<std::string, int> error_patterns;
        std::string hyp_normalized = benchmark_manager::normalize_text(results.hypothesis_text);
        std::string ref_normalized = benchmark_manager::normalize_text(results.reference_text);

        // Detect repetitions (e.g., "democracy. democracy.")
        std::regex repetition_pattern(R"(\b(\w+)[\.\s]+\1\b)");
        std::smatch matches;
        std::string::const_iterator search_start(hyp_normalized.cbegin());
        while (std::regex_search(search_start, hyp_normalized.cend(), matches, repetition_pattern)) {
            error_patterns["Repeated word: " + matches[1].str()]++;
            search_start = matches.suffix().first;
        }

        // Detect punctuation errors
        int ref_periods = std::count(ref_normalized.begin(), ref_normalized.end(), '.');
        int hyp_periods = std::count(hyp_normalized.begin(), hyp_normalized.end(), '.');
        if (std::abs(ref_periods - hyp_periods) > 3) {
            file << "  Punctuation errors: " << std::abs(ref_periods - hyp_periods)
            << " missing/extra periods" << std::endl;
        }

        // Report common error patterns
        for (const auto& [pattern, count] : error_patterns) {
            if (count > 1) {
                file << "  " << pattern << ": " << count << " occurrences" << std::endl;
            }
        }
    }

    file.close();
    std::cout << "[Benchmark] Results exported to: " << output_path << std::endl;
}

// Static helper methods
std::vector<std::string> benchmark_manager::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string word;

    while (stream >> word) {
        // Convert to lowercase and remove punctuation
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        word.erase(std::remove_if(word.begin(), word.end(),
                                  [](char c) { return !std::isalnum(c); }),
                   word.end());

        if (!word.empty()) {
            tokens.push_back(word);
        }
    }

    return tokens;
}

std::string benchmark_manager::normalize_text(const std::string& text) {
    std::string normalized = text;

    // Convert to lowercase
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);

    // Remove extra whitespace
    auto new_end = std::unique(normalized.begin(), normalized.end(),
                               [](char a, char b) { return std::isspace(a) && std::isspace(b); });
    normalized.erase(new_end, normalized.end());

    // Trim leading/trailing whitespace
    normalized.erase(0, normalized.find_first_not_of(" \t\n\r"));
    normalized.erase(normalized.find_last_not_of(" \t\n\r") + 1);

    return normalized;
}

int benchmark_manager::levenshtein_distance(const std::vector<std::string>& ref,
                                            const std::vector<std::string>& hyp,
                                            int* subs, int* dels, int* ins) {
    const size_t m = ref.size();
    const size_t n = hyp.size();

    // Create DP table
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

    // Initialize base cases
    for (size_t i = 0; i <= m; ++i) dp[i][0] = i;
    for (size_t j = 0; j <= n; ++j) dp[0][j] = j;

    // Fill DP table
    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            if (ref[i-1] == hyp[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + std::min({dp[i-1][j],    // deletion
                                         dp[i][j-1],      // insertion
                                         dp[i-1][j-1]});  // substitution
            }
        }
    }

    // Backtrack to count operation types
    if (subs != nullptr || dels != nullptr || ins != nullptr) {
        int sub_count = 0, del_count = 0, ins_count = 0;
        size_t i = m, j = n;

        while (i > 0 || j > 0) {
            if (i == 0) {
                ins_count++;
                j--;
            } else if (j == 0) {
                del_count++;
                i--;
            } else if (ref[i-1] == hyp[j-1]) {
                i--;
                j--;
            } else {
                int min_val = std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});

                if (dp[i-1][j-1] == min_val) {
                    sub_count++;
                    i--;
                    j--;
                } else if (dp[i-1][j] == min_val) {
                    del_count++;
                    i--;
                } else {
                    ins_count++;
                    j--;
                }
            }
        }

        if (subs != nullptr) *subs = sub_count;
        if (dels != nullptr) *dels = del_count;
        if (ins != nullptr) *ins = ins_count;
    }

    return dp[m][n];
}
