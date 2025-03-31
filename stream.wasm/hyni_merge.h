#ifndef HYNI_MERGE_H
#define HYNI_MERGE_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>

namespace hyni
{

class [[nodiscard]] HyniMerge final {  // Mark as final if not meant to be inherited
// Likely/unlikely macros for branch prediction
#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif
public:
    [[nodiscard]] static std::vector<std::string> splitAndNormalize(const std::string& text) noexcept {
        std::vector<std::string> result;
        if (text.empty()) return result;

        // More precise capacity estimation
        const int spaceCount = std::count(text.begin(), text.end(), ' ');
        result.reserve(spaceCount + 1);  // At least as many words as spaces + 1

        std::string current;
        current.reserve(32);  // Keep reasonable word length

        const char* data = text.data();
        const int length = text.length();

        for (int i = 0; i < length; ++i) {
            const char& ch = data[i];
            if (isFilteredChar(ch)) {
                if (!current.empty()) {
                    result.emplace_back(std::move(current));
                    current.clear();
                }
            } else {
                current += ch;
            }
        }

        if (!current.empty()) {
            result.emplace_back(std::move(current));
        }

        // Only shrink if significantly over-allocated
        if (result.capacity() > result.size() * 2) {
            result.shrink_to_fit();
        }
        return result;
    }

    [[nodiscard]] static std::string mergeStrings(const std::string& A, const std::string& B) noexcept {
        if (A.empty()) return B;
        if (B.empty()) return A;

        const std::vector<std::string> base = splitAndNormalize(A);
        const std::vector<std::string> tail = splitAndNormalize(B);

        if (base.empty()) return B;
        if (tail.empty()) return A;

        // Optimized hash function with better mixing
        auto computeTrigramHash = [](const std::string& a, const std::string& b, const std::string& c) noexcept {
            size_t h1 = std::hash<std::string>{}(a);
            size_t h2 = std::hash<std::string>{}(b);
            size_t h3 = std::hash<std::string>{}(c);
            return ((h1 * 0xFEA5B) ^ (h2 * 0x8DA6B) ^ (h3 * 0x7A97C)) * 0x9E3779B9;
        };

        // Build trigram index with view-based optimization
        std::unordered_map<uint64_t, std::vector<int>> trigramIndex;
        trigramIndex.reserve(std::max(0, static_cast<int>(base.size()) - 2));

        for (size_t i = 0; i + 2 < base.size(); ++i) {
            if (!base[i].empty() && !base[i + 1].empty() && !base[i + 2].empty()) {
                trigramIndex[computeTrigramHash(base[i], base[i + 1], base[i + 2])].push_back(i);
            }
        }

        // Optimized matching with early exit
        int bestMatchIndex = findBestMatch(tail, base, trigramIndex);

        // Calculate exact required size
        const int totalSize = calculateResultSize(A, B, base, tail, bestMatchIndex);
        std::string result;
        result.reserve(totalSize);

        if (bestMatchIndex > 0) {
            result = base[0];
            for (int i = 1; i < bestMatchIndex; ++i) {
                result += ' ';
                result += base[i];
            }
            result += ' ';
        } else if (bestMatchIndex < 0) {
            result = A;
            if (!result.empty() && result.back() != ' ') result += ' ';
        }

        result += B;
        return result;
    }

private:
    [[nodiscard]] static constexpr bool isFilteredChar(char ch) noexcept {
        const unsigned char uc = static_cast<unsigned char>(ch);
        return (uc == ' ') | (uc == ',') | (uc == '.') | (uc == ';') | (uc == '-');
    }

    [[nodiscard]] static int findBestMatch(
        const std::vector<std::string>& tail,
        const std::vector<std::string>& base,
        const std::unordered_map<uint64_t, std::vector<int>>& trigramIndex) noexcept {

        if (tail.size() < 3) return -1;

        for (size_t i = 0; i + 2 < tail.size(); ++i) {
            if (tail[i].empty() || tail[i + 1].empty() || tail[i + 2].empty()) {
                continue;
            }

            // Optimized hash function with better mixing
            auto computeTrigramHash = [](const std::string& a, const std::string& b, const std::string& c) noexcept {
                size_t h1 = std::hash<std::string>{}(a);
                size_t h2 = std::hash<std::string>{}(b);
                size_t h3 = std::hash<std::string>{}(c);
                return ((h1 * 0xFEA5B) ^ (h2 * 0x8DA6B) ^ (h3 * 0x7A97C)) * 0x9E3779B9;
            };

            const uint64_t tailHash = computeTrigramHash(tail[i], tail[i + 1], tail[i + 2]);

            if (const auto it = trigramIndex.find(tailHash); it != trigramIndex.end()) {
                for (int baseIndex : it->second) {
                    if (static_cast<size_t>(baseIndex) + 2 < base.size() &&
                        tail[i] == base[baseIndex] &&
                        tail[i + 1] == base[baseIndex + 1] &&
                        tail[i + 2] == base[baseIndex + 2]) {
                        return baseIndex;
                    }
                }
            }
        }

        return -1;
    }

    [[nodiscard]] static int calculateResultSize(
        const std::string& A, const std::string& B,
        const std::vector<std::string>& base, const std::vector<std::string>& tail,
        int bestMatchIndex) noexcept {

        if (bestMatchIndex > 0) {
            int size = std::accumulate(base.begin(), base.begin() + bestMatchIndex, 0,
                                       [](int sum, const std::string& s) { return sum + static_cast<int>(s.length()); });

            return size + bestMatchIndex - 1 + 1 + static_cast<int>(B.length());
        }

        return static_cast<int>(A.length()) + (A.empty() || A.back() == ' ' ? 0 : 1) + static_cast<int>(B.length());
    }
};

} // hyni

#endif
