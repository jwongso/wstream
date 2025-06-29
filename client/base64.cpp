#include "base64.h"
#include <stdexcept>
#include <algorithm>

// Optimized encode table (constexpr for compile-time computation)
static constexpr char encode_table[64] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
    'Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f',
    'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
    'w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','/'
};

// Optimized decode table (compile-time computed)
static constexpr int8_t decode_table[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 0-15
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 16-31
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,  // 32-47 ('+' and '/')
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,  // 48-63 ('0'-'9')
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14, // 64-79 ('A'-'O')
    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,  // 80-95 ('P'-'Z')
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40, // 96-111 ('a'-'o')
    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,  // 112-127 ('p'-'z')
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 128-143
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 144-159
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 160-175
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 176-191
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 192-207
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 208-223
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,  // 224-239
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1   // 240-255
};

std::string base64::encode(const void* data, size_t size) {
    if (size == 0) return {};

    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    const size_t output_size = ((size + 2) / 3) * 4;  // Calculate exact output size

    std::string result;
    result.reserve(output_size);

    // Process 3 bytes at a time
    size_t i = 0;
    for (; i + 2 < size; i += 3) {
        const uint32_t triple = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];

        result += encode_table[(triple >> 18) & 0x3F];
        result += encode_table[(triple >> 12) & 0x3F];
        result += encode_table[(triple >> 6) & 0x3F];
        result += encode_table[triple & 0x3F];
    }

    // Handle remaining bytes
    if (i < size) {
        uint32_t triple = bytes[i] << 16;
        if (i + 1 < size) {
            triple |= bytes[i + 1] << 8;
        }

        result += encode_table[(triple >> 18) & 0x3F];
        result += encode_table[(triple >> 12) & 0x3F];
        result += (i + 1 < size) ? encode_table[(triple >> 6) & 0x3F] : '=';
        result += '=';
    }

    return result;
}

std::vector<uint8_t> base64::decode(const std::string& encoded) {
    if (encoded.empty()) return {};

    const size_t in_len = encoded.size();
    std::vector<uint8_t> result;
    result.reserve((in_len * 3) / 4 + 3); // Reserve exact needed space + padding

    uint32_t val = 0;
    int bits = -8; // Start with negative to handle first iteration

    for (size_t i = 0; i < in_len; ++i) {
        const uint8_t c = encoded[i];
        if (c == '=') break; // Stop at padding

        const int8_t decoded = decode_table[c];
        if (decoded == -1) continue; // Skip invalid chars

        val = (val << 6) | decoded;
        bits += 6;

        if (bits >= 0) {
            result.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }

    return result;
}

bool base64::decode_audio_fast(const std::string& encoded, std::vector<int16_t>& output) {
    if (encoded.empty()) {
        output.clear();
        return true;
    }

    const size_t in_len = encoded.size();
    const size_t estimated_bytes = (in_len * 3) / 4;
    const size_t estimated_samples = estimated_bytes / sizeof(int16_t);

    // Pre-allocate output vector
    output.clear();
    output.reserve(estimated_samples + 1); // +1 for safety

    uint32_t val = 0;
    int bits = -8;
    uint8_t byte_buffer[2];
    int buffer_pos = 0;

    for (size_t i = 0; i < in_len; ++i) {
        const uint8_t c = encoded[i];
        if (c == '=') break;

        const int8_t decoded = decode_table[c];
        if (decoded == -1) continue;

        val = (val << 6) | decoded;
        bits += 6;

        if (bits >= 0) {
            const uint8_t byte = static_cast<uint8_t>((val >> bits) & 0xFF);
            byte_buffer[buffer_pos++] = byte;

            // When we have 2 bytes, convert to int16_t
            if (buffer_pos == 2) {
                int16_t sample;
                std::memcpy(&sample, byte_buffer, sizeof(int16_t));
                output.push_back(sample);
                buffer_pos = 0;
            }

            bits -= 8;
        }
    }

    // Handle leftover byte (should be rare for audio data)
    if (buffer_pos == 1) {
        // Pad with zero and add final sample
        byte_buffer[1] = 0;
        int16_t sample;
        std::memcpy(&sample, byte_buffer, sizeof(int16_t));
        output.push_back(sample);
    }

    return true;
}

std::string base64::encode_audio(const std::vector<int16_t>& samples) {
    if (samples.empty()) {
        return "";
    }

    // Convert int16_t vector to bytes
    const size_t byte_size = samples.size() * sizeof(int16_t);
    return encode(samples.data(), byte_size);
}
