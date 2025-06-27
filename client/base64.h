#pragma once

#include <string>
#include <vector>
#include <cstring>
#include <stdint.h>
#include <stdexcept>

/**
 * @brief Optimized Base64 encoding/decoding utilities
 */
namespace base64 {
/**
     * @brief Encodes binary data to Base64 string (optimized)
     * @param data Pointer to binary data
     * @param size Size of binary data in bytes
     * @return Base64 encoded string
     */
std::string encode(const void* data, size_t size);

/**
     * @brief Encodes vector of data to Base64 string (optimized)
     * @param data Vector containing binary data
     * @return Base64 encoded string
     */
template <typename T>
std::string encode(const std::vector<T>& data) {
    if (data.empty()) return {};
    return encode(data.data(), data.size() * sizeof(T));
}

/**
     * @brief Decodes Base64 string to binary data (optimized)
     * @param encoded Base64 encoded string
     * @return Vector of bytes containing decoded data
     */
std::vector<uint8_t> decode(const std::string& encoded);

/**
     * @brief Decodes Base64 string to vector of specific type (optimized)
     * @param encoded Base64 encoded string
     * @return Vector of specified type containing decoded data
     */
template <typename T>
std::vector<T> decode_to(const std::string& encoded) {
    std::vector<uint8_t> bytes = decode(encoded);
    if (bytes.size() % sizeof(T) != 0) {
        throw std::runtime_error("Base64 decoded data size is not a multiple of element size");
    }

    std::vector<T> result(bytes.size() / sizeof(T));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

/**
     * @brief Fast in-place decode for int16_t audio data (specialized)
     * @param encoded Base64 encoded string
     * @param output Output vector to store decoded int16_t samples
     * @return true if successful, false if invalid input
     */
bool decode_audio_fast(const std::string& encoded, std::vector<int16_t>& output);
}
