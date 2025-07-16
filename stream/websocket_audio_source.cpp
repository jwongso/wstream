#include "websocket_audio_source.h"
#include <iostream>
#include <algorithm>
#include <cstring>

// SIMD headers
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Thread-local buffers
thread_local std::unique_ptr<websocket_audio_source::AlignedBuffer>
    websocket_audio_source::t_conversion_buffer;

thread_local std::unique_ptr<websocket_audio_source::AlignedBuffer>
    websocket_audio_source::t_batch_buffer;

// AlignedBuffer implementation
websocket_audio_source::AlignedBuffer::AlignedBuffer(size_t size) : capacity(size) {
#ifdef _WIN32
    data = static_cast<float*>(_aligned_malloc(size * sizeof(float), SIMD_ALIGNMENT));
    if (!data) throw std::bad_alloc();
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, SIMD_ALIGNMENT, size * sizeof(float)) != 0) {
        throw std::bad_alloc();
    }
    data = static_cast<float*>(ptr);
#endif
}

websocket_audio_source::AlignedBuffer::~AlignedBuffer() {
#ifdef _WIN32
    _aligned_free(data);
#else
    free(data);
#endif
}

// Helper for cache prefetching
inline void prefetch_read(const void* addr) {
#ifdef __builtin_prefetch
    __builtin_prefetch(addr, 0, 3);  // Read, high temporal locality
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#endif
}

websocket_audio_source::AlignedBuffer*
websocket_audio_source::get_conversion_buffer(size_t min_size) {
    if (!t_conversion_buffer || t_conversion_buffer->capacity < min_size) {
        // Allocate with 50% headroom to reduce reallocations
        t_conversion_buffer = std::make_unique<AlignedBuffer>(min_size + min_size / 2);
    }
    return t_conversion_buffer.get();
}

websocket_audio_source::AlignedBuffer*
websocket_audio_source::get_batch_buffer() {
    if (!t_batch_buffer) {
        t_batch_buffer = std::make_unique<AlignedBuffer>(QUEUE_BATCH_SIZE);
    }
    return t_batch_buffer.get();
}

// SIMD conversion implementations
void websocket_audio_source::convert_int16_to_float_simd(const int16_t* src,
                                                         float* dst,
                                                         size_t count) {
#ifdef __AVX2__
    convert_int16_to_float_avx2(src, dst, count);
#elif defined(__SSE2__)
    convert_int16_to_float_sse2(src, dst, count);
#elif defined(__ARM_NEON)
    convert_int16_to_float_neon(src, dst, count);
#else
    convert_int16_to_float_scalar(src, dst, count);
#endif
}

#ifdef __AVX2__
void websocket_audio_source::convert_int16_to_float_avx2(const int16_t* src,
                                                         float* dst,
                                                         size_t count) {
    const __m256 scale = _mm256_set1_ps(1.0f / 32768.0f);
    size_t simd_count = count & ~15;  // Process 16 samples at a time (2 AVX2 ops)

    prefetch_read(src);

    for (size_t i = 0; i < simd_count; i += 16) {
        // Prefetch ahead
        if (i + 32 < count) {
            prefetch_read(src + i + 32);
        }

        // First 8 samples
        __m128i i16_vec1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256i i32_vec1 = _mm256_cvtepi16_epi32(i16_vec1);
        __m256 f32_vec1 = _mm256_cvtepi32_ps(i32_vec1);
        f32_vec1 = _mm256_mul_ps(f32_vec1, scale);

        // Second 8 samples
        __m128i i16_vec2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i + 8));
        __m256i i32_vec2 = _mm256_cvtepi16_epi32(i16_vec2);
        __m256 f32_vec2 = _mm256_cvtepi32_ps(i32_vec2);
        f32_vec2 = _mm256_mul_ps(f32_vec2, scale);

        // Store results (check alignment)
        if (reinterpret_cast<uintptr_t>(dst + i) % 32 == 0) {
            _mm256_store_ps(dst + i, f32_vec1);
            _mm256_store_ps(dst + i + 8, f32_vec2);
        } else {
            _mm256_storeu_ps(dst + i, f32_vec1);
            _mm256_storeu_ps(dst + i + 8, f32_vec2);
        }
    }

    // Handle remaining samples
    convert_int16_to_float_scalar(src + simd_count, dst + simd_count, count - simd_count);
}
#endif

#ifdef __SSE2__
void websocket_audio_source::convert_int16_to_float_sse2(const int16_t* src,
                                                         float* dst,
                                                         size_t count) {
    const __m128 scale = _mm_set1_ps(1.0f / 32768.0f);
    size_t simd_count = count & ~7;  // Process 8 samples at a time (2 SSE2 ops)

    for (size_t i = 0; i < simd_count; i += 8) {
        // First 4 samples
        __m128i i32_vec1 = _mm_set_epi32(
            static_cast<int32_t>(src[i + 3]),
            static_cast<int32_t>(src[i + 2]),
            static_cast<int32_t>(src[i + 1]),
            static_cast<int32_t>(src[i])
            );
        __m128 f32_vec1 = _mm_cvtepi32_ps(i32_vec1);
        f32_vec1 = _mm_mul_ps(f32_vec1, scale);

        // Second 4 samples
        __m128i i32_vec2 = _mm_set_epi32(
            static_cast<int32_t>(src[i + 7]),
            static_cast<int32_t>(src[i + 6]),
            static_cast<int32_t>(src[i + 5]),
            static_cast<int32_t>(src[i + 4])
            );
        __m128 f32_vec2 = _mm_cvtepi32_ps(i32_vec2);
        f32_vec2 = _mm_mul_ps(f32_vec2, scale);

        _mm_storeu_ps(dst + i, f32_vec1);
        _mm_storeu_ps(dst + i + 4, f32_vec2);
    }

    convert_int16_to_float_scalar(src + simd_count, dst + simd_count, count - simd_count);
}
#endif

#ifdef __ARM_NEON
void websocket_audio_source::convert_int16_to_float_neon(const int16_t* src,
                                                         float* dst,
                                                         size_t count) {
    const float32x4_t scale = vdupq_n_f32(1.0f / 32768.0f);
    size_t simd_count = count & ~7;  // Process 8 samples at a time

    for (size_t i = 0; i < simd_count; i += 8) {
        // Load 8 int16 values
        int16x8_t i16_vec = vld1q_s16(src + i);

        // Split into two sets of 4
        int16x4_t i16_low = vget_low_s16(i16_vec);
        int16x4_t i16_high = vget_high_s16(i16_vec);

        // Convert to int32
        int32x4_t i32_low = vmovl_s16(i16_low);
        int32x4_t i32_high = vmovl_s16(i16_high);

        // Convert to float and scale
        float32x4_t f32_low = vcvtq_f32_s32(i32_low);
        float32x4_t f32_high = vcvtq_f32_s32(i32_high);
        f32_low = vmulq_f32(f32_low, scale);
        f32_high = vmulq_f32(f32_high, scale);

        // Store
        vst1q_f32(dst + i, f32_low);
        vst1q_f32(dst + i + 4, f32_high);
    }

    convert_int16_to_float_scalar(src + simd_count, dst + simd_count, count - simd_count);
}
#endif

void websocket_audio_source::convert_int16_to_float_scalar(const int16_t* src,
                                                           float* dst,
                                                           size_t count) {
    const float scale = 1.0f / 32768.0f;

    // Unroll by 8 for better instruction-level parallelism
    size_t unroll_count = count & ~7;

    for (size_t i = 0; i < unroll_count; i += 8) {
        dst[i]     = src[i]     * scale;
        dst[i + 1] = src[i + 1] * scale;
        dst[i + 2] = src[i + 2] * scale;
        dst[i + 3] = src[i + 3] * scale;
        dst[i + 4] = src[i + 4] * scale;
        dst[i + 5] = src[i + 5] * scale;
        dst[i + 6] = src[i + 6] * scale;
        dst[i + 7] = src[i + 7] * scale;
    }

    for (size_t i = unroll_count; i < count; ++i) {
        dst[i] = src[i] * scale;
    }
}

int64_t websocket_audio_source::get_current_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Batch operations for better throughput
size_t websocket_audio_source::enqueue_batch(const float* data, size_t count) {
    size_t enqueued = 0;

    // Try to push in batches for better performance
    while (count > 0) {
        size_t batch_size = std::min(count, QUEUE_BATCH_SIZE);
        size_t pushed = m_audio_queue.push(data, batch_size);

        if (pushed == 0) {
            break;  // Queue is full
        }

        enqueued += pushed;
        data += pushed;
        count -= pushed;

        if (pushed < batch_size) {
            break;  // Queue is getting full
        }
    }

    return enqueued;
}

size_t websocket_audio_source::dequeue_batch(float* data, size_t count) {
    // Boost's pop() can retrieve multiple elements efficiently
    return m_audio_queue.pop(data, count);
}

websocket_audio_source::websocket_audio_source() {
    m_last_packet_time_ms = get_current_time_ms();

    // Pre-warm the thread-local buffers to avoid first-call allocation
    get_conversion_buffer(CHUNK_SIZE);
    get_batch_buffer();
}

websocket_audio_source::~websocket_audio_source() {
    stop();
}

bool websocket_audio_source::initialize() {
    // Reset the queue (Boost's SPSC queue doesn't need explicit initialization)
    m_audio_queue.reset();
    return true;
}

bool websocket_audio_source::start() {
    m_active = true;
    m_audio_queue.reset();  // Clear any existing data
    m_total_samples_processed = 0;
    m_total_samples_dropped = 0;
    return true;
}

void websocket_audio_source::stop() {
    m_active = false;
    m_audio_queue.reset();

    // Clear metadata queue
    AudioMetadata metadata;
    while (m_metadata_queue.try_dequeue(metadata)) {
        // Discard
    }

    // Log statistics if any samples were processed
    size_t processed = m_total_samples_processed.load();
    size_t dropped = m_total_samples_dropped.load();
    if (processed > 0) {
        double drop_rate = dropped > 0 ? (100.0 * dropped / (processed + dropped)) : 0.0;
        std::cout << "[WebSocket Audio] Stopped. Processed: " << processed
                  << " samples, Dropped: " << dropped
                  << " (" << drop_rate << "%)" << std::endl;
    }
}

bool websocket_audio_source::get_audio_samples(std::vector<float>& samples) {
    if (!m_active) return false;

    // Process metadata updates
    AudioMetadata metadata;
    while (m_metadata_queue.try_dequeue(metadata)) {
        m_current_session_id = std::move(metadata.session_id);
        m_current_language = std::move(metadata.language);
        m_last_packet_time_ms = get_current_time_ms();
    }

    // Use Boost's available() for quick check
    size_t available = m_audio_queue.read_available();

    // Check if we have enough samples for a full chunk
    if (available >= CHUNK_SIZE) {
        samples.resize(CHUNK_SIZE);
        size_t dequeued = dequeue_batch(samples.data(), CHUNK_SIZE);
        samples.resize(dequeued);
        return dequeued > 0;
    }

    // Check if we should flush partial buffer
    int64_t current_time = get_current_time_ms();
    int64_t time_since_last_packet = current_time - m_last_packet_time_ms.load(std::memory_order_relaxed);

    if (time_since_last_packet > FLUSH_TIMEOUT_MS && available >= MIN_CHUNK_SIZE) {
        samples.resize(available);
        size_t dequeued = dequeue_batch(samples.data(), available);
        samples.resize(dequeued);
        return dequeued > 0;
    }

    return false;
}

bool websocket_audio_source::is_active() const {
    if (!m_active) return false;

    int64_t current_time = get_current_time_ms();
    int64_t last_packet_time = m_last_packet_time_ms.load(std::memory_order_relaxed);
    return (current_time - last_packet_time) < ACTIVITY_TIMEOUT_MS;
}

void websocket_audio_source::handle_audio_data(const std::vector<int16_t>& pcm_samples,
                                               const std::string& session_id,
                                               const std::string& language) {
    if (!m_active || pcm_samples.empty()) return;

    // Dump audio if enabled
    if (m_dump_enabled.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(m_dump_mutex);
        if (m_audio_dump_file.is_open()) {
            m_audio_dump_file.write(reinterpret_cast<const char*>(pcm_samples.data()),
                                    pcm_samples.size() * sizeof(int16_t));
            m_total_samples_dumped.fetch_add(pcm_samples.size(), std::memory_order_relaxed);
        }
    }

    // Get aligned conversion buffer
    AlignedBuffer* conv_buffer = get_conversion_buffer(pcm_samples.size());

    // SIMD-optimized conversion
    convert_int16_to_float_simd(pcm_samples.data(), conv_buffer->data, pcm_samples.size());

    // Enqueue using batch operation
    size_t enqueued = enqueue_batch(conv_buffer->data, pcm_samples.size());

    // Update statistics
    m_total_samples_processed.fetch_add(enqueued, std::memory_order_relaxed);

    if (enqueued < pcm_samples.size()) {
        size_t dropped = pcm_samples.size() - enqueued;
        m_total_samples_dropped.fetch_add(dropped, std::memory_order_relaxed);

        // Rate-limited warning
        static std::atomic<int64_t> last_warning_time{0};
        int64_t current_time = get_current_time_ms();
        int64_t last_warning = last_warning_time.load(std::memory_order_relaxed);

        if (current_time - last_warning > 5000) {
            last_warning_time.store(current_time, std::memory_order_relaxed);
            size_t queue_size = m_audio_queue.read_available();
            std::cerr << "[WebSocket Audio] Buffer overflow! Dropped " << dropped
                      << " samples. Queue: " << queue_size << "/" << MAX_BUFFER_SIZE
                      << " (" << (100.0 * queue_size / MAX_BUFFER_SIZE) << "% full)"
                      << std::endl;
        }
    }

    // Queue metadata
    if (enqueued > 0) {
        AudioMetadata metadata{
            session_id,
            language,
            static_cast<uint64_t>(get_current_time_ms()),
            enqueued
        };
        m_metadata_queue.enqueue(std::move(metadata));
    }
}

void websocket_audio_source::enable_audio_dump(const std::string& filename) {
    std::lock_guard<std::mutex> lock(m_dump_mutex);
    if (m_audio_dump_file.is_open()) {
        m_audio_dump_file.close();
    }
    m_audio_dump_file.open(filename, std::ios::binary);
    m_dump_enabled.store(m_audio_dump_file.is_open(), std::memory_order_relaxed);
    m_total_samples_dumped.store(0, std::memory_order_relaxed);

    if (m_dump_enabled) {
        std::cout << "[WebSocket Audio] Audio dump enabled to file: " << filename << std::endl;
    }
}

void websocket_audio_source::disable_audio_dump() {
    m_dump_enabled.store(false, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(m_dump_mutex);
    if (m_audio_dump_file.is_open()) {
        m_audio_dump_file.close();
        std::cout << "[WebSocket Audio] Audio dump disabled. Total samples dumped: "
                  << m_total_samples_dumped.load(std::memory_order_relaxed) << std::endl;
    }
}
