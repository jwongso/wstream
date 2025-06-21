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

#pragma once

#include <string>
#include <string_view>

/**
 * @file text_processor.h
 * @brief Text processing and filtering for speech recognition output
 * @author WStream Development Team
 * @version 1.0
 * @date 2024
 */

/**
 * @class text_processor
 * @brief Processes and cleans transcribed text from speech recognition
 *
 * This class provides text processing capabilities to clean and filter
 * the raw output from speech recognition systems. Common processing includes:
 * - Removing transcription artifacts (brackets, parentheses)
 * - Trimming whitespace
 * - Filtering out non-speech elements
 * - Standardizing text format
 *
 * The processor is designed to be fast and memory-efficient for real-time
 * applications, using string views where possible to avoid unnecessary copies.
 *
 * @par Processing Features:
 * - **Bracket Removal**: Removes [bracketed] content (often transcription notes)
 * - **Parentheses Removal**: Removes (parenthetical) content (often noise indicators)
 * - **Whitespace Trimming**: Removes leading/trailing whitespace
 * - **Configurable**: Each feature can be enabled/disabled independently
 *
 * @par Performance:
 * - Uses efficient single-pass algorithms
 * - Minimizes memory allocations
 * - Optimized for streaming text processing
 *
 * @par Thread Safety:
 * This class is thread-safe for read operations. Multiple threads can
 * safely call process() simultaneously on the same instance.
 */
class text_processor {
public:
    /// Default bracket removal setting
    static constexpr bool DEFAULT_REMOVE_BRACKETS = true;

    /// Default parentheses removal setting
    static constexpr bool DEFAULT_REMOVE_PARENTHESES = true;

    /// Default whitespace trimming setting
    static constexpr bool DEFAULT_TRIM_WHITESPACE = true;

    /**
     * @struct config
     * @brief Configuration options for text processing
     */
    struct config {
        /// Remove content within square brackets [like this]
        bool remove_brackets;

        /// Remove content within parentheses (like this)
        bool remove_parentheses;

        /// Trim leading and trailing whitespace
        bool trim_whitespace;

        /**
         * @brief Default constructor with standard cleaning enabled
         *
         * Enables all common text cleaning operations that are typically
         * useful for speech recognition output:
         * - Bracket removal (transcription artifacts)
         * - Parentheses removal (noise indicators)
         * - Whitespace trimming (formatting cleanup)
         */
        config()
            : remove_brackets(DEFAULT_REMOVE_BRACKETS)
            , remove_parentheses(DEFAULT_REMOVE_PARENTHESES)
            , trim_whitespace(DEFAULT_TRIM_WHITESPACE) {}
    };

    /**
     * @brief Constructs text processor with specified configuration
     * @param cfg Processing configuration options
     */
    explicit text_processor(const config& cfg = config{}) : m_config(cfg) {}

    /**
     * @brief Processes input text according to configuration
     * @param input Text to process (string_view for efficiency)
     * @return Processed text string
     *
     * Applies all enabled processing operations to the input text:
     * 1. Removes bracketed content (if enabled)
     * 2. Removes parenthetical content (if enabled)
     * 3. Trims whitespace (if enabled)
     *
     * @par Example:
     * @code
     * text_processor processor;
     * std::string result = processor.process("  [NOISE] Hello (cough) world!  ");
     * // Result: "Hello world!"
     * @endcode
     *
     * @par Performance:
     * - Uses single-pass algorithm where possible
     * - Pre-allocates result buffer to minimize reallocations
     * - Optimized for typical speech recognition artifacts
     */
    std::string process(std::string_view input) const;

private:
    /// Processing configuration
    config m_config;

    /**
     * @brief Removes bracketed and parenthetical content from text
     * @param text Text to process (modified in-place)
     *
     * Efficiently removes content within brackets [] and/or parentheses ()
     * based on configuration settings. Uses a single-pass algorithm that
     * tracks nesting state and builds the result string incrementally.
     *
     * @par Algorithm:
     * - Single pass through the input string
     * - State machine tracking bracket/parentheses nesting
     * - Pre-allocated output buffer for efficiency
     * - Handles nested structures correctly
     */
    void remove_bracketed_text(std::string& text) const;

    /**
     * @brief Removes leading and trailing whitespace
     * @param text Text to trim (modified in-place)
     *
     * Efficiently removes whitespace characters from the beginning and
     * end of the string using standard library algorithms optimized
     * for this common operation.
     *
     * @par Whitespace Definition:
     * Uses std::isspace() to identify whitespace characters, which includes:
     * - Space (' ')
     * - Tab ('\t')
     * - Newline ('\n')
     * - Carriage return ('\r')
     * - Form feed ('\f')
     * - Vertical tab ('\v')
     */
    void trim_whitespace(std::string& text) const;
};
