cmake_minimum_required(VERSION 3.15)
project(wstream)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0167 OLD)

# Find required packages
find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(SDL2 REQUIRED)
find_package(Threads REQUIRED)
find_package(hyni_websocket_server REQUIRED)
find_path(CONCURRENTQUEUE_INCLUDE_DIR
    NAMES moodycamel/concurrentqueue.h
    PATHS /usr/include/concurrentqueue
          /usr/local/include/concurrentqueue
          /usr/include
          /usr/local/include
)

if(NOT CONCURRENTQUEUE_INCLUDE_DIR)
    if(EXISTS /etc/fedora-release)
        message(FATAL_ERROR
            "Could not find moodycamel/concurrentqueue.h\n"
            "On Fedora, install it with:\n"
            "  sudo dnf install moodycamel-concurrentqueue-devel\n"
            "Searched in: /usr/include/concurrentqueue")
    else()
        message(FATAL_ERROR
            "Could not find moodycamel/concurrentqueue.h\n"
            "Please install the concurrentqueue development headers.\n"
            "Searched in: /usr/include/concurrentqueue")
    endif()
endif()

# Option to build tests
option(BUILD_TESTS "Build test suite" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)
include_directories(${SDL2_INCLUDE_DIRS})

# Common SDL library
set(TARGET common-sdl)
add_library(${TARGET} STATIC
    common-sdl.h
    common-sdl.cpp
)

target_include_directories(${TARGET} PUBLIC
    ${SDL2_INCLUDE_DIRS}
    ${CONCURRENTQUEUE_INCLUDE_DIR}
)
target_link_libraries(${TARGET} PRIVATE ${SDL2_LIBRARIES})

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${TARGET} PROPERTIES FOLDER "libs")

# Main application library (for testing)
set(LIB_TARGET wstream_lib)

set(LIB_SOURCES
    wstream_app.cpp
    sdl_audio_source.cpp
    whisper_engine.cpp
    text_processor.cpp
    websocket_audio_source.cpp
    audio_source_factory.cpp
    base64.cpp
    benchmark_audio_source.cpp
    benchmark_manager.cpp
    transcription_marker.cpp
)

set(LIB_HEADERS
    wstream_app.h
    sdl_audio_source.h
    whisper_engine.h
    text_processor.h
    audio_source.h
    websocket_audio_source.h
    audio_source_factory.h
    base64.h
    benchmark_audio_source.h
    benchmark_manager.h
    transcription_marker.h
    audio_playback_manager.h
)

add_library(${LIB_TARGET} STATIC ${LIB_SOURCES} ${LIB_HEADERS})

target_link_libraries(${LIB_TARGET} PUBLIC
    common
    common-sdl
    whisper
    hyni_websocket_server::hyni_websocket_server
    nlohmann_json::nlohmann_json
    Boost::system
    Threads::Threads
)

# Main executable
set(TARGET wstream)
add_executable(${TARGET} main.cpp)

target_link_libraries(${TARGET} PRIVATE ${LIB_TARGET})

install(TARGETS ${TARGET} RUNTIME)

# Tests
if(BUILD_TESTING)
    enable_testing()

    # Find Google Test
    find_package(GTest REQUIRED)

    # Test executable
    set(TEST_TARGET wstream_tests)

    set(TEST_SOURCES
        tests/test_text_processor.cpp
        tests/test_whisper_engine.cpp
        tests/test_audio_source.cpp
        )

    add_executable(${TEST_TARGET} ${TEST_SOURCES})

    target_link_libraries(${TEST_TARGET} PRIVATE
        ${LIB_TARGET}
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
    )

    target_include_directories(${TEST_TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Add tests to CTest
    include(GoogleTest)
    gtest_discover_tests(${TEST_TARGET}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        PROPERTIES TIMEOUT 120
    )

    # Custom target to run tests
    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS ${TEST_TARGET}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# Optional: Coverage target
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND BUILD_TESTS)
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)

    if(LCOV AND GENHTML)
        add_custom_target(coverage
            COMMAND ${LCOV} --directory . --zerocounters
            COMMAND ${TEST_TARGET}
            COMMAND ${LCOV} --directory . --capture --output-file coverage.info
            COMMAND ${LCOV} --remove coverage.info '/usr/*' '*/tests/*' --output-file coverage.info.cleaned
            COMMAND ${GENHTML} coverage.info.cleaned --output-directory coverage_report
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            DEPENDS ${TEST_TARGET}
        )
    endif()
endif()
