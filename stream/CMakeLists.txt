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

target_include_directories(${TARGET} PUBLIC ${SDL2_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE ${SDL2_LIBRARIES})

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${TARGET} PROPERTIES FOLDER "libs")

# Main application library (for testing)
set(LIB_TARGET wstream_lib)

set(LIB_SOURCES
    wstream_app.cpp
    audio_processor.cpp
    whisper_engine.cpp
    text_processor.cpp
    websocket_server.cpp
    websocket_audio_source.cpp
    audio_source_factory.cpp
    base64.cpp
)

set(LIB_HEADERS
    wstream_app.h
    audio_processor.h
    whisper_engine.h
    text_processor.h
    websocket_server.h
    audio_source.h
    websocket_audio_source.h
    audio_source_factory.h
    base64.h
)

add_library(${LIB_TARGET} STATIC ${LIB_SOURCES} ${LIB_HEADERS})

target_link_libraries(${LIB_TARGET} PUBLIC
    common
    common-sdl
    whisper
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
if(BUILD_TESTS)
    enable_testing()

    # Find Google Test
    find_package(GTest REQUIRED)

    # Test executable
    set(TEST_TARGET wstream_tests)

    set(TEST_SOURCES
        tests/test_text_processor.cpp
        tests/test_audio_processor.cpp
        tests/test_whisper_engine.cpp
        tests/test_websocket_server.cpp
        tests/test_wstream_app.cpp
        tests/test_audio_source.cpp
        tests/test_websocket_audio_source.cpp
        tests/test_audio_processor_source.cpp
        tests/test_websocket_sever_command.cpp
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
