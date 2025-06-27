cmake_minimum_required(VERSION 3.16)
project(wstream_client)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0167 NEW)
# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find required packages
find_package(Boost REQUIRED COMPONENTS system)
find_package(nlohmann_json REQUIRED)

# Define source files
set(SOURCES
    main.cpp
    audio_recorder.cpp
    websocket_client.cpp
    base64.cpp
)

# Multi-method PortAudio detection
# Method 1: Try standard CMake find_package
find_package(PortAudio QUIET)

# Method 2: Try pkg-config if Method 1 failed
if(NOT PortAudio_FOUND)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(PORTAUDIO QUIET portaudio-2.0)
    endif()
endif()

# Method 3: Manual search if Methods 1 and 2 failed
if(NOT PortAudio_FOUND AND NOT PORTAUDIO_FOUND)
    message(STATUS "Searching for PortAudio manually...")

    # Define search paths based on platform
    if(APPLE)
        set(PORTAUDIO_SEARCH_PATHS
            /usr/local
            /opt/homebrew
            /opt/local
            ${CMAKE_SOURCE_DIR}/external/portaudio
        )
    elseif(UNIX)
        set(PORTAUDIO_SEARCH_PATHS
            /usr
            /usr/local
            ${CMAKE_SOURCE_DIR}/external/portaudio
        )
    elseif(WIN32)
        set(PORTAUDIO_SEARCH_PATHS
            "C:/Program Files/PortAudio"
            "C:/Program Files (x86)/PortAudio"
            ${CMAKE_SOURCE_DIR}/external/portaudio
        )
    endif()

    # Find headers
    find_path(PORTAUDIO_INCLUDE_DIR
        NAMES portaudio.h
        PATHS ${PORTAUDIO_SEARCH_PATHS}
        PATH_SUFFIXES include
    )

    # Find library
    find_library(PORTAUDIO_LIBRARIES
        NAMES portaudio
        PATHS ${PORTAUDIO_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64
    )

    if(PORTAUDIO_INCLUDE_DIR AND PORTAUDIO_LIBRARIES)
        set(PORTAUDIO_FOUND TRUE)
        message(STATUS "PortAudio found manually: ${PORTAUDIO_LIBRARIES}")
    endif()
endif()

# Check if PortAudio was found by any method
if(PortAudio_FOUND)
    message(STATUS "PortAudio found via find_package: ${PortAudio_LIBRARIES}")
    set(PORTAUDIO_INCLUDE_DIR ${PortAudio_INCLUDE_DIRS})
    set(PORTAUDIO_LIBRARIES ${PortAudio_LIBRARIES})
    set(PORTAUDIO_FOUND TRUE)
elseif(PORTAUDIO_FOUND)
    message(STATUS "PortAudio found via pkg-config or manual search")
else()
    message(FATAL_ERROR "PortAudio not found. Please install PortAudio development files:
    - macOS: brew install portaudio
    - Linux: sudo apt-get install portaudio19-dev
    - Windows: Download from http://www.portaudio.com/")
endif()

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PORTAUDIO_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    Boost::system
    nlohmann_json::nlohmann_json
    ${PORTAUDIO_LIBRARIES}
)

# Platform-specific settings
if(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE "-framework CoreFoundation" "-framework CoreAudio")
elseif(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
elseif(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE ws2_32 winmm)
endif()

# Compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Set output name to match expected name
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "wstream_client")

# Print summary
message(STATUS "Build configuration:")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  PortAudio: ${PORTAUDIO_LIBRARIES}")
message(STATUS "  Boost: ${Boost_LIBRARIES}")
message(STATUS "  nlohmann_json: ${nlohmann_json_DIR}")

# Create install target
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
