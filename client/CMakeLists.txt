cmake_minimum_required(VERSION 3.16)
project(wstream_client)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0167 NEW)
# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(nlohmann_json REQUIRED)

# Find ALSA
pkg_check_modules(ALSA REQUIRED alsa)

# Find WebSocket++ (header-only)
find_path(WEBSOCKETPP_INCLUDE_DIR
    NAMES websocketpp/config/asio_no_tls.hpp
    PATHS /usr/include /usr/local/include
)

if(NOT WEBSOCKETPP_INCLUDE_DIR)
    message(FATAL_ERROR "WebSocket++ not found. Install with: sudo apt-get install libwebsocketpp-dev")
endif()

# Create executable
add_executable(${PROJECT_NAME}
    main.cpp
    audio_recorder.cpp
    websocket_client.cpp
    base64.cpp
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${WEBSOCKETPP_INCLUDE_DIR}
    ${ALSA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    Boost::system
    nlohmann_json::nlohmann_json
    ${ALSA_LIBRARIES}
    pthread
)

# Compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    ${ALSA_CFLAGS_OTHER}
    -Wall -Wextra
)
