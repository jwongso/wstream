cmake_minimum_required(VERSION 3.15)
project(wstream)

cmake_policy(SET CMP0167 OLD)

find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(SDL2 REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)
include_directories(${SDL2_INCLUDE_DIRS})

set(TARGET wstream)
add_executable(${TARGET} stream.cpp chatapi.cpp chatapi.h config.h)

target_link_libraries(${TARGET} PRIVATE
    common
    common-sdl
    whisper
    curl
    nlohmann_json::nlohmann_json
    Boost::system
    ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)
