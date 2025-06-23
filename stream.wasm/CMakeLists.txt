cmake_minimum_required(VERSION 3.20)
project(wstream_wasm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Only build if using Emscripten
if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "This CMakeLists.txt is for Emscripten builds only. Use emcmake/emmake.")
    return()
endif()

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist)

# Emscripten settings
set(CMAKE_EXECUTABLE_SUFFIX ".js")

# Separate compile and link flags
set(EMCC_COMPILE_FLAGS "")
set(EMCC_LINK_FLAGS "")

# Build type specific flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND EMCC_COMPILE_FLAGS "-O0" "-g")
    list(APPEND EMCC_LINK_FLAGS "-s ASSERTIONS=2" "-s SAFE_HEAP=1" "-s STACK_OVERFLOW_CHECK=1")
else()
    list(APPEND EMCC_COMPILE_FLAGS
        "-O3"
        "-flto"  # Link-time optimization
        "-ffast-math"  # Faster math
        "-DNDEBUG"
        "-msimd128"
    )
endif()

# Link-only flags
list(APPEND EMCC_LINK_FLAGS
    "-s WASM=1"
    "-s USE_PTHREADS=1"
    "-s PTHREAD_POOL_SIZE=8"
    "-s ALLOW_MEMORY_GROWTH=1"
    "-s INITIAL_MEMORY=256MB"
    "-s MAXIMUM_MEMORY=2GB"
    "-s STACK_SIZE=5MB"
    "-s FORCE_FILESYSTEM=1"
    "-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS']"
    "-s MODULARIZE=1"
    "-s EXPORT_NAME='WStreamModule'"
    "-s FILESYSTEM=1"
    "-s NODERAWFS=0"
    "-s ENVIRONMENT='web,worker'"
    "--bind"
)

# Apply compile flags
string(REPLACE ";" " " EMCC_COMPILE_FLAGS_STR "${EMCC_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EMCC_COMPILE_FLAGS_STR}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EMCC_COMPILE_FLAGS_STR}")

# Apply link flags
string(REPLACE ";" " " EMCC_LINK_FLAGS_STR "${EMCC_LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EMCC_LINK_FLAGS_STR}")

# Source files
set(WASM_SOURCES
    src/wstream_app_wasm.cpp
    src/bindings.cpp
)

# Add stream library sources (only what we need)
set(STREAM_SOURCES
    ../stream/whisper_engine.cpp
    ../stream/text_processor.cpp
)

# Create executable
add_executable(${PROJECT_NAME}
    ${WASM_SOURCES}
    ${STREAM_SOURCES}
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../stream
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link with libraries built as part of the main project
target_link_libraries(${PROJECT_NAME} PRIVATE
    common
    whisper
)

# Post-build: copy web files to dist
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/web
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMENT "Copying web files to dist/"
)

# Post-build: create models directory in dist
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models
    COMMENT "Creating models directory in dist/"
)

set(FILES_TO_COPY ggml-tiny.en-q5_1.bin ggml-tiny.en.bin ggml-base.en.bin ggml-base.en-q5_1.bin)
foreach(f ${FILES_TO_COPY})
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/models/${f}
                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/${f}
    )
endforeach()
