set(TARGET libstream)

add_executable(${TARGET}
    emscripten.cpp
    hyni_merge.h
)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE
    whisper
)

set_target_properties(${TARGET} PROPERTIES LINK_FLAGS " \
    --bind \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE=8 \
    -s INITIAL_MEMORY=1024MB \
    -s MAXIMUM_MEMORY=4096MB \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s FORCE_FILESYSTEM=1 \
    -s EXPORTED_RUNTIME_METHODS=\"['print', 'printErr', 'ccall', 'cwrap', 'FS_createDataFile']\" \
    ${EXTRA_FLAGS} \
    ")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/helpers.js ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/helpers.js COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index.html ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/index.html COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/server.py ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/server.py COPYONLY)
