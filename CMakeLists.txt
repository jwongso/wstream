cmake_minimum_required(VERSION 3.14)
project("hyni" C CXX)
project("hyni" VERSION 0.0.1)
include(CheckIncludeFileCXX)

set(SOVERSION 1)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message("Found ccache ${CCACHE_FOUND}")
    message("Using ccache to speed up compilation")
    set(ENV{CCACHE_CPP2} "yes")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_FOUND})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_FOUND})
endif(CCACHE_FOUND)

set(CMAKE_WARN_UNUSED_CLI YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

string(
    APPEND opts
        "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>:-MMD;-Wall;-Wno-deprecated-declarations;-march=alderlake;-O3;-march=native;-std=c++20;-fPIC;-DBOOST_BIND_GLOBAL_PLACEHOLDERS;-flto;-fomit-frame-pointer;-DNDEBUG>"
        "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CXX>>:-MMD;-Wall;-Wno-deprecated-declarations;-O0;-g3;-std=c++20;-D_GLIBCXX_DEBUG;-fPIC;-DBOOST_BIND_GLOBAL_PLACEHOLDERS>"
        "$<$<AND:$<CONFIG:RELWITHDEBINFO>,$<COMPILE_LANGUAGE:CXX>>:-MMD;-Wall;-Wno-deprecated-declarations;-march=alderlake;-O3;-g;-std=c++20;-fPIC;-DBOOST_BIND_GLOBAL_PLACEHOLDERS;-flto;-fomit-frame-pointer>"
        "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:C>>:-MMD;-Wall;-Wno-deprecated-declarations;-march=alderlake;-O3;-fPIC;-DBOOST_BIND_GLOBAL_PLACEHOLDERS;-flto;-fomit-frame-pointer;-DNDEBUG>"
        "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:C>>:-MMD;-Wall;-Wno-deprecated-declarations;-O0;-g3;-D_GLIBCXX_DEBUG;-fPIC;-DBOOST_BIND_GLOBAL_PLACEHOLDERS>"
        "$<$<AND:$<CONFIG:RELWITHDEBINFO>,$<COMPILE_LANGUAGE:C>>:-MMD;-Wall;-Wno-deprecated-declarations;-march=alderlake;-O3;-g;-fPIC;-DBOOST_BIND_GLOBAL_PLACEHOLDERS;-flto;-fomit-frame-pointer>"
)
add_compile_options("${opts}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(ggml)
add_subdirectory(src)
add_subdirectory(stream)

