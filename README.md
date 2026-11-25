# wstream
Whisper-Stream 

A simplified, fine-tuned ASR project based on an example of whisper.cpp project

## Vendored Whisper sources

Synced from the local `../whisper.cpp` fork at commit
`76684141a5d059be71cbe23dc2f0ed552213ba2d` (Whisper 1.8.4, GGML 0.9.8).
Use the commit in that checkout as
the source of truth; wstream has its own application and CMake layout.

The native configurations use the shared Whisper/GGML engine, the CPU backend,
and optionally CUDA and OpenBLAS. For future syncs, compare these groups:

- `include/`, `src/whisper.cpp`, `src/whisper-arch.h`, and the relevant library
  settings in `src/CMakeLists.txt`.
- The common helpers built by `common/CMakeLists.txt`, mapped from the fork's
  `examples/` directory, including their audio decoder headers.
- `ggml/CMakeLists.txt`, `ggml/cmake/`, `ggml/include/`, and files directly in
  `ggml/src/`.
- `ggml/src/ggml-cpu/` shared files, `amx/` helpers, and `arch/x86/`; retain
  `arch/wasm/` for the browser build.
- `ggml/src/ggml-cuda/` and `ggml/src/ggml-blas/` for the GPU configuration.

Other accelerator implementations, CoreML/OpenVINO adapters, upstream apps,
bindings, models, and samples are outside these native build configurations.
Keep wstream's `stream/`, browser apps, and root/common CMake integration.
Unchanged files need no copy.

Local patches to retain when comparing with the fork:

- `ggml/src/ggml-alloc.c`: free-block bounds and allocation-failure checks.
- `ggml/src/ggml-cuda/common.cuh` and `ggml-cuda.cu`: allocation-size guards.
- `ggml/src/ggml-cuda/CMakeLists.txt`: narrowed FlashAttention template globs
  that exclude obsolete template files left by earlier manual syncs.
- `ggml/src/ggml-cuda/argsort.cu` and `top-k.cu`: explicit CCCL headers for
  the CUDA 13.4 toolkit, following
  [CCCL's direct-include guidance](https://github.com/NVIDIA/cccl).
- The MSVC warning pragmas in the common helpers.

Whisper's version is supplied by `WHISPER_VERSION` in the root CMake file,
independently of wstream's application version.

## Native builds

Dependencies: CMake, Ninja, a C/C++ compiler, SDL2, Boost headers, and
`nlohmann_json`. The CUDA configuration also needs the CUDA toolkit and
OpenBLAS.

The existing `build/CMakeCache.txt` records this GPU configuration for a
Ryzen AI 9 HX 370 and RTX 4060 Laptop GPU (CUDA architecture 89):

```sh
cmake -S . -B build/cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA=ON \
  -DGGML_NATIVE=ON -DGGML_LTO=ON -DGGML_OPENMP=ON \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON \
  -DGGML_AVX512_BF16=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX_VNNI=ON
cmake --build build/cuda --target wstream -j 4
```

`GGML_CUDA_F16` was present in the old command/cache but is unused by the
current GGML sources, so it is omitted here.

The existing `build-cpu/CMakeCache.txt` records this CPU configuration:

```sh
cmake -S . -B build/cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=ON -DGGML_LTO=ON \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_BMI2=ON \
  -DGGML_FMA=ON -DGGML_F16C=ON
cmake --build build/cpu --target wstream -j 6
```

Root CMake additionally applies `-O3`, `-march=znver5` on the detected Ryzen,
`-flto`, `-fPIC`, and `-fomit-frame-pointer` in Release. Root flags add
`-std=c++20`; GGML separately declares C++17 for its targets.
Its architecture tuning applies even when `GGML_NATIVE=OFF`; the CPU command
therefore does not produce a portable binary for older CPUs.

Use a fresh build directory if an old cache references another checkout path.
Run from the repository root; the model path is the first positional argument:

```sh
./build/cuda/bin/wstream models/ggml-small.en-q5_1.bin
```

The WebSocket endpoint defaults to `ws://127.0.0.1:8765`; override it with
`WSTREAM_BIND` and `WSTREAM_PORT`. Privileged commands require
`WSTREAM_CONTROL_TOKEN`.
