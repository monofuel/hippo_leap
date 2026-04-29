# Split GPU build: separate .so for inference

## Problem

hipcc compiles all source files as HIP code, which means host-only libraries
(mummy, zippy, curly, etc.) get compiled with the GPU offload compiler. This
causes issues:

- zippy's SSE/PCLMULQDQ SIMD intrinsics fail to compile for GPU targets
  (gfx1100, gfx1151) — currently worked around with `-d:zippyNoSimd`
- All source files get compiled twice (host + device), doubling build time
- Harmless but noisy `array-bounds` warnings from Nim's seq implementation
  under hipcc's stricter clang frontend

## Proposed solution

Split the binary into two compilation units:

1. **`libhippo_leap_gpu.so`** — GPU inference backend, compiled with
   `nim cpp --cc:hipcc --app:lib`. Contains only:
   - `backends/naive.nim` (GPU kernels + orchestrators)
   - `backend_types.nim` (GPU type definitions)
   - Transitive deps: `model.nim`, `tensor.nim`, `quant.nim`, `gguf_loader.nim`

2. **`hippo_leap`** — HTTP server, compiled with `nim cpp` (regular clang++).
   Contains:
   - `server.nim`, `handlers.nim`, `inference.nim`, `config.nim`
   - All of mummy/zippy/curly/jsony (compiled normally with full SIMD)
   - Links against `libhippo_leap_gpu.so` at runtime

## Implementation approach

Use [treeform/genny](https://github.com/treeform/genny) to generate the FFI
boundary between the two compilation units. genny generates C-compatible
wrapper functions and Nim import stubs from annotated Nim procs.

The backend interface is already minimal (4 procs):
```nim
proc loadModelBackend*(m: var Model, hp: HParams)
proc unloadModelBackend*()
proc forwardPrefill*(m: var Model, tokens: seq[int32], cache: var KvCache): Tensor
proc forwardDecode*(m: var Model, token: int32, cache: var KvCache): Tensor
```

Plus `initKvCache`. These would become the .so's exported API.

## Benefits

- HTTP server gets full SIMD optimizations (zippy CRC32, etc.)
- GPU code compiles only once for the target arch
- Build times roughly halved (hipcc only touches GPU code)
- Cleaner separation — could swap GPU backends by swapping .so files at
  runtime instead of compile-time `-d:backendNaive`
- Could even enable runtime backend selection (load different .so per GPU)

## Trade-offs

- Adds a build step and deployment artifact (.so must be alongside the binary)
- FFI boundary means no cross-unit inlining (irrelevant — the GPU kernel
  launches dominate, not the dispatch call)
- Need to handle ABI compatibility for types passed across the boundary
  (Tensor, KvCache contain seqs — may need to flatten to C-compatible structs)
- genny dependency

## When to do this

Not urgent — the `-d:zippyNoSimd` workaround is functional and the build
succeeds. Worth pursuing when:
- Build times become painful (larger models, more backends)
- We want runtime backend selection
- We add backends that use different compilers (e.g., CUDA via nvcc)
