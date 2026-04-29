# Benchmarks — hippo_leap vs llama.cpp

## Llama 3.2 1B Instruct (Q4_K_M)

**Date**: 2026-04-28

### Setup

- **Model**: Llama 3.2 1B Instruct Q4_K_M (762.8 MiB, 1.24B params)
- **GPU**: AMD Radeon 8060S (Strix Halo, RDNA3.5, gfx1151)
- **OS**: NixOS, Linux 6.12.78

### Compile modes

**hippo_leap** (Nim + hippo, hand-written HIP kernels):
```
nim cpp --cc:hipcc -d:release -d:useMalloc -d:backendNaive -d:zippyNoSimd
```

**llama.cpp** (C++, Vulkan backend, cooperative matrix):
```
cmake -B build-vulkan -DGGML_VULKAN=ON && cmake --build build-vulkan
llama-bench -ngl 99
```

### Results

| Metric            | llama.cpp Vulkan (tok/s) | hippo_leap HIP (tok/s) |   Delta |
|:------------------|-------------------------:|-----------------------:|--------:|
| Prompt (pp128)    |          5534.08 ± 94.62 |                      — |       — |
| Decode (tg64)     |            212.89 ± 0.67 |           35.7 (mean)  |  -83.2% |

hippo_leap decode: 3 runs, 1 warmup, 64 tokens. Min 34.1, max 37.4 tok/s.

### Notes

hippo_leap is at ~17% of llama.cpp Vulkan decode throughput. Key differences:
- llama.cpp uses cooperative matrix extensions (KHR_coopmat), hippo_leap uses scalar warp-per-row kernels
- llama.cpp has optimized prompt processing (batched GEMM), hippo_leap processes tokens sequentially during prefill
- No FlashAttention in hippo_leap yet — using shared-memory attention kernel

---

## TinyLlama 1.1B (Q2_K) — historical

**Date**: 2025 (pre-hippo_leap, standalone tinylama binary)

### Setup

- **Model**: TinyLlama 1.1B Chat v1.0 (Q2_K GGUF, ~459 MiB)
- **GPU**: AMD Radeon 7900 XTX (RDNA3, 24GB, 960 GB/s)
- **OS**: NixOS, Linux 6.19.9-zen1, ROCm
- **Benchmark**: 3 warmup runs, 10 sample runs, avg tok/s ± stdev

### Compile modes

**tinylama** (Nim + hippo, hand-written HIP kernels):
```
HIP_PLATFORM=amd nim cpp -d:release --cc:hipcc -d:useHippo -d:useMalloc --path:../hippo/src
```

**llama.cpp** (C++, ROCm backend, hipBLAS + FlashAttention):
```
nix build '.#rocm'   # from llama.cpp flake
llama-bench -ngl 99
```

### Results

| Decode Tokens | tinylama (tok/s) | llama.cpp (tok/s) |  Delta |
|:-------------:|-----------------:|------------------:|-------:|
|            32 |   363.04 ± 7.12  |   353.28 ± 5.94   | +2.8%  |
|           128 |   360.39 ± 10.38 |   349.56 ± 0.72   | +3.1%  |
|           256 |   346.54 ± 11.24 |   349.69 ± 2.78   | -0.9%  |
|           512 |   325.66 ± 1.96  |   339.50 ± 1.25   | -4.1%  |

### Analysis

tinylama wins at short sequences (32-128 tokens) where kernel launch
overhead dominates — our fused kernels eliminate more launches than
llama.cpp. As sequences grow (256-512), llama.cpp's FlashAttention
scales better than our shared-memory attention kernel, and the gap
reverses.

The crossover point is around 256 tokens — suggesting the attention
kernel is the primary target for further optimization at longer contexts.
