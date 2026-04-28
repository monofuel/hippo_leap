# tinylama vs llama.cpp — Decode Benchmark

## Setup

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

## Results

| Decode Tokens | tinylama (tok/s) | llama.cpp (tok/s) |  Delta |
|:-------------:|-----------------:|------------------:|-------:|
|            32 |   363.04 ± 7.12  |   353.28 ± 5.94   | +2.8%  |
|           128 |   360.39 ± 10.38 |   349.56 ± 0.72   | +3.1%  |
|           256 |   346.54 ± 11.24 |   349.69 ± 2.78   | -0.9%  |
|           512 |   325.66 ± 1.96  |   339.50 ± 1.25   | -4.1%  |

## Analysis

tinylama wins at short sequences (32-128 tokens) where kernel launch
overhead dominates — our fused kernels eliminate more launches than
llama.cpp. As sequences grow (256-512), llama.cpp's FlashAttention
scales better than our shared-memory attention kernel, and the gap
reverses.

tinylama also shows higher variance (stdev) at shorter decode lengths,
likely due to GPU clock boosting and ROCm scheduler jitter having more
impact when total wall time is small.

The crossover point is around 256 tokens — suggesting the attention
kernel is the primary target for further optimization at longer contexts.
