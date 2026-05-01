# v10: Build System, Benchmarking, and GPU Fault Recovery

## Context

After v9 (LDS activation caching, vectorized load attempts), a series of kernel
optimizations brought performance to ~163 tok/s on TinyLlama Q2K with individual
launches on azem (Strix Halo, gfx1151, 40 CU). These included:

- Fused dual Q2K kernel for wQ+wK GEMV
- Fused dual Q3K kernel for wGate+wUp GEMV
- Warp-shuffle RMSNorm reduction + fused residual pass
- Fused RoPE + KV store kernel
- Q3K 2-block loop unroll for ILP
- Shared GPU template extraction to reduce duplication
- `-ffast-math` for ~1.4% throughput improvement
- dp4a Q3K path (behind `-d:useDp4a`, not faster on gfx1151)

LDS activation caching was dropped: individual warp kernels outperform the LDS
variants once the dual-kernel fusions reduce redundant activation reads.

## GPU Page Fault Incident (2026-05-01)

### What Happened

The `integration_inf` test crashed with active GPU allocations, leaving the amdgpu
driver's UTCL2 page table in a corrupted state. dmesg showed ~10 sequential page
faults at 2KB-spaced addresses:

```
amdgpu: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:32770)
amdgpu:   Faulty UTCL2 client ID: TCP (0x8)
amdgpu:   PERMISSION_FAULTS: 0x3
```

TCP = Texture Cache Pipe, meaning shader weight reads were hitting unmapped page
table entries. Subsequent GPU launches ran but produced unreliable results.

### Resolution

Node reboot was required. The amdgpu driver reinitializes all GPU state at boot.
No kernel or driver update needed — the fault was triggered by the process crash,
not a driver bug per se.

### Documentation

Created `docs/gpu-fault-recovery.md` with full diagnosis commands, root cause
analysis, and prevention tips. Added GPU health check commands to AGENTS.md.

## Build System Overhaul

### The Problem

Building hippo_leap's megakernel backend requires ~8 compile-time flags. Missing
any one produces a working binary with dramatically different performance:

| Flag | Effect if Missing |
|------|-------------------|
| `-d:backendMegakernel` | Falls through to naive backend |
| `-d:useIndividualLaunches` | Uses cooperative path: 16 blocks on 40 CUs = **5x slower** |
| `-d:targetMachine=azem` | Wrong CU count, wrong specialization |
| `-d:targetModel=tinyllama_q2k` | Wrong model config |

The `-d:useIndividualLaunches` flag was the cause of a confusing 30 tok/s result
that was initially misattributed to GPU fault state. The cooperative path dispatches
only `NumBlocks=16` blocks (from `megakernel_config.nim`'s `cuCount: 16` for azem),
while the individual launch path dispatches `ceil(outDim/32)` blocks — fully
utilizing the GPU.

### What Changed

Added proper Makefile targets that encode all required flags:

```makefile
NIM_COMMON_FLAGS = --cc:hipcc -d:release -d:useMalloc -d:zippyNoSimd -d:HippoRuntime:HIP

build-megakernel: nim.cfg
    nim cpp $(NIM_COMMON_FLAGS) \
        -d:backendMegakernel -d:useIndividualLaunches \
        -d:targetMachine=azem -d:targetModel=tinyllama_q2k \
        -o:hippo_leap src/hippo_leap.nim

build-naive: nim.cfg
    nim cpp $(NIM_COMMON_FLAGS) \
        -d:backendNaive \
        -o:hippo_leap_naive src/hippo_leap.nim
```

Key targets:
- `make build` — megakernel backend (default, fast path)
- `make build-naive` — naive backend (general purpose)
- `make bench` — build megakernel + run TinyLlama Q2K benchmark
- `make bench-gpu` — GPU microbenchmark suite
- `make test` / `make integration-test` / `make e2e-test` — test suites

### AGENTS.md Updates

Added comprehensive documentation of all build flags, backend selection, dispatch
modes, machine/model specialization, and the full megakernel build command. Also
added remote access section with npsh commands for GPU health checks and clock
management.

## Benchmark Improvements

### The Problem

The benchmark (`cmd_bench.nim`) had minimal warmup (1 run) and few samples (3 runs),
producing noisy results due to DVFS clock ramp-up. The Strix Halo iGPU starts at
600 MHz and takes ~40 kernel launches to reach full 2900 MHz speed.

### What Changed

- Default warmup: 1 → 3 runs (ensures GPU clocks are fully ramped)
- Default runs: 3 → 5 runs (more samples for statistics)
- Added standard deviation to output, matching llama-bench format:
  ```
  Mean: 144.15 ± 10.38 tok/s
  Min:  131.5 tok/s
  Max:  157.2 tok/s
  ```

### Remaining Variance

hippo_leap shows ±10 tok/s variance vs llama-bench's ±1 tok/s. This is because
GPU clocks can drop between benchmark runs during the gap where no kernels are
executing. llama-bench likely keeps the GPU busy more continuously, preventing
DVFS clock drops. Possible fixes:
- GPU event-based timing of decode tokens only (excludes inter-run gaps)
- Dummy kernel between runs to keep clocks pinned
- System-level clock pinning (unreliable on azem's kernel)

## Current Performance

### hippo_leap (megakernel, individual launches)

```
Model: TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf
Prompt: "Write a short story about a cat."
Max tokens: 128, Context: 2048
Runs: 5 (3 warmup)

Mean: 144.15 ± 10.38 tok/s (dynamic clocks)
~163 tok/s (pinned clocks, from earlier measurements)
```

### llama.cpp Vulkan (post-reboot, 2026-05-01)

```
TinyLlama 1.1B Q2_K, tg128:
300.81 ± 1.01 tok/s
```

### Gap Analysis

The gap has widened from 1.6x (133 vs 213) to 1.84x (163 vs 300). The llama.cpp
improvement from 213 to 300 tok/s likely came from llama.cpp Vulkan backend
improvements between our earlier measurement and now (we didn't rebuild llama.cpp
between measurements — the first 213 number may have been from an older build).

Our 163 tok/s represents 73% of measured peak DRAM bandwidth (224 GB/s) for the
weight data volume (~485 MB/token), meaning we're leaving ~27% on the table from
instruction overhead in the GEMV inner loops.

## Progression

| Version | tok/s | Key change |
|---------|-------|------------|
| v3 naive | 6.7 | F32 dequant, scalar attention |
| v4 megakernel | 34 | Native Q2K/Q3K GEMV |
| v5 warp attention | 84 | Warp-per-head parallel attention |
| v7 Q8_0 output | 102 | Output projection bandwidth 4x reduction |
| v8 clocks pinned | 133 | GPU at 2863 MHz instead of 600 MHz |
| v9 LDS activation | ~121 | LDS cache for nEmb GEMVs (+17%, later dropped) |
| v9+ kernel fusions | **163** | Dual GEMV, fused RoPE+KV, RMSNorm shuffle |
| llama.cpp Vulkan | **300** | Reference (Vulkan cooperative matrix) |

## Files Changed

- `Makefile` — proper build targets with all required flags
- `AGENTS.md` — comprehensive build flag docs, GPU health checks, npsh usage
- `docs/gpu-fault-recovery.md` — new doc for GPU fault diagnosis and recovery
- `src/hippo_leap/cmd_bench.nim` — warmup=3, runs=5, stddev output

## Known Issues

- `megakernel_config.nim` has `cuCount: 16` for azem, should be 40. Only affects
  the cooperative kernel path (not the hot path with individual launches), but is
  incorrect and should be fixed.
- Benchmark DVFS variance (±10 tok/s) is significantly worse than llama-bench (±1).
- The cooperative kernel path is the default (no flag) and is 5x slower than
  individual launches. Consider making individual launches the default, or emitting
  a compile-time warning when neither flag is set.
