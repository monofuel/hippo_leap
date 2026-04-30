# v8: HIP Graph Capture, Codegen, and GPU Clock Investigation

## Context

After v7 (Q8_0 output quantization, ~102 tok/s), profiling suggested
~82% of the 9.8ms/token budget was kernel launch overhead (269 launches
per token, only 18% bandwidth utilization). This motivated three
optimization attempts targeting that overhead.

## Hardware

- AMD Strix Halo (gfx1151, RDNA 3.5), 16 CUs, wave32
- Unified LPDDR5X ~256 GB/s, 32MB L2 cache
- Max GPU clock: 2900 MHz, idle: 600 MHz
- ROCm 7.2 (HIP 7.2, clang 22.1)

## What We Tried

### 1. HIP Graph Capture

**Hypothesis**: Replace 269 individual `hipLaunchKernel` calls with a
single `hipGraphLaunch` to eliminate per-kernel CPU dispatch overhead.

**Implementation**: Added HIP Graph API bindings to hippo
(`hipStreamBeginCapture`, `hipStreamEndCapture`, `hipGraphInstantiate`,
`hipGraphLaunch`). Created graph-compatible kernel variants that read
variable per-token arguments (tokenId, pos, seqLen) from a device-side
config struct updated via a single `hipMemcpyAsync` before each replay.
Available via `-d:useGraphCapture`.

**Result**: No improvement. ~132 tok/s graph vs ~133 tok/s individual
(with clocks pinned). The launch overhead was not the bottleneck.

**Why**: The GPU is memory-bandwidth-bound, not dispatch-bound. Each
kernel waits on weight reads from LPDDR5X. While the GPU waits for
memory, the CPU has plenty of time to dispatch the next kernel. Launch
overhead is fully hidden behind memory latency.

### 2. gfx1100 Codegen Workaround

**Hypothesis**: ROCm generates ~2x slower code for gfx1151 vs gfx1100
(per llama.cpp issue #13565). Compiling as gfx1100 with
`HSA_OVERRIDE_GFX_VERSION=11.0.0` should improve performance.

**Implementation**: Build with `--passC:'--offload-arch=gfx1100'
--passL:'--offload-arch=gfx1100'`, run with
`HSA_OVERRIDE_GFX_VERSION=11.0.0`.

**Result**: ~17% improvement with auto clocks (102 → ~124 tok/s), but
**zero improvement** with clocks pinned (~133 tok/s either way).

**Why**: The apparent codegen improvement was a measurement artifact.
The gfx1100 binary triggers faster DVFS clock ramp-up, making short
benchmarks appear faster. With clocks already at max, there is no
difference. Our kernels are simple enough that gfx1151 codegen is fine.

### 3. GPU Clock Pinning

**Hypothesis**: The GPU idles at 600 MHz and may not ramp to full speed
during short inference bursts.

**Implementation**: `echo high > /sys/class/drm/card0/device/power_dpm_force_performance_level`
pins the GPU to 2863 MHz.

**Result**: **+30% improvement** (102 → 133 tok/s). This was the actual
bottleneck.

**Why**: With `auto` power management, the GPU starts at 600 MHz and
takes several tokens to ramp up. For short inference runs (8-32 tokens),
a significant fraction of tokens execute at reduced clocks. The iGPU's
DVFS is conservative — it ramps slowly because the workload (many small
kernels with gaps between them) doesn't look like sustained load to the
power controller.

We are **not** pinning the performance level in the codebase. This is a
system-level setting with power/thermal implications for the iGPU. The
correct fix is longer-running inference where DVFS has time to ramp, or
a future kernel design that presents a more sustained load profile.

## Combined Results

All measurements with warmup (8 tokens discarded) + 33 token benchmark.

| Variant | Auto clocks | Pinned 2863 MHz |
|---------|-------------|-----------------|
| Individual launches (gfx1151) | ~102 tok/s | **~133 tok/s** |
| HIP Graph capture (gfx1151) | ~106 tok/s | ~132 tok/s |
| Individual (gfx1100 override) | ~124 tok/s | ~133 tok/s |
| Graph + gfx1100 | ~107 tok/s | ~132 tok/s |

## What This Tells Us

The performance model we had was wrong. We estimated 82% launch
overhead based on counting launches and measuring total time, but this
conflated memory-latency-bound idle time with CPU dispatch overhead.
The GPU spends most of its time waiting for weight data from LPDDR5X,
and the CPU dispatches the next kernel during that wait.

The real breakdown is closer to:
- ~7.5ms: weight reads from LPDDR5X (bandwidth-bound)
- ~1.5ms: compute (ALU-bound phases like attention, RMSNorm)
- ~0.5ms: actual launch overhead (hidden behind the above)

This means the remaining gap to llama.cpp Vulkan (~213 tok/s, ~4.7ms)
must come from more efficient weight access: better dequantization
kernels, WMMA-accelerated GEMV, or more aggressive quantization.

## Progression

| Version | tok/s | Key change |
|---------|-------|------------|
| v3 naive | 6.7 | F32 dequant, scalar attention |
| v4 megakernel | 34 | Native Q2K/Q3K GEMV |
| v5 warp attention | 84 | Warp-per-head parallel attention |
| v7 Q8_0 output | 102 | Output projection bandwidth 4x reduction |
| v8 clocks pinned | 133 | GPU at 2863 MHz instead of 600 MHz |
| llama.cpp Vulkan | 213 | Target reference |

## Files Changed

- `../hippo/src/hip.nim` — HIP Graph API types and FFI procs
- `../hippo/src/hippo.nim` — hippo-level graph wrapper templates
- `src/hippo_leap/backends/megakernel.nim` — `DecodeStepConfig`,
  graph-mode kernels, `forwardDecodeGraph`, `-d:useGraphCapture`
- `tests/integration_inference.nim` — `testBenchmark` with warmup
