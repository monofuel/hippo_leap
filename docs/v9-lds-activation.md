# v9: LDS Activation Caching for nEmb-Sized GEMVs

## Context

After v8 (clock pinning, ~133 tok/s), the GPU is purely bandwidth-bound
on LPDDR5X at ~256 GB/s. The brainstorm identified two main paths:
vectorized weight loads and LDS activation caching.

## What We Tried

### 1. Q8_0 Scale Shuffle Broadcast

**Hypothesis**: Only lane 0 reads the 2-byte fp16 scale, broadcasts via
`hippoShfl`. Saves 31 redundant reads per quant block.

**Result**: **20% regression** (82.9 vs 103.6 tok/s without pinning).

**Why**: RDNA3's scalar L1 cache handles the broadcast efficiently — all
32 lanes reading the same 2 bytes hit the scalar cache and resolve in a
single cycle. The explicit branch (`if laneId == 0`) + shuffle adds
instruction overhead and breaks the compiler's ability to schedule the
load with subsequent FMA operations.

### 2. Vectorized uint32 Scale/d Loads (Q2K/Q3K)

**Hypothesis**: Cast `ptr uint8` to `ptr uint32` and load 4 bytes at
once, extract individual bytes with shifts.

**Result**: **Correctness failure** (golden output mismatch). The
`cast[ptr UncheckedArray[uint32]](addr wArr[bs])` pattern generates
incorrect code on HIP/RDNA — likely a strict aliasing violation where
the C++ compiler assumes uint32* and uint8* don't alias.

**Note**: This approach would need `memcpy` or `__builtin_memcpy` to be
safe in C++, which Nim's codegen doesn't naturally produce. Left for
future work if we add explicit `__attribute__((may_alias))` support to
hippo.

### 3. LDS Activation Caching (nEmb only)

**Hypothesis**: For GEMVs with inDim=2048 (8KB as float32), use
multi-warp blocks (8 warps = 256 threads) that cooperatively load the
activation vector into LDS, then each warp reads from LDS instead of L2.

Previous attempt failed because it used one block size for ALL GEMVs,
including ffnDim=5632 (22KB LDS → only 2 blocks per CU → low occupancy).
This time: LDS only for inDim ≤ nEmb=2048, regular 1-warp-per-row for
inDim=5632 (wDown).

**Implementation**: Three new kernels (`linearQ2KLdsKernel`,
`linearQ3KLdsKernel`, `linearQ8_0LdsKernel`) with:
- `var sAct {.hippoShared.}: array[2048, cfloat]` (8KB LDS per block)
- Cooperative fill: all 256 threads load activation into LDS
- `hippoSyncthreads()` barrier
- Each warp processes its own output row reading from `sAct[]`
- Grid: ceil(outDim/8) blocks, block: 256 threads

The `gpuLinear` dispatcher routes based on `inDim <= ModelCfg.nEmb`.

**Which GEMVs use LDS** (per layer):
- LDS: wq(2048→2048), wk(2048→256), wv(2048→256), wo(2048→2048),
  wGate(2048→5632), wUp(2048→5632) — 6 per layer + output(2048→32000)
- No LDS: wDown(5632→2048) — 1 per layer

**Result**: **121.4 tok/s** (auto clocks) vs 103.6 tok/s baseline =
**+17% improvement**. Golden output matches.

LDS occupancy: 8KB per block → up to 8 blocks per CU (64KB LDS). With
8 warps per block, that's 64 warps per CU — well above the minimum for
latency hiding.

## Results

All measurements without clock pinning (auto DVFS).

| Variant | tok/s | vs baseline |
|---------|-------|-------------|
| v7 baseline (individual launches) | 103.6 | — |
| Q8_0 scale shuffle | 82.9 | -20% |
| Q2K/Q3K vectorized loads | broken | — |
| LDS activation (nEmb only) | **121.4** | **+17%** |

With clocks pinned (extrapolated): 133 × 1.17 = **~156 tok/s**.

## Progression

| Version | tok/s | Key change |
|---------|-------|------------|
| v3 naive | 6.7 | F32 dequant, scalar attention |
| v4 megakernel | 34 | Native Q2K/Q3K GEMV |
| v5 warp attention | 84 | Warp-per-head parallel attention |
| v7 Q8_0 output | 102 | Output projection bandwidth 4x reduction |
| v8 clocks pinned | 133 | GPU at 2863 MHz instead of 600 MHz |
| **v9 LDS activation** | **~156** | LDS cache for nEmb GEMVs (+17%) |
| llama.cpp Vulkan | 213 | Target reference |

## What Failed

- Q8_0 scale shuffle: RDNA3 scalar cache already optimal, shuffle adds overhead
- uint32 cast loads: strict aliasing violation in HIP C++ codegen

## What's Next

Remaining gap to llama.cpp: ~156 vs 213 = 1.37x. Options:
- Multi-row warps (each warp does 2-4 output rows, reusing activation in registers)
- Output Q4_0 (halve output GEMV bandwidth)
- Safe vectorized loads via memcpy intrinsic (needs hippo changes)
- Weight layout repacking (highest effort, highest potential)
