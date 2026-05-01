# v9 Brainstorm: Closing the Gap to llama.cpp Vulkan

## Current State

133 tok/s (hippo_leap, HIP) vs 213 tok/s (llama.cpp Vulkan) on Strix
Halo gfx1151. Gap is 1.6x. The GPU is bandwidth-bound on LPDDR5X at
~256 GB/s, with 269 kernel launches per token. Launch overhead is
hidden behind memory latency (proven by HIP Graph experiment).

## Corrected Understanding

The profiling agent claimed our byte-at-a-time reads are "not
coalesced." This is partially wrong. RDNA3's VMEM coalescer looks at
addresses across all 32 lanes. If consecutive lanes read consecutive
bytes, the hardware coalesces into minimal 32B/64B/128B transactions
regardless of element size. Our inner loops where `laneId` indexes
consecutive bytes within a quant block ARE coalesced.

What IS wasteful: we issue one load instruction per byte. RDNA3
supports `global_load_b128` (16 bytes per lane per instruction). We're
using ~16x more load instructions than necessary, wasting issue slots
and instruction cache.

## Ranked Optimizations

### 1. Vectorized Weight Loads (highest confidence)

**Problem**: Q2K inner loop reads `wArr[bs + offset]` one byte at a
time. Each byte load is a separate instruction. With 84 bytes per Q2K
block and 32 lanes, we issue dozens of individual byte loads.

**Fix**: Read uint32 or uint64 chunks and extract quant values with
shifts and masks in registers.

Q2K block layout (84 bytes):
- bytes 0-1: fp16 scale (d)
- bytes 2-3: fp16 scale_min (dmin)
- bytes 4-19: 16 sub-block scales (4 bits each, packed)
- bytes 20-83: 64 quant bytes (2 bits per value, 4 values per byte)

Instead of 64 byte reads for the quant values, read 16 uint32s
(or 8 uint64s). Each uint32 holds 16 Q2K values (2 bits each).
Extract with `(val shr (2*i)) and 3`.

Expected: fewer load instructions → more issue slots for FMA. Not a
bandwidth improvement (same bytes transferred) but an IPC improvement.

Estimated impact: 1.3-1.5x for Q2K/Q3K GEMV (the dominant cost).

### 2. LDS Activation Caching — Revisited

**Previous result**: 13% regression with 8 warps/block, 22KB LDS
(ffnDim=5632 activation).

**What was wrong**: We used a single LDS size for ALL GEMVs. The
ffnDim GEMVs (wGate, wUp, wDown) need 5632 floats = 22KB. At 8
warps/block, that's 22KB LDS per block → only 2 blocks per CU
(44KB of 64KB) → low occupancy.

**Revised approach**: Use two block sizes:
- nEmb GEMVs (Q/K/V/O projections, 10 per layer): inDim=2048 →
  8KB LDS. At 8 warps/block: 8KB per block, 8 blocks per CU
  possible (64KB total). Excellent occupancy.
- ffnDim GEMVs (gate, up, down, 3 per layer): inDim=5632 →
  22KB LDS. Could use 2 warps/block instead of 8: 22KB per block,
  2 blocks per CU. Or skip LDS for these entirely.

The research says LDS activation tiling is "the single biggest win"
for RDNA3 quantized GEMV. llama.cpp Vulkan does exactly this. Worth
a second attempt with proper per-GEMV block sizing.

Note on LDS bank conflicts: RDNA3 LDS has 32 banks, 4 bytes wide.
32 lanes reading consecutive floats = 32 banks hit once each = no
conflict. Our float32 activation vector is conflict-free.

Estimated impact: 1.2-1.4x if done correctly (reduces redundant
activation reads from L2).

### 3. Multi-Row Warps

**Problem**: Each warp processes 1 output row. The activation vector
(8KB for nEmb=2048) is read once per warp from L2. With 2048 output
rows, that's 2048 × 8KB = 16MB of activation reads — all redundant.

**Fix**: Each warp processes 2-4 output rows. Load the activation
segment into registers once, then accumulate for multiple weight rows.

This is complementary to LDS caching but works even without it. Each
lane holds a few activation values in registers and reuses them across
rows. The cost is more register pressure (one accumulator per row).

For Q2K with 2 rows per warp: each lane accumulates 2 partial sums
instead of 1. Register pressure doubles but is still well within RDNA3
limits (256 VGPRs per thread).

Estimated impact: 1.2-1.3x (halves activation read traffic).

### 4. Warp Shuffle for Q8_0 Scale Broadcast

**Problem**: Q8_0 kernel reads the 2-byte scale via memory: all 32
lanes read `wArr[bs]` and `wArr[bs+1]`. Only lane 0 gets an L1 hit;
others stall or replay.

**Fix**: Lane 0 reads the scale, broadcasts via `hippoShfl(dRaw, 0)`.
One memory read instead of 32.

Estimated impact: 5-10% for output GEMV (Q8_0). Trivial to implement.

### 5. DPP/Subgroup Reductions

**Problem**: Our warp reductions use `hippoShflDown` in a tree pattern
(5 steps for wave32). This works but llama.cpp uses `subgroupAdd()`
which maps to hardware-accelerated DPP instructions.

**Fix**: On RDNA3, `__builtin_amdgcn_ds_swizzle_b32` or
`__builtin_amdgcn_permlane*` give single-cycle butterfly reductions.
Could expose via hippo as a `hippoSubgroupAdd` intrinsic.

Estimated impact: Small — reduction is a tiny fraction of total time.
But free performance if hippo exposes it.

### 6. Output Weight Q4_0

**Problem**: Output GEMV uses Q8_0 (65MB per token). Q4_0 would be
~33MB. At 256 GB/s, saves ~0.13ms per token.

**Fix**: Add `quantizeRowQ4_0` and `linearQ4_0WarpKernel`. Load-time
requantize like we do for Q8_0.

Risk: Q4_0 may degrade output quality noticeably for the argmax. Need
to verify golden output still matches.

Estimated impact: ~5-8% total (output GEMV is ~15% of token time).

### 7. Weight Layout Reorganization

**Problem**: GGUF Q2K/Q3K block layout is optimized for CPU SIMD, not
GPU wave32. The block interleaving (scales, then quant bytes, then
hmask for Q3K) means each warp reads from scattered offsets within the
block.

**Fix**: At load time, repack weights into a GPU-friendly layout where
each 128-byte aligned chunk contains exactly what one wave32 needs for
one iteration. For Q2K: pack {scales for 32 rows, quant values for 32
rows} so a single `global_load_b128` fetches a lane's share.

This is what llama.cpp's TQ4_0 format does — a "tensor-quantized"
layout optimized for GPU access patterns.

Estimated impact: potentially 1.5-2x for Q2K/Q3K, but high effort
and creates a custom weight format.

### 8. Vulkan Backend

**Nuclear option**: Write a Vulkan compute backend instead of HIP.
Vulkan has lower dispatch overhead on RDNA (RADV driver goes directly
to kernel, vs HIP's HSA runtime layer). But our HIP Graph experiment
showed dispatch isn't the bottleneck — we're bandwidth-bound.

Estimated impact: <10% for decode (dispatch overhead is hidden).
Enormous effort. Not worth it unless we want prompt processing speed.

## Priority Order

| # | Optimization | Impact | Effort | Dependencies |
|---|---|---|---|---|
| 1 | Vectorized weight loads | 1.3-1.5x | Medium | None |
| 2 | LDS activation (nEmb only) | 1.2-1.4x | Medium | None |
| 3 | Multi-row warps | 1.2-1.3x | Medium | Works with #1 |
| 4 | Q8_0 scale shuffle | 5-10% | Trivial | None |
| 5 | Output Q4_0 | 5-8% | Low | None |
| 6 | DPP reductions | 2-3% | Low | Hippo change |
| 7 | Weight repacking | 1.5-2x | High | Custom format |
| 8 | Vulkan backend | <10% | Very high | New backend |

Optimizations 1-3 are independent and could be combined. If each
delivers the low end of its range: 1.3 × 1.2 × 1.2 = 1.87x →
133 × 1.87 = **~249 tok/s** (exceeding llama.cpp Vulkan).

Even just #1 alone at 1.5x would give **~200 tok/s**.

## Key Insight from Research

The research confirms our GPU is purely bandwidth-bound on LPDDR5X.
The 1.6x gap is NOT from:
- Launch overhead (HIP Graph proved this)
- Codegen quality (gfx1100 workaround proved this)
- Clock speed (pinned clocks proved this)

It IS from:
- Too many load instructions per byte (vectorized loads fix this)
- Redundant activation reads across warps (LDS/multi-row fixes this)
- Sub-optimal instruction-level parallelism (vectorized loads + DPP)

The llama.cpp Vulkan shaders achieve ~213 tok/s by doing exactly
optimizations 1+2+3: vectorized loads, LDS activation tiling, and
multi-column workgroups. Their per-quant-block inner loop issues
~4 load instructions where ours issues ~20+.
