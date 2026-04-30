# v6 Performance Analysis: Megakernel on Strix Halo (gfx1151)

## Hardware

- AMD Strix Halo integrated GPU (gfx1151, RDNA 3.5)
- 16 CUs, wave32, 64KB LDS per CU
- Unified LPDDR5X, ~256 GB/s bandwidth
- 32MB L2 cache
- Model weights ~18MB Q2K (fits in L2 after first pass)

## Model

TinyLlama 1.1B Chat Q2_K (GGUF): 22 layers, nEmb=2048, nHead=32,
nHeadKv=4 (GQA 8:1), headDim=64, ffnDim=5632, nVocab=32000.
Mixed quant: Q2K (attn_q/k), Q3K (attn_v, wo, ffn_*), F32 (output).

## Current Numbers

| Backend | tok/s | ms/token |
|---------|-------|----------|
| Naive (F32 dequant, scalar attn) | ~6.7 | ~149 |
| Megakernel individual, scalar attn | ~34 | ~29 |
| Megakernel individual, warp attn | ~84 | ~12 |
| + Q8_0 output projection | **~102** | ~9.8 |
| Megakernel cooperative persistent | ~17 | ~58 |
| llama.cpp Vulkan | ~213 | ~4.7 |

## Issue 1: Cooperative Grid Sync is Wrong Primitive on gfx1151

`cooperative_groups::this_grid().sync()` on AMD is implemented as a
global-memory atomic counter loop. There is no hardware grid-sync
primitive on RDNA 3.5 (unlike recent NVIDIA architectures).

Measured cost: ~187 us per grid sync. With ~246 syncs per token,
that's ~46ms of pure sync overhead — explaining the 5x slowdown
vs individual launches.

Why it's so expensive on Strix Halo specifically:
- Unified LPDDR5X has ~100-200ns latency per access
- All 16 CUs spin-polling the same global flag = contention
- 1 block per CU (cooperative launch requirement) = minimal occupancy,
  zero latency hiding
- Spin-wait draws power without useful work, may trigger iGPU
  downclocking (ROCm issues #5750, #5745 document Strix Halo
  getting stuck at ~885 MHz under compute load instead of 2900 MHz)

### Verdict

The cooperative kernel as currently written uses the wrong primitive
on this hardware. Not "needs tuning" — wrong primitive. Don't sink
more time into making `grid.sync()` faster.

### What to do instead

The path serious persistent-kernel work has converged on (Hazy Research
"No Bubbles" megakernel, Mirage MPK from CMU/UW/NVIDIA, ThunderMLA) is
to delete grid syncs entirely and replace them with a **per-block
interpreter** that reads instructions from global memory and
synchronizes through atomic counters used as semaphores.

Design: an array of integers in global memory. Each instruction
increments its counter when done. Dependent instructions spin on the
counter they need. No `grid.sync()`.

Reference: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
(section on synchronization).

This also fixes the occupancy problem: an interpreter-style persistent
kernel runs more blocks per CU and pipelines weight loads from the next
instruction during the current one. That's where the real bandwidth win
comes from, not from removing kernel launches per se.

However, this is a major architecture change. The hybrid approach
(below) captures most of the benefit at a fraction of the complexity.

## Issue 2: Vulkan Target is Partially a Measurement Artifact

llama.cpp has a known issue on gfx1151 (ggml-org/llama.cpp#13565)
where the HIP backend is roughly 2x slower than Vulkan. Compiling
with `gfx1100` + `HSA_OVERRIDE_GFX_VERSION=11.0.0` nearly doubles
HIP performance vs native gfx1151 builds. So part of the 84 -> 213
tok/s gap is "ROCm's gfx1151 codegen is bad", not "our kernels are
bad."

TODO: Run llama.cpp's HIP backend on azem for a fairer reference.
TODO: Check `rocm-smi` clocks during inference to verify we're not
hitting the ~885 MHz clock bug.

## Issue 3: GEMV Has No LDS Activation Caching

Current warp-per-row Q2K/Q3K GEMV reads the activation vector
(nEmb=2048, 8KB) from global/L2 memory redundantly across all CUs.
With 32-lane warps, each lane reads ~64 activation elements per
quant block — repeated global/L2 traffic.

LDS on RDNA 3.5 is very low-latency and 64KB per CU. Loading the
activation vector into LDS once per block of rows and broadcasting
to warps would cut global reads dramatically.

llama.cpp Vulkan shaders do exactly this (shared-memory tiling for
quantized weights + activation broadcast).

Re-tested on megakernel backend with multi-warp blocks (8 warps/block,
BlockSize=256) and LDS activation cache. Result: **13% regression**
(220ms vs 192ms). The syncthreads overhead and reduced occupancy from
22KB LDS per block outweigh the bandwidth savings on this unified
memory iGPU with its large L2 cache. LDS caching is not beneficial
on Strix Halo for this workload.

## Issue 4: Attention Parallelism is Low

Current: 1 warp per head (32 heads), warps 32-127 idle during
attention = 75% of compute units wasted.

Possible improvements:
- Split across sequence positions (multi-warp per head for long KV)
- Flash-attention style tiling for decode
- Better V accumulation parallelism

For short sequences (decode), the current warp-per-head is adequate.
For longer KV caches this becomes the bottleneck.

## Issue 5: Output Projection is F32 and Bandwidth-Bound

The output GEMV (nEmb=2048 -> nVocab=32000) uses F32 weights:
~250MB touched per token. This blows out L2 (32MB) and hits pure
LPDDR5X bandwidth.

At 256 GB/s, 250MB takes ~1ms just to read — a significant chunk
of the 12ms budget.

Quantizing to Q8_0 (~4x smaller) or Q6_K (~5x smaller) would
dramatically reduce this. llama.cpp uses mixed quants for lm_head.
The GGUF file already has the output weight as F32 but it could be
requantized at load time.

## Issue 6: No Kernel Fusion

Current individual-launch path: ~330 kernel launches per token.
Each launch has overhead + forces a memory round-trip for
intermediate results.

Fusion opportunities (in priority order):
1. **Residual + RMSNorm**: already fused in individual path
   (`residualRmsnormKernel`), saves 1 launch + 1 memory RT
2. **Q/K/V projection + RoPE + KV store**: 5 launches -> 1.
   All read from act1, write to separate buffers.
3. **Gate + Up projection + SiLU*mul**: 3 launches -> 1.
   Both projections read act1, SiLU*mul is elementwise on results.
4. **Down projection + residual add**: 2 launches -> 1.
   GEMV epilogue adds to act0 directly.

Each fusion saves a launch (~5-30 us) and eliminates a memory
round-trip for intermediate activations (the bigger win — writing
then reading ~8KB through L2 instead of keeping it in registers
or LDS).

Aggressive fusion could cut launches from ~330 to <100 per token.

## Prioritized Action Plan

| Priority | Action | Result | Notes |
|----------|--------|--------|-------|
| 1 | Quantize output projection to Q8_0 | **~102 tok/s (+21%)** | 250MB → 65MB, 4x bandwidth reduction |
| 2 | LDS activation caching in GEMV | **Regression (-13%)** | Tested: syncthreads + LDS overhead > L2 cache benefit on iGPU |
| 3 | Fuse gate+up+SiLU (sequential) | **Regression (-10%)** | Tested: sequential 2x compute per row slower than parallel launches |
| 4 | Check gfx1151 clock bug (rocm-smi) | Idle at 600 MHz | Test too short to catch sustained clocks; needs longer benchmark |
| 5 | Hybrid persistent for tight phases | Not attempted | Requires interpreter-style design |
| 6 | Interpreter-style persistent kernel | Not attempted | Very high effort |

**Current best: ~102 tok/s** (Q8_0 output, no LDS, no fusion).

Key finding: On Strix Halo iGPU with unified LPDDR5X and 32MB L2,
both LDS caching and sequential kernel fusion are counterproductive.
The L2 is already efficient for the 8KB activation vector, and
individual kernel launches enable better parallelism across CUs than
fused sequential approaches.

## What to Drop

- Full cooperative persistent kernel with grid.sync() — wrong
  primitive on this hardware, measured 5x regression
- Further optimization of grid sync barriers — fundamental
  limitation of the implementation on RDNA 3.5 iGPU
- Chasing exact parity with llama.cpp Vulkan numbers — part of
  the gap is gfx1151 codegen quality, not kernel design
