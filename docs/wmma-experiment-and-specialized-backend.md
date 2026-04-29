# WMMA Experiment Results and Specialized Backend Design

**Date**: 2026-04-29

This document captures findings from the WMMA (Wave Matrix Multiply-Accumulate)
experiment on RDNA3/3.5 and outlines the design for a new specialized backend
using Nim compile-time metaprogramming.

---

## Part 1: WMMA Experiment

### Background

hippo_leap decodes Llama 3.2 1B Q4_K_M at **35.7 tok/s** on Strix Halo
(gfx1151, ~250 GB/s LPDDR5X). llama.cpp Vulkan achieves **212.9 tok/s** using
cooperative matrix extensions. The hypothesis was that replacing scalar
warp-per-row GEMV with WMMA tensor ops would close the gap by reducing
instruction count.

AMD's WMMA builtin `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` performs a
16x16x16 fp16 matmul producing fp32 results, distributed across 32 threads
(one wave). This replaces hundreds of scalar FMA operations per tile.

### Phase 1: Lane Mapping Discovery (gfx1151)

The WMMA instruction distributes matrix elements across threads in a
hardware-defined pattern that AMD does not fully document. We determined
the mapping empirically using probe kernels (`tests/integration_wmma.nim`).

**Confirmed mapping (RDNA3/3.5, wave32):**

```
Fragment A: A(tid, i) = A[tid % 16][i]          thread = row, index = col
Fragment B: B(tid, i) = B[i][tid % 16]          index = row, thread = col
Fragment C: C(tid, i) = C[2*i + tid/16][tid % 16]   interleaved even/odd rows
```

- Each thread holds 16 fp16 values per input fragment and 8 fp32 values per
  output fragment
- 32 threads x 16 = 512 slots for a 256-element matrix: each element is
  redundantly stored by 2 threads (threads 0-15 and 16-31 mirror each other
  for A and B)
- For C, threads 0-15 cover even rows, threads 16-31 cover odd rows

**GEMV broadcast pattern:** To compute `y = W * x` using WMMA, broadcast the
x vector into fragment B (`B[k][j] = x[k]` for all j). All 16 columns of C
then hold the same dot-product results, and we extract one column.

### Phase 2: WMMA Q4K Kernel

Implemented `linearQ4KWmmaDecodeKernel` in `naive.nim`, available behind
`-d:useWmma`. The kernel tiles 16 output rows together:

1. Each warp processes 16 rows of the weight matrix
2. Iterates over Q4K blocks (256 elements = 16 WMMA tiles of 16 columns)
3. Per tile: dequantize 16 weight elements per row to fp16, broadcast 16 x
   values to fp16, execute WMMA instruction, accumulate into fp32 fragC
4. Extract 16 results from fragC and write to output

Grid: `ceil(outRows / 16)` blocks of 32 threads each.

### Performance Results

| Kernel | tok/s | vs scalar |
|--------|------:|----------:|
| Scalar warp-per-row (baseline) | 34.9 | 1.0x |
| WMMA v1 (per-element dequant) | 7.9 | 0.23x |
| WMMA v2 (hoisted Q4K headers) | 20.7 | 0.59x |

**WMMA is ~40% slower than scalar for quantized GEMV.**

### Why WMMA Doesn't Help for Q4K GEMV

The scalar kernel does direct fp32 FMAs: load a nibble, dequantize to fp32,
multiply-accumulate in one step. The WMMA path requires:

1. Dequantize nibble to fp32
2. Convert fp32 to fp16 (for WMMA input)
3. Store into fragment register via `hippoWmmaSetF16`
4. Execute WMMA instruction
5. Read fp32 result from fragment via `hippoWmmaGetF32`

Steps 2-3 and 5 are pure overhead. For every 16x16 tile, we do 256 dequant +
fp16-conversion operations before a single WMMA instruction. The WMMA compute
savings cannot overcome the data preparation cost.

**WMMA is designed for cases where data is already in fp16** (or can be cheaply
loaded as fp16). Quantized weights require per-element dequantization that
dominates the cost. This is a fundamental mismatch, not an implementation issue.

### What WMMA Would Help

- **FP16 weight models**: If weights were stored as fp16, fragment fill becomes
  a simple load, and WMMA would provide ~16x compute density improvement
- **Prefill GEMM**: Batched token processing where the same weights are reused
  across many tokens. Dequant cost is amortized across the batch dimension
- **Attention GEMV**: Q*K and score*V are already fp16/fp32, no dequant needed

### Conclusion

The WMMA experiment proved that **compute is not the bottleneck** for quantized
decode GEMV. The real bottleneck is memory bandwidth utilization and kernel
orchestration overhead. This motivated a strategic pivot toward a specialized
backend focused on memory system optimization rather than compute throughput.

---

## Part 2: Performance Gap Analysis

### The Numbers

Llama 3.2 1B Q4_K_M on Strix Halo (gfx1151, LPDDR5X ~250 GB/s):

| System | tok/s | % of theoretical max |
|--------|------:|---------------------:|
| Theoretical bandwidth ceiling | ~345 | 100% |
| llama.cpp Vulkan (KHR_coopmat) | 212.9 | 62% |
| hippo_leap (scalar warp-per-row) | 35.7 | 10% |

**hippo_leap uses only 10% of available memory bandwidth.** The remaining 90%
is lost to:

### Where the bandwidth goes

1. **Kernel launch overhead (~15% of token time)**
   - ~170 kernel launches per token
   - Each launch: CPU-side command buffer construction, ring buffer submission,
     GPU command processor dispatch, CU allocation, wavefront creation
   - ~2000-5000 cycles per launch vs ~100 cycles for a barrier

2. **Memory access pattern inefficiency**
   - Scalar warp-per-row: each warp independently reads its own weight row
   - No cross-warp coordination or prefetching
   - Q4K block headers (d, dmin, scales) re-read by every warp touching that
     column range
   - x vector re-read from global memory by every GEMV kernel launch

3. **Pipeline starvation between kernels**
   - GPU drains completely between kernel launches
   - No overlap between kernel epilogue and next kernel prologue
   - L2 cache cold-starts on every launch (weights evicted between kernels)

4. **Redundant memory traffic**
   - x vector (2048 fp32 = 8KB) read by every GEMV kernel (5 per layer, 16 layers)
   - That's 80 reads of the same data = 640KB wasted per token
   - RMSnorm output written to global memory, immediately read by next GEMV

### Why llama.cpp is 6x faster

llama.cpp Vulkan's advantage is not primarily from cooperative matrix (their
version of WMMA). It's from:

- **Shader dispatch batching**: Vulkan command buffers batch multiple dispatches
  with minimal inter-dispatch overhead
- **Descriptor-based weight access**: weights stay in GPU-visible memory with
  minimal cache thrashing
- **Cooperative matrix for compute density**: reduces instruction pressure,
  leaving more bandwidth for memory operations
- **Optimized memory access patterns**: coalesced reads, tiled weight layouts

The cooperative matrix helps because llama.cpp's weights are already formatted
for efficient matrix tile loading. hippo_leap's Q4K dequant path is
fundamentally different — it unpacks nibbles on-the-fly.

---

## Part 3: Project Philosophy

### What hippo_leap Is

hippo_leap is not trying to be llama.cpp. llama.cpp already exists and does a
great job — we use it via lm-studio for real inference workloads. hippo_leap
exists to prove that **a single person with Nim can build a legible, hackable
inference stack that's fast enough to be useful and simple enough to experiment
with.**

This is part of a broader full-Nim AI stack:
- **openai_leap** — OpenAI-compatible client library
- **typos** — agent harness
- **scriptorium** — agent orchestrator (dozens of instances running in production)
- **hippo** — GPU compute library (HIP bindings + abstractions)
- **hippo_leap** — LLM inference engine

The value proposition isn't raw performance — it's that the entire stack is one
language, readable, and hackable. No Python/C++/CUDA polyglot, no massive
framework dependencies, no DSLs. Just Nim.

### Design Principles

1. **Simplicity over generality.** It's easier to make something fast by
   cutting cruft than by hammering at a general-purpose framework. Rather than
   building a system that works for everything, specialize for what we actually
   run.

2. **Experimentation platform.** The field is changing rapidly. hippo_leap
   should make it easy to try new ideas (exotic quant formats, novel
   architectures, different hardware targets) without fighting layers of
   abstraction.

3. **CPU is a first-class citizen.** Not every interesting model needs a GPU.
   Low-bit models (ternary, 2-bit) are surprisingly competitive on CPU with
   SIMD, and having a CPU path means the code runs everywhere.

4. **Compile-time specialization where it's simple.** Nim's `when` blocks and
   `static` params make specialization natural and readable. Use them where
   they make the code cleaner, not as a grand macro DSL.

5. **"Fast enough to be useful"** — not "fastest possible." If we're within
   striking distance of llama.cpp for the models we care about, that's a win.
   The real wins are in hackability and experimentation velocity.

---

## Part 4: Directions for Exploration

### Low-Bit and Ternary Models

Personal interest area with genuine technical advantages for a Nim-based stack:

**Ternary (1.58-bit) models** use weights from {-1, 0, +1}. The "dequant" is
just a sign mask — no floating-point conversion at all. GEMV becomes additions
and subtractions. This is:
- Inherently more deterministic (no floating-point rounding variance)
- Naturally fast on CPU (SIMD popcount + conditional negate)
- Memory bandwidth optimal (1.58 bits per weight vs 4+ for Q4K)
- A genuine advantage for Nim — generating tight inner loops for exotic formats
  is exactly where compile-time specialization shines vs frameworks that assume
  weights are floats

**2-bit models** (Q2_K and friends) are already supported in hippo_leap and
performed well on 7900 XTX (371 tok/s, beating llama.cpp). These models are
interesting for:
- Maximum throughput on bandwidth-limited hardware (Strix Halo LPDDR5X)
- Running larger models in less memory
- Exploring quality-vs-speed tradeoffs at the extreme end

The general theme: as quantization gets more aggressive, the "dequant" step
simplifies, and the advantage of general-purpose matrix accelerators (WMMA,
cooperative matrix) shrinks. At 1.58 bits there's nothing to accelerate —
the operation is already trivial. This plays to our strengths.

### Compile-Time Specialization (Incremental, Not Grand)

Rather than building a full macro DSL megakernel system, pursue specialization
incrementally:

**Step 1: Compile-time model constants.** Make dimensions, head counts, layer
counts available as `const` values. Loop bounds become literals, the compiler
unrolls and optimizes naturally. No new abstractions needed — just `when` and
`static`.

```nim
const HiddenDim {.intdefine.} = 2048
const FFNDim {.intdefine.} = 8192
const NHeads {.intdefine.} = 32
```

**Step 2: Quant-type specialization.** Instead of runtime dispatch on quant
type, compile separate binaries (or use `when` blocks) for Q4K, Q6K, Q2K,
ternary. Dead code elimination removes unused quant paths entirely.

**Step 3: Kernel fusion where it's obvious.** Fuse operations that always run
back-to-back (RMSnorm+GEMV, GEMV+residual). Don't try to fuse everything —
just the cases where the intermediate data is small and the launch overhead
is large relative to the work.

**Step 4: Machine-aware tuning.** Block sizes, warp counts, and shared memory
usage tuned per GPU. Not a runtime autotuner — just a config file per target
that feeds compile-time constants.

```nim
# machine_gfx1151.nim
const CUCount* = 16
const LDSSize* = 65536
const MemBandwidthGBs* = 250

# machine_gfx1100.nim
const CUCount* = 96
const LDSSize* = 65536
const MemBandwidthGBs* = 960
```

Each step is independently useful and testable. No step requires the previous
ones to be "complete." This is the opposite of a grand unified plan — it's a
toolkit of simple techniques that compose.

### CPU Backend

For low-bit models especially, a well-optimized CPU path using Nim's existing
SIMD support (nimsimd) could be competitive:

- **Ternary GEMV on CPU**: Each weight is 2 bits. Pack 64 weights per uint128.
  Use SIMD to mask-and-add in bulk. No GPU launch overhead, no driver stack.
- **AVX-512 / NEON paths**: Nim compiles to C++, so platform SIMD intrinsics
  are available. nimsimd already wraps common ones.
- **Useful for edge deployment**: Run on machines without AMD GPUs. ARM servers.
  Laptops without ROCm.

### Industry Context

For reference, here's where the major approaches sit:

| Approach | Complexity | Flexibility | Performance |
|----------|-----------|-------------|-------------|
| llama.cpp | Medium | High (many backends) | Good |
| Triton/FlashInfer | High (JIT, Python) | Medium (CUDA-focused) | Excellent |
| TensorRT-LLM | Very high | Low (NVIDIA only) | Best |
| hippo_leap naive | Low | Medium (HIP + CPU) | Decent |
| hippo_leap specialized | Low-Medium | Low (per-model) | Good (goal) |

We're not competing on the performance axis. We're competing on the
**simplicity + hackability** axis, where Nim gives us a genuine edge. The
performance just needs to be good enough that the models are usable for
real work.

---

## Part 5: What's Next

### Keep Working

- **Naive backend**: Stays as the general-purpose, easy-to-understand reference.
  Continue adding model support and quant types here.
- **WMMA knowledge**: Lane mapping is documented and tested. Useful if we later
  do fp16 attention or prefill GEMM, or if someone else needs RDNA3 WMMA docs.
- **Integration tests**: Keep the test suite running across both GPU targets
  (gfx1100, gfx1151) and CPU (SIMPLE backend).

### Explore When Interesting

- **Ternary model support**: Add a ternary quant type to the GGUF loader and
  write CPU + GPU kernels. See how simple the code can be.
- **Compile-time constants**: Try defining model dims as `-d:HiddenDim=2048`
  and measure if the compiler makes measurably better code. Low effort, might
  surprise us.
- **Kernel fusion experiments**: Pick one obvious fusion (RMSnorm+GEMV or
  GEMV+residual) and see how much it helps. One experiment, not a framework.
- **CPU SIMD GEMV**: Write a Q2K or ternary GEMV using nimsimd, benchmark
  against the GPU path for small models.

### Don't Over-Plan

The field is moving fast. New model architectures, new quantization schemes,
and new hardware capabilities appear regularly. The best strategy is to keep
the codebase simple enough that adapting to new developments is easy, rather
than building elaborate infrastructure for today's assumptions.

hippo_leap's strength is that one person can understand the entire stack, top
to bottom. Preserve that.

---

## Appendix: Strix Halo Memory Notes

The AMD Radeon 8060S in Strix Halo uses **unified memory** (LPDDR5X shared
between CPU and GPU). There is no discrete VRAM. The `rocm-smi` "VRAM%"
field is misleading — it shows a small pre-allocated pool, not actual GPU
memory usage. Real GPU memory consumption appears under the **GTT** (Graphics
Translation Table) pool, currently capped at ~112 GB.

For bandwidth analysis, the relevant number is the LPDDR5X bandwidth
(~250 GB/s peak, shared with CPU). This is approximately 4x lower than the
7900 XTX's 960 GB/s GDDR6X, which explains why the same kernels that matched
llama.cpp on 7900 XTX (TinyLlama Q2_K) are 6x slower on Strix Halo — the
bandwidth-to-compute ratio is fundamentally different.

This also means that **L2 cache efficiency matters much more on Strix Halo**.
Weight data that stays in L2 across kernel phases avoids the slow LPDDR5X
round-trip entirely. This is a strong argument for the persistent megakernel
approach on unified memory systems.
