# hippo_leap future optimizations

Reference baseline (tinylama on Radeon 7900 XTX, TinyLlama 1.1B Q2_K):
- tinylama: 371 tok/s
- llama.cpp: 353 tok/s
- Theoretical bandwidth limit: ~500-600 tok/s

## Where the time goes

Per-token decode takes ~2.7ms. Rough breakdown per layer (22 layers):
- GEMV projections (Q, K+V fused, O, gate+up fused, down): ~70% of time
- Attention (score + softmax + V accumulation): ~10%
- Kernel launch overhead (~180 launches): ~15%
- Small kernels (RMSnorm, residual, RoPE, KV store): ~5%

## High-impact opportunities

### Persistent kernels (eliminate launch overhead)
Instead of launching ~8 kernels per layer, launch one persistent kernel per
layer that stays resident on the CUs and executes all operations in sequence
using barriers. This eliminates ~180 kernel launches per token (~360-900us).
This is the single biggest remaining win but requires rethinking the entire
dispatch model. llama.cpp doesn't do this either.

### hipGraph / command batching
ROCm's hipGraph API records a sequence of kernel launches into a graph, then
replays it with minimal CPU overhead. Since the decode loop is the same
kernels with the same grid/block dims every step (only pos changes), this is
a natural fit. Could cut launch overhead by 50-80% with much less code change
than persistent kernels.

### Fuse RMSnorm + Q projection
Currently: RMSnorm kernel writes normalized x to tmp, then Q GEMV reads it.
Fusing these would eliminate one global memory round-trip of 2048 floats (8KB)
per layer and one kernel launch. Same pattern as the existing
residual+rmsnorm fusion.

### KV cache layout optimization
Current layout is [head][dim][pos] (column-major, good for prefill).
Decode reads one position at a time across all dims, which is strided.
A row-major [head][pos][dim] layout would make decode reads contiguous.
Trade-off: prefill would need transposed writes. Could maintain dual layouts
or switch based on phase.

### FlashAttention-style tiled attention
Benchmarks show tinylama loses to llama.cpp at 256+ decode tokens because our
attention kernel uses a flat shared memory scores array (O(seq_len)) with
strided KV cache reads. llama.cpp uses FlashAttention which tiles across
positions — processing KV cache in contiguous blocks of 32-64 positions,
never materializing the full scores, and using online softmax to merge tiles.

A naive online softmax attempt (warp-per-head, 64-dim output in registers)
regressed to 209 tok/s because per-position correction across all 64 output
dims was too expensive. FlashAttention avoids this by tiling across positions
(not dimensions), keeping score tiles + V accumulators in shared memory.

Implementation would require:
- Tiled KV cache reads (e.g., 32 positions per tile)
- Per-tile online softmax with inter-tile correction factors
- Careful shared memory budgeting (scores tile + V accumulator + Q vector
  competing for 64KB LDS on RDNA3)
- Separate code paths for prefill (batched) vs decode (single query)

## Medium-impact opportunities

### Fuse attention + O projection
After attention computes the output (nEmb floats), we immediately multiply by
Wo. Fusing these avoids writing/reading nEmb floats from global memory. Complex
because attention is 1 block per head but O projection is a full GEMV.

### Fuse down projection + residual add
The down projection output gets added to the residual. Could fold the addition
into the GEMV epilogue (write `out[i] + residual[i]` instead of `out[i]`).
Saves one elementwise kernel launch + memory round-trip.

### Speculative decoding
Generate multiple candidate tokens in parallel using a smaller draft model,
then verify with the full model. Not a kernel optimization but an algorithmic
one. Requires a second model or self-speculative approach.

## Lower-impact / research

### Warp-specialized attention
For longer sequences, split attention into producer warps (compute scores) and
consumer warps (accumulate V). Uses warp-level pipelining. Only matters when
curLen is large (>512).

### FP16 accumulation
Currently all accumulation is FP32. RDNA3 has 2x FP16 throughput. Could use
FP16 for intermediate GEMV accumulation with FP32 final reduction. Risk:
precision loss in softmax and long dot products.

### Quantized KV cache
Store KV cache in FP16 or INT8 instead of FP32. Halves or quarters the
memory bandwidth for attention. Requires quantize-on-store and
dequantize-on-load kernels.

### Multi-query batching
When generating multiple sequences, batch the GEMV operations into GEMM.
GPU utilization jumps dramatically. Not useful for single-sequence decode
but important for serving.

## The endgame: single mega-kernel decode

The theoretical optimal decode is a single persistent kernel launch per token
that stays resident on all CUs and executes everything — embedding lookup,
all layers, final norm, output logits, argmax — returning one token ID.

Key components:
- **Grid-level barriers** (`__grid_sync()` via cooperative launch) replace
  kernel launches. Each barrier costs ~100 cycles vs ~2000-5000 cycles for a
  launch round-trip.
- **Warp specialization**: some warps handle GEMV, others handle attention,
  others handle norms/residuals/RoPE. They communicate through global memory
  + barriers, not kernel boundaries.
- **Persistent LDS state**: the input vector x stays in LDS across phases
  instead of global memory round-trips.
- **Software pipelining**: while one layer's attention runs, prefetch the next
  layer's weights into L2. Currently each kernel launch resets prefetch.

The pure weight bandwidth limit for TinyLlama Q2_K is ~400MB of weights at
960 GB/s = ~2400 tok/s. We'll never hit that (attention reads KV cache,
irreducible compute) but it shows the gap between current performance and
the hardware ceiling.

## Compile-time kernel fusion via Nim macros

Use Nim's compile-time metaprogramming to automatically generate fused
mega-kernels from a high-level pipeline description.

### The concept

Define the decode pipeline as a compile-time dataflow graph:

```nim
decodeKernel(TinyLlama):
  let xNorm = rmsnorm(x, attnNorm)
  let q = gemvQ2K(xNorm, wq)
  let kv = fusedGemvQ2KQ3K(xNorm, wk, wv)
  let qk = rope(q, kv.k, thetaTable, pos)
  storeKV(kv, cache, pos)
  let attn = attention(qk, cache, pos)
  let xOut = gemvQ2K(attn, wo)
  residualAdd(x, xOut)
  # ... FFN block
```

A Nim macro would see the entire dataflow and at compile time:

- **Auto-fuse adjacent ops** — if rmsnorm output feeds directly into a GEMV,
  emit one kernel that keeps the intermediate in LDS instead of global memory
- **Assign warps to roles** — macro knows CU count, LDS budget, register
  pressure. Partitions warps across ops, inserts grid barriers between phases
- **Eliminate temporaries** — sees that `xNorm` is consumed once, keeps it in
  registers/LDS instead of writing to global memory
- **Specialize per model** — model params become compile-time constants. nEmb,
  headDim, nLayer become exact loop bounds, unrolled to known dimensions.
  No runtime conditionals
- **Specialize per GPU** — WarpSize, CU count, LDS size are compile-time
  constants. Generates a kernel tuned exactly for the target architecture

### Type-safe kernel composition

Nim's type system could enforce correctness at compile time:

```nim
type GpuTensor[Q: static QuantType, Rows, Cols: static int] = object
```

A `GpuTensor[Q2K, 2048, 256]` carries its quant format and dimensions.
`fusedGemv` refuses to compile if quant types don't match what the fused
kernel supports. Compile-time errors instead of silent numerical garbage.

### Why this is better than existing approaches

- **TVM/Triton** — generate kernels from Python DSLs but can't easily do
  cross-kernel fusion. Optimize individual kernels, not whole pipelines.
- **torch.compile** — traces Python but the abstraction is too high. Can't
  reason about quant block layouts or warp assignment.
- **CUTLASS** — C++ templates that specialize GEMM tiles. Powerful but limited
  to single operations, not full transformer layers.
- **Nim macros** — operate on the AST with full Turing-complete logic. Could
  write a scheduler that assigns CU workgroups at compile time based on model
  architecture.

### What hippo needs for this

1. `hippoGridSync` primitive (wraps `cooperative_groups::grid_group::sync()`)
2. `hippoCooperativeLaunch` (wraps `hipLaunchCooperativeKernel`)
3. A Nim macro library that takes the pipeline DSL and emits fused kernel code
4. A compile-time scheduler mapping operations to warp groups given HW constraints

### Incremental path

1. Start by auto-fusing two ops (e.g., rmsnorm + GEMV), validate against the
   hand-written kernels
2. Add grid sync + cooperative launch to hippo
3. Expand the macro to handle a full layer
4. Eventually generate the entire decode mega-kernel from the DSL

The end result: an ML compiler that fits in a Nim macro library instead of
being a massive infrastructure project like XLA or TVM. Compile-time kernel
fusion, model-specialized, GPU-specialized, with type safety.
