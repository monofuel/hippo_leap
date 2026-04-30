# Nim GPU Pitfalls

Lessons learned writing GPU kernels in Nim via hippo (HIP/ROCm backend).
These are actionable items for improving hippo's ergonomics.

## 1. No compound assignment operators (`+=`, `*=`, `-=`)

Nim's `+=` and friends are defined as `template` in `system.nim` and
expand to code that calls host-only copy constructors. Inside a GPU
kernel (which compiles as `__global__` or `__device__` C++), the
compiler rejects them.

**Workaround**: Always write `x = x + y` instead of `x += y`.

**Hippo improvement**: Generate device-compatible `+=` / `*=` / `-=`
overloads for numeric types inside `{.hippoGlobal.}` and
`{.hippoDevice.}` proc bodies, or inject them automatically during
the macro transform.

## 2. Struct copies and array init generate `nimZeroMem`

Declaring `var arr: array[64, cfloat]` inside a kernel emits a call to
`nimZeroMem`, which is a host-only Nim runtime function. This causes
either a link error or a runtime crash.

The same applies to **struct copies**: `let lw = weights.layers[i]`
copies an `object` type, which Nim zero-initializes the destination
first via `nimZeroMem`. Access struct fields directly through the
original variable instead of copying.

**Workaround for arrays**: Use `{.emit.}` to declare and zero-fill C arrays:
```nim
{.emit: "float __attn_acc[64];".}
{.emit: "for (int __i = 0; __i < 64; __i++) __attn_acc[__i] = 0.0f;".}
```

**Workaround for structs**: Don't copy — access through the original:
```nim
# BAD: let lw = weights.layers[layer]  # nimZeroMem!
# GOOD: weights.layers[layer].wq       # direct field access
```

**Hippo improvement**: Intercept `nimZeroMem` calls inside GPU proc
bodies and replace them with a simple memset loop or `__builtin_memset`.
Alternatively, mark the generated code with `{.noinit.}` when inside a
device function context so Nim skips zero-initialization entirely.

## 3. `hippoArgs` requires lvalues

The `hippoArgs` template takes addresses of its arguments to build the
`void**` kernel parameter array. If you pass a cast expression or a
literal directly, Nim can't take its address — you get a cryptic
"cannot take address of expression" error.

**Workaround**: Stage every kernel argument as a `var` local:
```nim
var dstP = cast[ptr cfloat](buf.act0)
var wP = cast[ptr cfloat](weights.tokEmb)
var tid = cint(tokenId)
hippoLaunchKernel(myKernel, ..., args = hippoArgs(dstP, wP, tid))
```

**Hippo improvement**: Have `hippoArgs` automatically wrap non-lvalue
arguments in temporaries. The macro could detect which arguments are not
addressable and generate `let __arg_N = expr; addr __arg_N` for those.

## 4. Templates don't resolve inside `{.hippoGlobal.}` scope

The `hippoGlobal` macro transforms the proc body into a C++ `__global__`
function. During this transformation, module-level Nim templates are NOT
visible — they fail with "undeclared identifier." This affects
`hippoGridSync`, `hippoSyncthreads`, and any user-defined templates.

`{.importcpp.}` procs DO work because they're resolved as C++ symbols,
not Nim template expansions.

**Workaround**: Define critical operations as `{.importcpp.}` procs
instead of templates:
```nim
proc gridSync() {.importcpp: """
  do {
    cooperative_groups::grid_group __hl_grid = cooperative_groups::this_grid();
    __hl_grid.sync();
  } while(0)""", header: "hip/hip_cooperative_groups.h", nodecl, used.}
```

**Hippo improvement**: This is the biggest ergonomics issue. The macro
needs to either:
- Expand templates *before* transforming the body, or
- Inject template definitions into the transformed scope, or
- Convert key templates (`hippoGridSync`, `hippoSyncthreads`) to
  `{.importcpp.}` procs that survive the macro transform

## 5. `{.hippoDevice.}` procs need `{.used.}` pragma

Device functions (`__device__`) that are only called from other GPU code
may be eliminated by Nim's dead code elimination, since the Nim compiler
doesn't see them being called from any Nim code path. The call happens
in the C++ `__global__` function body, which Nim doesn't analyze.

**Workaround**: Add `{.used.}` to every `{.hippoDevice.}` proc.

**Hippo improvement**: The `hippoDevice` macro should automatically add
`{.used.}` to the proc it transforms. One-line fix in the macro.

## 6. Register pressure in persistent megakernels

A single persistent kernel that includes all phases (embedding, GEMV,
attention, normalization) for all 22 transformer layers compiles into
one enormous function. The GPU compiler cannot easily manage register
allocation across the entire function body, leading to excessive
register spills to VRAM (slow).

**Workaround**: Factor each phase into a separate `{.hippoDevice.}`
function. The compiler manages registers per call frame and spills only
across function boundaries, which is much more efficient. Keep the
`{.hippoGlobal.}` megakernel as a thin orchestrator that calls device
functions.

**Hippo improvement**: Not directly a hippo issue, but documenting this
pattern in hippo's guide would save users from the trap.

## 7. No `{.emit.}` blocks with Nim variable interpolation in GPU code

When using `{.emit.}` inside GPU kernels, you can't use Nim's `emit`
backtick interpolation to reference Nim variables. The generated C++
may reference variables that don't exist in the translated GPU scope.

**Workaround**: Use only C-level variable names in emit blocks, or
assign Nim values to C variables via emit first:
```nim
let qOff = head * headDim
{.emit: ["int __qoff = ", qOff, ";"].}
{.emit: "// now use __qoff freely in subsequent emit blocks".}
```

## 8. Kernel argument size limit (4KB on HIP)

HIP has a 4KB limit on total kernel argument size passed by value.
Large structs like `MkModelWeights` (~2.8KB of pointers) can easily
exceed this when combined with `MkBuffers` (~400B) and scalar args.

**Workaround**: Upload large structs to device memory and pass a single
pointer instead of passing by value:
```nim
var devWeights: pointer
let alloc = hippoMalloc(sizeof(MkModelWeights))
hippoMemcpy(alloc.p, addr mkWeights, sizeof(MkModelWeights), HostToDevice)
devWeights = alloc.p
```

**Hippo improvement**: Document the 4KB limit. Could also provide a
helper template like `hippoUploadArgs` that handles the alloc + copy
pattern for large argument structs.

## Summary for hippo improvement priorities

| Priority | Issue | Effort |
|----------|-------|--------|
| High | #4 Templates in hippoGlobal scope | Medium — macro rework |
| High | #1 Compound assignment operators | Low — inject overloads |
| High | #5 Auto-add `{.used.}` to hippoDevice | Trivial — one line |
| Medium | #3 hippoArgs lvalue requirement | Low — macro enhancement |
| Medium | #2 nimZeroMem in GPU code | Medium — intercept or noinit |
| Low | #7 emit interpolation | Hard — Nim compiler limitation |
| Low | #8 4KB arg limit docs | Trivial — documentation |
| Low | #6 Register pressure docs | Trivial — documentation |
