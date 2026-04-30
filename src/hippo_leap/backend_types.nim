## GPU backend type definitions shared across all backends.

when not defined(cpp):
  {.error: "backend_types requires Nim's C++ backend. Build with `nim cpp`.".}

import
  hippo,
  ./tensor

type
  HippoAllocRef* = type(hippoMalloc(1))

  LayerKind* = enum
    lkAttentionFfn
    lkAttention
    lkFfnOnly
    lkSsm
    lkSsmAttnMoe
    lkAttnMoe

  GpuTensor* = object
    devicePtr*: pointer
    alloc*: HippoAllocRef
    shape*: seq[int]
    sizeBytes*: int

  GpuKvCache* = object
    k*: seq[GpuTensor]
    v*: seq[GpuTensor]
    curLen*: int
    maxLen*: int
    nHeadKv*: int
    headDim*: int

  SsmGpuState* = object
    convState*: seq[GpuTensor]
    recState*: seq[GpuTensor]
    ssmLayerMap*: seq[int]

  KvCache* = object
    k*: seq[Tensor]
    v*: seq[Tensor]
    curLen*: int
    maxLen*: int
    nHeadKv*: int
    headDim*: int
    gpuCache*: GpuKvCache
    ssmState*: SsmGpuState

  GpuQuantWeight* = object
    devicePtr*: pointer
    alloc*: HippoAllocRef
    sizeBytes*: int
    nRows*: int
    nCols*: int

  LayerGpuPtrs* = object
    kind*: LayerKind
    attnNorm*, ffnNorm*: pointer
    wq*, wk*, wv*, wo*: pointer
    wGate*, wUp*, wDown*: pointer
    attnQNorm*, attnKNorm*: pointer
    wqQ*, wkQ*, wvQ*, woQ*: pointer
    wGateQ*, wUpQ*, wDownQ*: pointer
    wColsQ*, wColsDown*: int
    wqQType*, wkQType*, wvQType*, woQType*: int32
    wGateQType*, wUpQType*, wDownQType*: int32
    # SSM (Mamba2) fields
    ssmInQ*, ssmOutQ*: pointer
    ssmInQType*, ssmOutQType*: int32
    ssmConv1dW*, ssmConv1dBias*: pointer
    ssmDtBias*, ssmA*, ssmD*: pointer
    ssmNorm*: pointer
    layerNFfn*: int
    layerNHeadKv*: int
    # Fused QKV (qwen3.6)
    wqkvQ*: pointer
    wqkvQType*: int32
    # SSM gate projection (qwen3.6 Delta Net: nEmb → ssmInnerSize)
    ssmGateQ*: pointer
    ssmGateQType*: int32
    # Post-attention norm (qwen3.6)
    postAttnNorm*: pointer
    # MoE router + experts
    moeRouterW*: pointer
    moeGateExpsQ*: pointer
    moeUpExpsQ*: pointer
    moeDownExpsQ*: pointer
    moeGateExpsQType*, moeUpExpsQType*, moeDownExpsQType*: int32
    moeGateExpsSliceBytes*, moeUpExpsSliceBytes*, moeDownExpsSliceBytes*: int
    # Shared expert
    moeShGateQ*, moeShUpQ*, moeShDownQ*: pointer
    moeShGateQType*, moeShUpQType*, moeShDownQType*: int32
    moeShGateScalar*: pointer
    # Qwen3.6 SSM (alpha/beta formulation)
    ssmAlphaQ*, ssmBetaQ*: pointer
    ssmAlphaQType*, ssmBetaQType*: int32

  ModelGpuPtrs* = object
    layers*: seq[LayerGpuPtrs]
    tokEmb*: pointer
    normWeight*, outputWeight*: pointer
    outputWeightQ*: pointer
    outputQType*: int32
    outputShape0*, outputShape1*: int
    ropeTheta*: pointer
    initialized*: bool

  GpuContext* = object
    initialized*: bool
    stream*: HippoStream
    act0*: GpuTensor
    act1*: GpuTensor
    actCapBytes*: int
    scratch0*: GpuTensor
    scratch1*: GpuTensor
    scratch2*: GpuTensor
    scratch3*: GpuTensor
    scratchCapBytes*: int
    argmaxScratch*: GpuTensor
    argmaxResult*: GpuTensor
    attnScratch*: GpuTensor
    attnScratchBytes*: int
