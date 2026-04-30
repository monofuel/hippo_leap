## Megakernel backend: HPC-style specialized binary for (model, machine, quant).
##
## All dimensions are compile-time constants. One persistent cooperative kernel
## per decode step with grid-level barriers replacing kernel launches.
##
## Build: nim cpp -r --cc:hipcc -d:backendMegakernel -d:targetMachine=azem
##        -d:targetModel=tinyllama_q2k -d:useMalloc -d:zippyNoSimd

import
  std/[tables, math, strformat],
  hippo,
  ../[backend_types, tensor, model, gguf_loader, quant]

import ../megakernel_config as mkc
from ../megakernel_config import
  ModelCfg, Machine, QDim, KvDim, HeadsPerKvGroup,
  NumBlocks, BlockSize, WarpsPerBlock, TotalWarps, MaxDim

when not defined(cpp):
  {.error: "megakernel backend requires Nim's C++ backend. Build with `nim cpp`.".}

# ---------------------------------------------------------------------------
# Weight structure — compile-time sized per model config
# ---------------------------------------------------------------------------

type
  MkWeight = object
    p: pointer
    qtype: int32            # GgmlType (0=F32, 10=Q2K, 11=Q3K, etc.)

  MkLayerWeights = object
    attnNorm: pointer       # F32 [nEmb]
    ffnNorm: pointer        # F32 [nEmb]
    wq: MkWeight
    wk: MkWeight
    wv: MkWeight
    wo: MkWeight
    wGate: MkWeight
    wUp: MkWeight
    wDown: MkWeight

  MkModelWeights = object
    tokEmb: pointer         # F32 [nVocab, nEmb]
    outputNorm: pointer     # F32 [nEmb]
    outputWeight: MkWeight
    ropeTheta: pointer      # F32 [ropeDim/2] precomputed
    layers: array[ModelCfg.nLayers, MkLayerWeights]

  MkBuffers = object
    act0: pointer           # [MaxDim] f32 — persistent activation
    act1: pointer           # [MaxDim] f32 — normalized activation
    scratch0: pointer       # [MaxDim] f32
    scratch1: pointer       # [MaxDim] f32
    scratch2: pointer       # [MaxDim] f32
    logits: pointer         # [nVocab] f32
    kvK: array[ModelCfg.nLayers, pointer]  # [KvDim, maxLen] f32
    kvV: array[ModelCfg.nLayers, pointer]

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

var
  mkWeights: MkModelWeights
  mkBuf: MkBuffers
  mkStream: HippoStream
  mkMaxLen: int
  mkInitialized: bool
  mkWeightAllocs: seq[HippoAllocRef]
  mkBufAllocs: seq[HippoAllocRef]

# ---------------------------------------------------------------------------
# Q2K constants
# ---------------------------------------------------------------------------

const
  QK_K = 256
  BlockQ2KSize = 2 + 2 + (QK_K div 16) + (QK_K div 4)  # 84 bytes

# ---------------------------------------------------------------------------
# GPU Kernels — individual launches for correctness verification
# ---------------------------------------------------------------------------

proc embeddingKernel(dst: ptr cfloat, weight: ptr cfloat,
                     tokenId: cint, nEmb: cint) {.hippoGlobal.} =
  let tid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  if tid < nEmb:
    let w = cast[ptr UncheckedArray[cfloat]](weight)
    let d = cast[ptr UncheckedArray[cfloat]](dst)
    d[tid] = w[cint(tokenId) * nEmb + tid]

proc rmsnormKernel(dst: ptr cfloat, src: ptr cfloat, weight: ptr cfloat,
                   dim: cint, eps: cfloat) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[BlockSize, cfloat]
  let tid = cint(threadIdx.x)
  let s = cast[ptr UncheckedArray[cfloat]](src)
  let d = cast[ptr UncheckedArray[cfloat]](dst)
  let wt = cast[ptr UncheckedArray[cfloat]](weight)
  var sumSq: cfloat = 0.0
  var i = tid
  while i < dim:
    sumSq = sumSq + s[i] * s[i]
    i = i + cint(blockDim.x)
  sdata[tid] = sumSq
  hippoSyncthreads()
  var stride = cint(blockDim.x) div 2
  while stride > 0:
    if tid < stride:
      sdata[tid] = sdata[tid] + sdata[tid + stride]
    hippoSyncthreads()
    stride = stride div 2
  let rms = 1.0f / sqrtf(sdata[0] / cfloat(dim) + eps)
  i = tid
  while i < dim:
    d[i] = wt[i] * s[i] * rms
    i = i + cint(blockDim.x)

proc residualRmsnormKernel(normOut: ptr cfloat, x: ptr cfloat,
                           residual: ptr cfloat, weight: ptr cfloat,
                           dim: cint, eps: cfloat) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[BlockSize, cfloat]
  let tid = cint(threadIdx.x)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let rArr = cast[ptr UncheckedArray[cfloat]](residual)
  let nArr = cast[ptr UncheckedArray[cfloat]](normOut)
  let wt = cast[ptr UncheckedArray[cfloat]](weight)
  var i = tid
  while i < dim:
    xArr[i] = xArr[i] + rArr[i]
    i = i + cint(blockDim.x)
  hippoSyncthreads()
  var sumSq: cfloat = 0.0
  i = tid
  while i < dim:
    sumSq = sumSq + xArr[i] * xArr[i]
    i = i + cint(blockDim.x)
  sdata[tid] = sumSq
  hippoSyncthreads()
  var stride = cint(blockDim.x) div 2
  while stride > 0:
    if tid < stride:
      sdata[tid] = sdata[tid] + sdata[tid + stride]
    hippoSyncthreads()
    stride = stride div 2
  let rms = 1.0f / sqrtf(sdata[0] / cfloat(dim) + eps)
  i = tid
  while i < dim:
    nArr[i] = wt[i] * xArr[i] * rms
    i = i + cint(blockDim.x)

proc addKernel(dst: ptr cfloat, src: ptr cfloat, dim: cint) {.hippoGlobal.} =
  let tid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  if tid < dim:
    let d = cast[ptr UncheckedArray[cfloat]](dst)
    let s = cast[ptr UncheckedArray[cfloat]](src)
    d[tid] = d[tid] + s[tid]

proc linearQ2KWarpKernel(dst: ptr cfloat, x: ptr cfloat, w: ptr uint8,
                         inDim: cint, outDim: cint) {.hippoGlobal.} =
  ## Warp-per-row Q2K GEMV. Matches naive backend's proven-correct layout:
  ## Q2K block = [scales:16][qs:64][d:2][dmin:2] = 84 bytes.
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let row = cint(blockIdx.x) * warpsPerBlock + warpId
  if row >= outDim: return

  let wArr = cast[ptr UncheckedArray[uint8]](w)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  let nBlocksPerRow = inDim div cint(QK_K)
  let rowSizeBytes = nBlocksPerRow * cint(BlockQ2KSize)
  let rowBase = row * rowSizeBytes

  let sub = (laneId shr 4'i32) and 1'i32
  let qsOff0 = 16'i32 + sub * 16'i32 + (laneId and 15'i32)
  let qsOff1 = 48'i32 + sub * 16'i32 + (laneId and 15'i32)

  var acc: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx < nBlocksPerRow:
    let bs = rowBase + blkIdx * cint(BlockQ2KSize)
    let eb = blkIdx * cint(QK_K)
    let dRaw = uint16(wArr[bs + 80'i32]) or (uint16(wArr[bs + 81'i32]) shl 8)
    let dmRaw = uint16(wArr[bs + 82'i32]) or (uint16(wArr[bs + 83'i32]) shl 8)
    let d = hippoHalfToFloat(dRaw)
    let dm = hippoHalfToFloat(dmRaw)
    let qb0 = wArr[bs + qsOff0]
    let qb1 = wArr[bs + qsOff1]
    let sc0 = wArr[bs + sub]
    acc = acc + (d * cfloat(sc0 and 0x0F'u8) * cfloat(qb0 and 3'u8) - dm * cfloat(sc0 shr 4)) * xArr[eb + laneId]
    let sc1 = wArr[bs + 2'i32 + sub]
    acc = acc + (d * cfloat(sc1 and 0x0F'u8) * cfloat((qb0 shr 2) and 3'u8) - dm * cfloat(sc1 shr 4)) * xArr[eb + laneId + 32'i32]
    let sc2 = wArr[bs + 4'i32 + sub]
    acc = acc + (d * cfloat(sc2 and 0x0F'u8) * cfloat((qb0 shr 4) and 3'u8) - dm * cfloat(sc2 shr 4)) * xArr[eb + laneId + 64'i32]
    let sc3 = wArr[bs + 6'i32 + sub]
    acc = acc + (d * cfloat(sc3 and 0x0F'u8) * cfloat((qb0 shr 6) and 3'u8) - dm * cfloat(sc3 shr 4)) * xArr[eb + laneId + 96'i32]
    let sc4 = wArr[bs + 8'i32 + sub]
    acc = acc + (d * cfloat(sc4 and 0x0F'u8) * cfloat(qb1 and 3'u8) - dm * cfloat(sc4 shr 4)) * xArr[eb + laneId + 128'i32]
    let sc5 = wArr[bs + 10'i32 + sub]
    acc = acc + (d * cfloat(sc5 and 0x0F'u8) * cfloat((qb1 shr 2) and 3'u8) - dm * cfloat(sc5 shr 4)) * xArr[eb + laneId + 160'i32]
    let sc6 = wArr[bs + 12'i32 + sub]
    acc = acc + (d * cfloat(sc6 and 0x0F'u8) * cfloat((qb1 shr 4) and 3'u8) - dm * cfloat(sc6 shr 4)) * xArr[eb + laneId + 192'i32]
    let sc7 = wArr[bs + 14'i32 + sub]
    acc = acc + (d * cfloat(sc7 and 0x0F'u8) * cfloat((qb1 shr 6) and 3'u8) - dm * cfloat(sc7 shr 4)) * xArr[eb + laneId + 224'i32]
    blkIdx = blkIdx + 1'i32

  acc = acc + hippoShflDown(acc, 16)
  acc = acc + hippoShflDown(acc, 8)
  acc = acc + hippoShflDown(acc, 4)
  acc = acc + hippoShflDown(acc, 2)
  acc = acc + hippoShflDown(acc, 1)
  if laneId == 0'i32:
    outArr[row] = acc

proc linearQ3KWarpKernel(dst: ptr cfloat, x: ptr cfloat, w: ptr uint8,
                         inDim: cint, outDim: cint) {.hippoGlobal.} =
  ## Warp-per-row Q3_K GEMV. Ported from naive backend's proven-correct kernel.
  let tid = cint(threadIdx.x)
  let row = cint(blockIdx.x)
  if row >= outDim: return

  let wArr = cast[ptr UncheckedArray[uint8]](w)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  let nBlocksPerRow = inDim div 256'i32
  let rowSizeBytes = nBlocksPerRow * 110'i32
  let rowBase = row * rowSizeBytes

  let sub = (tid shr 4'i32) and 1'i32
  let qsOff0 = 32'i32 + sub * 16'i32 + (tid and 15'i32)
  let qsOff1 = 64'i32 + sub * 16'i32 + (tid and 15'i32)
  let hmOff = sub * 16'i32 + (tid and 15'i32)

  var acc: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx < nBlocksPerRow:
    let bs = rowBase + blkIdx * 110'i32
    let eb = blkIdx * 256'i32
    let dRaw = uint16(wArr[bs + 108'i32]) or (uint16(wArr[bs + 109'i32]) shl 8)
    let dAll = hippoHalfToFloat(dRaw)
    let qb0 = wArr[bs + qsOff0]
    let qb1 = wArr[bs + qsOff1]
    let hmByte = cint(wArr[bs + hmOff])

    template q3kElem(scaleIdx: cint, qByte: untyped, qShift, hmBitPos, xOff: cint) {.dirty.} =
      block:
        let si = scaleIdx
        let big = si and 3'i32
        let ai = si shr 2'i32
        let sByteVal = cint(wArr[bs + 96'i32 + (ai and 1'i32) * 4'i32 + big])
        let tByteVal = cint(wArr[bs + 104'i32 + big])
        let low = (sByteVal shr ((ai shr 1'i32) * 4'i32)) and 0x0F'i32
        let high = ((tByteVal shr (ai * 2'i32)) and 0x03'i32) shl 4'i32
        let scByte = low or high
        let scSigned = (scByte xor 0x80'i32) - 0x80'i32
        let dl = dAll * cfloat(scSigned - 32'i32)
        let qval = cint((qByte shr qShift) and 3)
        let hm = 4'i32 - ((hmByte shr hmBitPos) and 1'i32) * 4'i32
        acc = acc + dl * cfloat(qval - hm) * xArr[eb + xOff]

    q3kElem(sub,            qb0, 0, 0, tid)
    q3kElem(2'i32 + sub,    qb0, 2, 1, tid + 32'i32)
    q3kElem(4'i32 + sub,    qb0, 4, 2, tid + 64'i32)
    q3kElem(6'i32 + sub,    qb0, 6, 3, tid + 96'i32)
    q3kElem(8'i32 + sub,    qb1, 0, 4, tid + 128'i32)
    q3kElem(10'i32 + sub,   qb1, 2, 5, tid + 160'i32)
    q3kElem(12'i32 + sub,   qb1, 4, 6, tid + 192'i32)
    q3kElem(14'i32 + sub,   qb1, 6, 7, tid + 224'i32)
    blkIdx = blkIdx + 1'i32

  acc = acc + hippoShflDown(acc, 16)
  acc = acc + hippoShflDown(acc, 8)
  acc = acc + hippoShflDown(acc, 4)
  acc = acc + hippoShflDown(acc, 2)
  acc = acc + hippoShflDown(acc, 1)
  if tid == 0'i32:
    outArr[row] = acc

proc ropeQKDecodeKernel(q: ptr cfloat, k: ptr cfloat,
                        theta: ptr cfloat,
                        nHeadQ: cint, nHeadK: cint, headDim: cint,
                        ropeDim: cint, pos: cint) {.hippoGlobal.} =
  let tid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  let halfRope = ropeDim div 2
  let totalPairs = (nHeadQ + nHeadK) * halfRope
  if tid >= totalPairs: return

  let thetaArr = cast[ptr UncheckedArray[cfloat]](theta)
  let isK = tid >= nHeadQ * halfRope
  let head = if isK: (tid - nHeadQ * halfRope) div halfRope
             else: tid div halfRope
  let pairIdx = if isK: (tid - nHeadQ * halfRope) mod halfRope
                else: tid mod halfRope

  let basePtr = if isK: cast[ptr UncheckedArray[cfloat]](k)
                else: cast[ptr UncheckedArray[cfloat]](q)
  let offset = head * headDim + pairIdx

  let freq = thetaArr[pairIdx] * cfloat(pos)
  let cosVal = cosf(freq)
  let sinVal = sinf(freq)

  let v0 = basePtr[offset]
  let v1 = basePtr[offset + halfRope]
  basePtr[offset] = v0 * cosVal - v1 * sinVal
  basePtr[offset + halfRope] = v0 * sinVal + v1 * cosVal

proc storeKVPairKernel(kCache: ptr cfloat, kSrc: ptr cfloat,
                       vCache: ptr cfloat, vSrc: ptr cfloat,
                       kvDim: cint, cacheCols: cint,
                       pos: cint) {.hippoGlobal.} =
  let tid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  if tid < kvDim:
    let kc = cast[ptr UncheckedArray[cfloat]](kCache)
    let vc = cast[ptr UncheckedArray[cfloat]](vCache)
    let ks = cast[ptr UncheckedArray[cfloat]](kSrc)
    let vs = cast[ptr UncheckedArray[cfloat]](vSrc)
    kc[tid * cacheCols + pos] = ks[tid]
    vc[tid * cacheCols + pos] = vs[tid]

proc attentionDecodeKernel(dst: ptr cfloat, q: ptr cfloat,
                           kCache: ptr cfloat, vCache: ptr cfloat,
                           nHead: cint, nHeadKv: cint, headDim: cint,
                           curLen: cint, cacheCols: cint) {.hippoGlobal.} =
  ## MVP: thread 0 does sequential online softmax attention per head.
  ## One block per head, only thread 0 active.
  let head = cint(blockIdx.x)
  if head >= nHead: return
  let tid = cint(threadIdx.x)
  if tid != 0: return

  let qArr = cast[ptr UncheckedArray[cfloat]](q)
  let kArr = cast[ptr UncheckedArray[cfloat]](kCache)
  let vArr = cast[ptr UncheckedArray[cfloat]](vCache)
  let dArr = cast[ptr UncheckedArray[cfloat]](dst)

  let qOff = head * headDim
  let kvHead = head div (nHead div nHeadKv)
  let kvOff = kvHead * headDim
  let scale = 1.0f / sqrtf(cfloat(headDim))

  var mS: cfloat = -1e30f
  var sE: cfloat = 0.0f
  {.emit: "float __attn_acc[64];".}
  {.emit: "for (int __i = 0; __i < 64; __i++) __attn_acc[__i] = 0.0f;".}

  var p: cint = 0
  while p < curLen:
    var sc: cfloat = 0.0f
    var dd: cint = 0
    while dd < headDim:
      sc = sc + qArr[qOff + dd] * kArr[(kvOff + dd) * cacheCols + p]
      dd = dd + 1
    sc = sc * scale
    if sc > mS:
      let corr = expf(mS - sc)
      sE = sE * corr + 1.0f
      {.emit: """
      for (int __i = 0; __i < `headDim`; __i++)
        __attn_acc[__i] = __attn_acc[__i] * `corr` + ((float*)(`vArr`))[((`kvOff` + __i) * `cacheCols` + `p`)];
      """.}
      mS = sc
    else:
      let w = expf(sc - mS)
      sE = sE + w
      {.emit: """
      for (int __i = 0; __i < `headDim`; __i++)
        __attn_acc[__i] += `w` * ((float*)(`vArr`))[((`kvOff` + __i) * `cacheCols` + `p`)];
      """.}
    p = p + 1

  let invSum = 1.0f / sE
  {.emit: """
  for (int __i = 0; __i < `headDim`; __i++)
    ((float*)(`dArr`))[`qOff` + __i] = __attn_acc[__i] * `invSum`;
  """.}

proc linearF32WarpKernel(dst: ptr cfloat, x: ptr cfloat, w: ptr cfloat,
                         inDim: cint, outDim: cint) {.hippoGlobal.} =
  ## Warp-per-row F32 GEMV for unquantized output weights.
  let row = cint(blockIdx.x)
  if row >= outDim: return
  let tid = cint(threadIdx.x)
  let wArr = cast[ptr UncheckedArray[cfloat]](w)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  var acc: cfloat = 0.0
  var i = tid
  while i < inDim:
    acc = acc + wArr[row * inDim + i] * xArr[i]
    i = i + cint(blockDim.x)
  acc = acc + hippoShflDown(acc, 16)
  acc = acc + hippoShflDown(acc, 8)
  acc = acc + hippoShflDown(acc, 4)
  acc = acc + hippoShflDown(acc, 2)
  acc = acc + hippoShflDown(acc, 1)
  if tid == 0'i32:
    outArr[row] = acc

proc siluMulKernel(gate: ptr cfloat, up: ptr cfloat,
                   dim: cint) {.hippoGlobal.} =
  let tid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  if tid < dim:
    let g = cast[ptr UncheckedArray[cfloat]](gate)
    let u = cast[ptr UncheckedArray[cfloat]](up)
    let x = g[tid]
    g[tid] = (x / (1.0f + expf(-x))) * u[tid]

# ---------------------------------------------------------------------------
# Host-side dispatch helpers
# ---------------------------------------------------------------------------

proc grid1d(n: int): Dim3 = newDim3(((n + BlockSize - 1) div BlockSize).uint32)
proc block1d(): Dim3 = newDim3(BlockSize.uint32)

proc gpuEmbedding(dst, weight: pointer, tokenId: int32) =
  var dstP = cast[ptr cfloat](dst)
  var wP = cast[ptr cfloat](weight)
  var tid = cint(tokenId)
  var nEmb = cint(ModelCfg.nEmb)
  hippoLaunchKernel(embeddingKernel,
    gridDim = grid1d(ModelCfg.nEmb), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(dstP, wP, tid, nEmb))

proc gpuRmsNorm(dst, src, weight: pointer) =
  var dstP = cast[ptr cfloat](dst)
  var srcP = cast[ptr cfloat](src)
  var wP = cast[ptr cfloat](weight)
  var dim = cint(ModelCfg.nEmb)
  var eps = cfloat(ModelCfg.rmsEps)
  hippoLaunchKernel(rmsnormKernel,
    gridDim = newDim3(1), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(dstP, srcP, wP, dim, eps))

proc gpuResidualRmsNorm(normOut, x, residual, weight: pointer) =
  var nP = cast[ptr cfloat](normOut)
  var xP = cast[ptr cfloat](x)
  var rP = cast[ptr cfloat](residual)
  var wP = cast[ptr cfloat](weight)
  var dim = cint(ModelCfg.nEmb)
  var eps = cfloat(ModelCfg.rmsEps)
  hippoLaunchKernel(residualRmsnormKernel,
    gridDim = newDim3(1), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(nP, xP, rP, wP, dim, eps))

proc gpuRopeQKDecode(q, k, theta: pointer, pos: int) =
  var qP = cast[ptr cfloat](q)
  var kP = cast[ptr cfloat](k)
  var tP = cast[ptr cfloat](theta)
  var nHQ = cint(ModelCfg.nHead)
  var nHK = cint(ModelCfg.nHeadKv)
  var hDim = cint(ModelCfg.headDim)
  var rDim = cint(ModelCfg.ropeDim)
  var p = cint(pos)
  let halfRope = ModelCfg.ropeDim div 2
  let totalPairs = (ModelCfg.nHead + ModelCfg.nHeadKv) * halfRope
  hippoLaunchKernel(ropeQKDecodeKernel,
    gridDim = grid1d(totalPairs), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(qP, kP, tP, nHQ, nHK, hDim, rDim, p))

proc gpuStoreKVPair(kCache, kSrc, vCache, vSrc: pointer, pos, cacheCols: int) =
  var kcP = cast[ptr cfloat](kCache)
  var ksP = cast[ptr cfloat](kSrc)
  var vcP = cast[ptr cfloat](vCache)
  var vsP = cast[ptr cfloat](vSrc)
  var kvd = cint(KvDim)
  var cc = cint(cacheCols)
  var p = cint(pos)
  hippoLaunchKernel(storeKVPairKernel,
    gridDim = grid1d(KvDim), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(kcP, ksP, vcP, vsP, kvd, cc, p))

proc gpuAttentionDecode(dst, q, kCache, vCache: pointer, curLen, cacheCols: int) =
  var dstP = cast[ptr cfloat](dst)
  var qP = cast[ptr cfloat](q)
  var kcP = cast[ptr cfloat](kCache)
  var vcP = cast[ptr cfloat](vCache)
  var nH = cint(ModelCfg.nHead)
  var nHKv = cint(ModelCfg.nHeadKv)
  var hDim = cint(ModelCfg.headDim)
  var cLen = cint(curLen)
  var cc = cint(cacheCols)
  hippoLaunchKernel(attentionDecodeKernel,
    gridDim = newDim3(ModelCfg.nHead.uint32), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(dstP, qP, kcP, vcP, nH, nHKv, hDim, cLen, cc))

proc gpuSiluMul(gate, up: pointer, dim: int) =
  var gP = cast[ptr cfloat](gate)
  var uP = cast[ptr cfloat](up)
  var d = cint(dim)
  hippoLaunchKernel(siluMulKernel,
    gridDim = grid1d(dim), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(gP, uP, d))

proc gpuAdd(dst, src: pointer, dim: int) =
  var dstP = cast[ptr cfloat](dst)
  var srcP = cast[ptr cfloat](src)
  var d = cint(dim)
  hippoLaunchKernel(addKernel,
    gridDim = grid1d(dim), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(dstP, srcP, d))

proc gpuLinearF32(dst, x, w: pointer, inDim, outDim: int) =
  var dstP = cast[ptr cfloat](dst)
  var xP = cast[ptr cfloat](x)
  var wP = cast[ptr cfloat](w)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  hippoLaunchKernel(linearF32WarpKernel,
    gridDim = newDim3(outDim.uint32), blockDim = newDim3(mkc.WarpSize.uint32),
    stream = mkStream,
    args = hippoArgs(dstP, xP, wP, iDim, oDim))

proc gpuLinearQ2K(dst, x, w: pointer, inDim, outDim: int) =
  var dstP = cast[ptr cfloat](dst)
  var xP = cast[ptr cfloat](x)
  var wP = cast[ptr uint8](w)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  hippoLaunchKernel(linearQ2KWarpKernel,
    gridDim = newDim3(outDim.uint32), blockDim = newDim3(mkc.WarpSize.uint32),
    stream = mkStream,
    args = hippoArgs(dstP, xP, wP, iDim, oDim))

proc gpuLinearQ3K(dst, x, w: pointer, inDim, outDim: int) =
  var dstP = cast[ptr cfloat](dst)
  var xP = cast[ptr cfloat](x)
  var wP = cast[ptr uint8](w)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  hippoLaunchKernel(linearQ3KWarpKernel,
    gridDim = newDim3(outDim.uint32), blockDim = newDim3(mkc.WarpSize.uint32),
    stream = mkStream,
    args = hippoArgs(dstP, xP, wP, iDim, oDim))

proc gpuLinear(dst, x: pointer, w: MkWeight, inDim, outDim: int) =
  case w.qtype
  of GgmlTypeF32.int32: gpuLinearF32(dst, x, w.p, inDim, outDim)
  of GgmlTypeQ2K.int32: gpuLinearQ2K(dst, x, w.p, inDim, outDim)
  of GgmlTypeQ3K.int32: gpuLinearQ3K(dst, x, w.p, inDim, outDim)
  else: raise newException(ValueError, "unsupported qtype for gpuLinear: " & $w.qtype)

# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

proc quantRowSize(nCols: int, qtype: int32): int =
  case qtype
  of GgmlTypeQ2K.int32: rowSizeQ2K(nCols)
  of GgmlTypeQ3K.int32: rowSizeQ3K(nCols)
  of GgmlTypeQ4K.int32: rowSizeQ4K(nCols)
  of GgmlTypeQ6K.int32: rowSizeQ6K(nCols)
  of GgmlTypeQ8_0.int32: rowSizeQ8_0(nCols)
  else: raise newException(ValueError, "unsupported quant type: " & $qtype)

proc uploadQuantRaw(m: var Model, tensorName: string): MkWeight =
  let info = m.infos[tensorName]
  let dataPtr = tensorDataPtr(m.gguf, info)
  let nCols = int(info.ne[0])
  let nRows = tensorElemCount(info) div nCols
  let qtype = info.elemType.int32
  let rowSize = quantRowSize(nCols, qtype)
  let totalBytes = rowSize * nRows
  when defined(debugMegakernel):
    echo "[mk] upload ", tensorName, " type=", qtype, " ne=", info.ne[0..1],
         " nRows=", nRows, " bytes=", totalBytes
  let alloc = hippoMalloc(totalBytes)
  hippoMemcpyAsync(alloc.p, dataPtr, totalBytes, HippoMemcpyHostToDevice, mkStream)
  mkWeightAllocs.add(alloc)
  MkWeight(p: alloc.p, qtype: qtype)

proc uploadAsF32(m: var Model, tensorName: string): MkWeight =
  let t = m.getTensor(tensorName)
  let bytes = t.data.len * sizeof(float32)
  let alloc = hippoMalloc(bytes)
  hippoMemcpyAsync(alloc.p, addr t.data[0], bytes, HippoMemcpyHostToDevice, mkStream)
  mkWeightAllocs.add(alloc)
  MkWeight(p: alloc.p, qtype: GgmlTypeF32.int32)

proc uploadF32(m: var Model, tensorName: string): pointer =
  let t = m.getTensor(tensorName)
  let bytes = t.data.len * sizeof(float32)
  let alloc = hippoMalloc(bytes)
  hippoMemcpyAsync(alloc.p, addr t.data[0], bytes, HippoMemcpyHostToDevice, mkStream)
  mkWeightAllocs.add(alloc)
  alloc.p

proc uploadF16AsF32(m: var Model, tensorName: string): pointer =
  let info = m.infos[tensorName]
  let nElems = tensorElemCount(info)
  let bytes = nElems * sizeof(float32)
  let dataPtr = tensorDataPtr(m.gguf, info)
  var f32Buf = newSeq[float32](nElems)
  let src = cast[ptr UncheckedArray[uint16]](dataPtr)
  for i in 0 ..< nElems:
    f32Buf[i] = quant.halfToFloat(src[i])
  let alloc = hippoMalloc(bytes)
  hippoMemcpyAsync(alloc.p, addr f32Buf[0], bytes, HippoMemcpyHostToDevice, mkStream)
  mkWeightAllocs.add(alloc)
  alloc.p

proc loadF32Ptr(m: var Model, tensorName: string): pointer =
  let info = m.infos[tensorName]
  if info.elemType == GgmlTypeF16:
    return uploadF16AsF32(m, tensorName)
  else:
    return uploadF32(m, tensorName)

# ---------------------------------------------------------------------------
# Backend interface: initKvCache
# ---------------------------------------------------------------------------

proc initKvCache*(hp: HParams, maxLen: int): KvCache =
  result.maxLen = maxLen
  result.curLen = 0
  result.nHeadKv = hp.nHeadKv
  result.headDim = hp.headDim
  result.k = newSeq[Tensor](hp.nLayer)
  result.v = newSeq[Tensor](hp.nLayer)
  # GPU KV cache allocated in loadModelBackend

# ---------------------------------------------------------------------------
# Backend interface: loadModelBackend
# ---------------------------------------------------------------------------

proc loadModelBackend*(m: var Model, hp: HParams) =
  if mkInitialized: return
  mkStream = hippoStreamCreate()

  # Verify GGUF matches static config
  doAssert hp.arch == "llama", "megakernel MVP only supports llama arch, got: " & hp.arch
  doAssert hp.nEmb == ModelCfg.nEmb, &"nEmb mismatch: GGUF={hp.nEmb} config={ModelCfg.nEmb}"
  doAssert hp.nLayer == ModelCfg.nLayers, &"nLayer mismatch: GGUF={hp.nLayer} config={ModelCfg.nLayers}"
  doAssert hp.nHead == ModelCfg.nHead, &"nHead mismatch: GGUF={hp.nHead} config={ModelCfg.nHead}"
  doAssert hp.nHeadKv == ModelCfg.nHeadKv, &"nHeadKv mismatch: GGUF={hp.nHeadKv} config={ModelCfg.nHeadKv}"
  doAssert hp.nVocab == ModelCfg.nVocab, &"nVocab mismatch: GGUF={hp.nVocab} config={ModelCfg.nVocab}"

  # Token embedding
  mkWeights.tokEmb = loadF32Ptr(m, "token_embd.weight")

  # Per-layer weights
  for layer in 0 ..< ModelCfg.nLayers:
    let lp = "blk." & $layer & "."
    mkWeights.layers[layer].attnNorm = loadF32Ptr(m, lp & "attn_norm.weight")
    mkWeights.layers[layer].ffnNorm = loadF32Ptr(m, lp & "ffn_norm.weight")
    mkWeights.layers[layer].wq = uploadQuantRaw(m, lp & "attn_q.weight")
    mkWeights.layers[layer].wk = uploadQuantRaw(m, lp & "attn_k.weight")
    mkWeights.layers[layer].wv = uploadQuantRaw(m, lp & "attn_v.weight")
    mkWeights.layers[layer].wo = uploadQuantRaw(m, lp & "attn_output.weight")
    mkWeights.layers[layer].wGate = uploadQuantRaw(m, lp & "ffn_gate.weight")
    mkWeights.layers[layer].wUp = uploadQuantRaw(m, lp & "ffn_up.weight")
    mkWeights.layers[layer].wDown = uploadQuantRaw(m, lp & "ffn_down.weight")

  # Output — always dequant to F32 for simplicity (avoid needing Q6K/Q3K GEMV kernels)
  mkWeights.outputNorm = loadF32Ptr(m, "output_norm.weight")
  let outName = if m.infos.hasKey("output.weight"): "output.weight"
                else: "token_embd.weight"
  mkWeights.outputWeight = uploadAsF32(m, outName)

  # RoPE theta
  let halfRope = ModelCfg.ropeDim div 2
  var thetaBuf = newSeq[float32](halfRope)
  for i in 0 ..< halfRope:
    thetaBuf[i] = pow(1.0'f32 / ModelCfg.ropeTheta.float32,
                      (2.0'f32 * i.float32) / ModelCfg.ropeDim.float32)
  let thetaAlloc = hippoMalloc(halfRope * sizeof(float32))
  hippoMemcpyAsync(thetaAlloc.p, addr thetaBuf[0], halfRope * sizeof(float32),
                    HippoMemcpyHostToDevice, mkStream)
  mkWeightAllocs.add(thetaAlloc)
  mkWeights.ropeTheta = thetaAlloc.p

  # Activation buffers
  for alloc in [hippoMalloc(MaxDim * sizeof(float32)),
                hippoMalloc(MaxDim * sizeof(float32)),
                hippoMalloc(MaxDim * sizeof(float32)),
                hippoMalloc(MaxDim * sizeof(float32)),
                hippoMalloc(MaxDim * sizeof(float32)),
                hippoMalloc(ModelCfg.nVocab * sizeof(float32))]:
    mkBufAllocs.add(alloc)
  mkBuf.act0 = mkBufAllocs[^6].p
  mkBuf.act1 = mkBufAllocs[^5].p
  mkBuf.scratch0 = mkBufAllocs[^4].p
  mkBuf.scratch1 = mkBufAllocs[^3].p
  mkBuf.scratch2 = mkBufAllocs[^2].p
  mkBuf.logits = mkBufAllocs[^1].p

  # KV cache
  mkMaxLen = m.hparams.nCtx
  if mkMaxLen <= 0: mkMaxLen = 2048
  for layer in 0 ..< ModelCfg.nLayers:
    let kAlloc = hippoMalloc(KvDim * mkMaxLen * sizeof(float32))
    let vAlloc = hippoMalloc(KvDim * mkMaxLen * sizeof(float32))
    mkBufAllocs.add(kAlloc)
    mkBufAllocs.add(vAlloc)
    mkBuf.kvK[layer] = kAlloc.p
    mkBuf.kvV[layer] = vAlloc.p

  hippoStreamSynchronize(mkStream)
  mkInitialized = true
  echo &"[megakernel] Loaded {ModelCfg.nLayers} layers, nEmb={ModelCfg.nEmb}, Q2K"
  echo &"[megakernel] Machine: {Machine.hostname}, {Machine.gpu.cuCount} CU"

proc unloadModelBackend*() =
  if mkInitialized:
    mkWeightAllocs = @[]
    mkBufAllocs = @[]
    hippoStreamDestroy(mkStream)
    mkInitialized = false

# ---------------------------------------------------------------------------
# Forward decode (individual kernel launches — correctness path)
# ---------------------------------------------------------------------------

proc forwardDecodeInternal(token: int32, curLen: int): seq[float32] =
  let cacheCols = mkMaxLen

  when defined(debugMegakernel):
    echo "[mk] forward token=", token, " curLen=", curLen
  gpuEmbedding(mkBuf.act0, mkWeights.tokEmb, token)

  for layer in 0 ..< ModelCfg.nLayers:
    let lw = mkWeights.layers[layer]
    let kvK = mkBuf.kvK[layer]
    let kvV = mkBuf.kvV[layer]

    if layer == 0:
      gpuRmsNorm(mkBuf.act1, mkBuf.act0, lw.attnNorm)

    gpuLinear(mkBuf.scratch0, mkBuf.act1, lw.wq, ModelCfg.nEmb, QDim)
    gpuLinear(mkBuf.scratch1, mkBuf.act1, lw.wk, ModelCfg.nEmb, KvDim)
    gpuLinear(mkBuf.scratch2, mkBuf.act1, lw.wv, ModelCfg.nEmb, KvDim)
    gpuRopeQKDecode(mkBuf.scratch0, mkBuf.scratch1, mkWeights.ropeTheta, curLen)
    gpuStoreKVPair(kvK, mkBuf.scratch1, kvV, mkBuf.scratch2, curLen, cacheCols)
    gpuAttentionDecode(mkBuf.scratch1, mkBuf.scratch0, kvK, kvV, curLen + 1, cacheCols)
    gpuLinear(mkBuf.scratch0, mkBuf.scratch1, lw.wo, QDim, ModelCfg.nEmb)

    gpuResidualRmsNorm(mkBuf.act1, mkBuf.act0, mkBuf.scratch0, lw.ffnNorm)
    gpuLinear(mkBuf.scratch0, mkBuf.act1, lw.wGate, ModelCfg.nEmb, ModelCfg.ffnDim)
    gpuLinear(mkBuf.scratch1, mkBuf.act1, lw.wUp, ModelCfg.nEmb, ModelCfg.ffnDim)
    gpuSiluMul(mkBuf.scratch0, mkBuf.scratch1, ModelCfg.ffnDim)
    gpuLinear(mkBuf.scratch1, mkBuf.scratch0, lw.wDown, ModelCfg.ffnDim, ModelCfg.nEmb)

    if layer < ModelCfg.nLayers - 1:
      gpuResidualRmsNorm(mkBuf.act1, mkBuf.act0, mkBuf.scratch1,
                         mkWeights.layers[layer + 1].attnNorm)
    else:
      gpuAdd(mkBuf.act0, mkBuf.scratch1, ModelCfg.nEmb)

  gpuRmsNorm(mkBuf.act1, mkBuf.act0, mkWeights.outputNorm)
  gpuLinear(mkBuf.logits, mkBuf.act1, mkWeights.outputWeight,
            ModelCfg.nEmb, ModelCfg.nVocab)

  hippoStreamSynchronize(mkStream)
  result = newSeq[float32](ModelCfg.nVocab)
  hippoMemcpy(addr result[0], mkBuf.logits, ModelCfg.nVocab * sizeof(float32),
              HippoMemcpyDeviceToHost)

# ---------------------------------------------------------------------------
# Backend interface: forward functions
# ---------------------------------------------------------------------------

proc forwardPrefill*(m: var Model, tokens: seq[int32], cache: var KvCache): Tensor =
  var logits: seq[float32]
  for i, tok in tokens:
    logits = forwardDecodeInternal(tok, cache.curLen)
    inc cache.curLen
  result = Tensor(shape: @[1, ModelCfg.nVocab], data: logits)

proc forwardDecode*(m: var Model, token: int32, cache: var KvCache): Tensor =
  let logits = forwardDecodeInternal(token, cache.curLen)
  inc cache.curLen
  result = Tensor(shape: @[1, ModelCfg.nVocab], data: logits)

proc forwardDecodeToken*(m: var Model, token: int32, cache: var KvCache): int32 =
  let logits = forwardDecodeInternal(token, cache.curLen)
  inc cache.curLen
  # Argmax
  var maxVal = logits[0]
  result = 0
  for i in 1 ..< logits.len:
    if logits[i] > maxVal:
      maxVal = logits[i]
      result = int32(i)
