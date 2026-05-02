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

{.passC: "-ffast-math".}

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
    actQ8: pointer          # Q8_1 quantized activation buffer [(MaxDim/32)*36 bytes]
    logits: pointer         # [nVocab] f32
    argmaxResult: pointer   # [1] cint — GPU-side argmax result
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
  devWeightsPtr: pointer
  devBufsPtr: pointer

type
  DecodeStepConfig = object
    tokenId: cint
    pos: cint
    seqLen: cint
    cacheCols: cint

var
  mkStepConfig: DecodeStepConfig
  mkStepConfigDev: pointer
  mkStepConfigAlloc: HippoAllocRef
  mkGraphExec: hipGraphExec_t
  mkGraphCaptured: bool = false
  mkBenchEvent0: HippoEvent
  mkBenchEvent1: HippoEvent
  mkBenchEventsCreated: bool = false

# ---------------------------------------------------------------------------
# Q2K constants
# ---------------------------------------------------------------------------

proc gridSync() {.importcpp: """
  do {
    cooperative_groups::grid_group __hl_grid = cooperative_groups::this_grid();
    __hl_grid.sync();
  } while(0)""", header: "hip/hip_cooperative_groups.h", nodecl, used.}

const
  QK_K = 256
  BlockQ2KSize = 2 + 2 + (QK_K div 16) + (QK_K div 4)  # 84 bytes
  QK8_1 = 32'i32
  BlockQ8_1Size = 36 # 4 bytes (d,s fp16 pair) + 32 bytes (int8 qs)

# ---------------------------------------------------------------------------
# Shared GPU templates
# ---------------------------------------------------------------------------

template warpReduceSum(val: var cfloat) {.dirty.} =
  val = val + hippoShflDown(val, 16)
  val = val + hippoShflDown(val, 8)
  val = val + hippoShflDown(val, 4)
  val = val + hippoShflDown(val, 2)
  val = val + hippoShflDown(val, 1)

template warpReduceMax(val: var cfloat) {.dirty.} =
  val = hippoFmaxf(val, hippoShflDown(val, 16))
  val = hippoFmaxf(val, hippoShflDown(val, 8))
  val = hippoFmaxf(val, hippoShflDown(val, 4))
  val = hippoFmaxf(val, hippoShflDown(val, 2))
  val = hippoFmaxf(val, hippoShflDown(val, 1))

template readHalf(arr: untyped, offset: cint): cfloat =
  hippoHalfToFloat(uint16(arr[offset]) or (uint16(arr[offset + 1'i32]) shl 8))

template q3kDecodeScale(wArr: untyped, bs: cint, scaleIdx: cint): cint =
  block:
    let si = scaleIdx
    let big = si and 3'i32
    let ai = si shr 2'i32
    let sByteVal = cint(wArr[bs + 96'i32 + (ai and 1'i32) * 4'i32 + big])
    let tByteVal = cint(wArr[bs + 104'i32 + big])
    let low = (sByteVal shr ((ai shr 1'i32) * 4'i32)) and 0x0F'i32
    let high = ((tByteVal shr (ai * 2'i32)) and 0x03'i32) shl 4'i32
    let scByte = low or high
    (scByte xor 0x80'i32) - 0x80'i32

template q3kDecodeScaleVec(raw96, raw100, raw104: uint32, scaleIdx: cint): cint =
  ## Decode a Q3K 6-bit signed scale from preloaded uint32 values.
  block:
    let si = scaleIdx
    let big = si and 3'i32
    let ai = si shr 2'i32
    let sRaw = if (ai and 1'i32) == 0: raw96 else: raw100
    let sByteVal = cint((sRaw shr (uint32(big) * 8'u32)) and 0xFF'u32)
    let tByteVal = cint((raw104 shr (uint32(big) * 8'u32)) and 0xFF'u32)
    let low = (sByteVal shr ((ai shr 1'i32) * 4'i32)) and 0x0F'i32
    let high = ((tByteVal shr (ai * 2'i32)) and 0x03'i32) shl 4'i32
    let scByte = low or high
    (scByte xor 0x80'i32) - 0x80'i32

template q3kElem(accVar: var cfloat, wArr: untyped, bs: cint, dAll: untyped,
                 scaleIdx: cint, qByte: untyped, qShift: cint,
                 hmByte: untyped, hmBitPos: cint,
                 xArr: untyped, xIdx: cint) {.dirty.} =
  block:
    let scSigned = q3kDecodeScale(wArr, bs, scaleIdx)
    let qval = cint((qByte shr qShift) and 3)
    let hm = 4'i32 - ((hmByte shr hmBitPos) and 1'i32) * 4'i32
    let intProd = (scSigned - 32'i32) * (qval - hm)
    accVar = accVar + dAll * cfloat(intProd) * xArr[xIdx]

template q3kElemVec(accVar: var cfloat, raw96, raw100, raw104: uint32, dAll: untyped,
                    scaleIdx: cint, qByte: untyped, qShift: cint,
                    hmByte: untyped, hmBitPos: cint,
                    xArr: untyped, xIdx: cint) {.dirty.} =
  block:
    let scSigned = q3kDecodeScaleVec(raw96, raw100, raw104, scaleIdx)
    let qval = cint((qByte shr qShift) and 3)
    let hm = 4'i32 - ((hmByte shr hmBitPos) and 1'i32) * 4'i32
    let intProd = (scSigned - 32'i32) * (qval - hm)
    accVar = accVar + dAll * cfloat(intProd) * xArr[xIdx]

template q3kElemVecX(accVar: var cfloat, raw96, raw100, raw104: uint32, dAll: cfloat,
                     scaleIdx: cint, qByte: untyped, qShift: cint,
                     hmByte: cint, hmBitPos: cint,
                     xVal: cfloat) {.dirty.} =
  block:
    let scSigned = q3kDecodeScaleVec(raw96, raw100, raw104, scaleIdx)
    let qval = cint((qByte shr qShift) and 3)
    let hm = 4'i32 - ((hmByte shr hmBitPos) and 1'i32) * 4'i32
    let intProd = (scSigned - 32'i32) * (qval - hm)
    accVar = accVar + dAll * cfloat(intProd) * xVal

template q3kAccumBlockVec(accVar: var cfloat, wArr: untyped, bs: cint,
                           sub: cint, qsOff0, qsOff1, hmOff: cint,
                           xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7: cfloat) {.dirty.} =
  block:
    let dAll = readHalf(wArr, bs + 108'i32)
    let qb0 = wArr[bs + qsOff0]
    let qb1 = wArr[bs + qsOff1]
    let hmByte = cint(wArr[bs + hmOff])
    let r96 = hippoLoadU32(addr wArr[bs + 96'i32])
    let r100 = hippoLoadU32(addr wArr[bs + 100'i32])
    let r104 = hippoLoadU32(addr wArr[bs + 104'i32])
    q3kElemVecX(accVar, r96, r100, r104, dAll, sub,            qb0, 0, hmByte, 0, xv0)
    q3kElemVecX(accVar, r96, r100, r104, dAll, 2'i32 + sub,    qb0, 2, hmByte, 1, xv1)
    q3kElemVecX(accVar, r96, r100, r104, dAll, 4'i32 + sub,    qb0, 4, hmByte, 2, xv2)
    q3kElemVecX(accVar, r96, r100, r104, dAll, 6'i32 + sub,    qb0, 6, hmByte, 3, xv3)
    q3kElemVecX(accVar, r96, r100, r104, dAll, 8'i32 + sub,    qb1, 0, hmByte, 4, xv4)
    q3kElemVecX(accVar, r96, r100, r104, dAll, 10'i32 + sub,   qb1, 2, hmByte, 5, xv5)
    q3kElemVecX(accVar, r96, r100, r104, dAll, 12'i32 + sub,   qb1, 4, hmByte, 6, xv6)
    q3kElemVecX(accVar, r96, r100, r104, dAll, 14'i32 + sub,   qb1, 6, hmByte, 7, xv7)

template q2kAccumBlockX(acc: var cfloat, wArr: untyped, bs: cint,
                        d, dm: cfloat, qb0, qb1: uint8, sub: cint,
                        xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7: cfloat) {.dirty.} =
  block:
    let subShift = uint32(sub) * 8'u32
    let sc32_0 = hippoLoadU32(addr wArr[bs])
    let sc32_1 = hippoLoadU32(addr wArr[bs + 4'i32])
    let sc32_2 = hippoLoadU32(addr wArr[bs + 8'i32])
    let sc32_3 = hippoLoadU32(addr wArr[bs + 12'i32])
    let sc0 = uint8((sc32_0 shr subShift) and 0xFF'u32)
    let sc1 = uint8((sc32_0 shr (16'u32 + subShift)) and 0xFF'u32)
    let sc2 = uint8((sc32_1 shr subShift) and 0xFF'u32)
    let sc3 = uint8((sc32_1 shr (16'u32 + subShift)) and 0xFF'u32)
    let sc4 = uint8((sc32_2 shr subShift) and 0xFF'u32)
    let sc5 = uint8((sc32_2 shr (16'u32 + subShift)) and 0xFF'u32)
    let sc6 = uint8((sc32_3 shr subShift) and 0xFF'u32)
    let sc7 = uint8((sc32_3 shr (16'u32 + subShift)) and 0xFF'u32)
    acc = acc + (d * cfloat(sc0 and 0x0F'u8) * cfloat(qb0 and 3'u8) - dm * cfloat(sc0 shr 4)) * xv0
    acc = acc + (d * cfloat(sc1 and 0x0F'u8) * cfloat((qb0 shr 2) and 3'u8) - dm * cfloat(sc1 shr 4)) * xv1
    acc = acc + (d * cfloat(sc2 and 0x0F'u8) * cfloat((qb0 shr 4) and 3'u8) - dm * cfloat(sc2 shr 4)) * xv2
    acc = acc + (d * cfloat(sc3 and 0x0F'u8) * cfloat((qb0 shr 6) and 3'u8) - dm * cfloat(sc3 shr 4)) * xv3
    acc = acc + (d * cfloat(sc4 and 0x0F'u8) * cfloat(qb1 and 3'u8) - dm * cfloat(sc4 shr 4)) * xv4
    acc = acc + (d * cfloat(sc5 and 0x0F'u8) * cfloat((qb1 shr 2) and 3'u8) - dm * cfloat(sc5 shr 4)) * xv5
    acc = acc + (d * cfloat(sc6 and 0x0F'u8) * cfloat((qb1 shr 4) and 3'u8) - dm * cfloat(sc6 shr 4)) * xv6
    acc = acc + (d * cfloat(sc7 and 0x0F'u8) * cfloat((qb1 shr 6) and 3'u8) - dm * cfloat(sc7 shr 4)) * xv7

template q2kAccumBlock(acc: var cfloat, wArr: untyped, bs: cint,
                       d, dm: cfloat, qb0, qb1: uint8, sub: cint,
                       xSrc: untyped, eb: cint, laneOff: cint) {.dirty.} =
  block:
    let subShift = uint32(sub) * 8'u32
    let sc32_0 = hippoLoadU32(addr wArr[bs])
    let sc32_1 = hippoLoadU32(addr wArr[bs + 4'i32])
    let sc32_2 = hippoLoadU32(addr wArr[bs + 8'i32])
    let sc32_3 = hippoLoadU32(addr wArr[bs + 12'i32])
    let sc0 = uint8((sc32_0 shr subShift) and 0xFF'u32)
    let sc1 = uint8((sc32_0 shr (16'u32 + subShift)) and 0xFF'u32)
    let sc2 = uint8((sc32_1 shr subShift) and 0xFF'u32)
    let sc3 = uint8((sc32_1 shr (16'u32 + subShift)) and 0xFF'u32)
    let sc4 = uint8((sc32_2 shr subShift) and 0xFF'u32)
    let sc5 = uint8((sc32_2 shr (16'u32 + subShift)) and 0xFF'u32)
    let sc6 = uint8((sc32_3 shr subShift) and 0xFF'u32)
    let sc7 = uint8((sc32_3 shr (16'u32 + subShift)) and 0xFF'u32)
    acc = acc + (d * cfloat(sc0 and 0x0F'u8) * cfloat(qb0 and 3'u8) - dm * cfloat(sc0 shr 4)) * xSrc[eb + laneOff]
    acc = acc + (d * cfloat(sc1 and 0x0F'u8) * cfloat((qb0 shr 2) and 3'u8) - dm * cfloat(sc1 shr 4)) * xSrc[eb + laneOff + 32'i32]
    acc = acc + (d * cfloat(sc2 and 0x0F'u8) * cfloat((qb0 shr 4) and 3'u8) - dm * cfloat(sc2 shr 4)) * xSrc[eb + laneOff + 64'i32]
    acc = acc + (d * cfloat(sc3 and 0x0F'u8) * cfloat((qb0 shr 6) and 3'u8) - dm * cfloat(sc3 shr 4)) * xSrc[eb + laneOff + 96'i32]
    acc = acc + (d * cfloat(sc4 and 0x0F'u8) * cfloat(qb1 and 3'u8) - dm * cfloat(sc4 shr 4)) * xSrc[eb + laneOff + 128'i32]
    acc = acc + (d * cfloat(sc5 and 0x0F'u8) * cfloat((qb1 shr 2) and 3'u8) - dm * cfloat(sc5 shr 4)) * xSrc[eb + laneOff + 160'i32]
    acc = acc + (d * cfloat(sc6 and 0x0F'u8) * cfloat((qb1 shr 4) and 3'u8) - dm * cfloat(sc6 shr 4)) * xSrc[eb + laneOff + 192'i32]
    acc = acc + (d * cfloat(sc7 and 0x0F'u8) * cfloat((qb1 shr 6) and 3'u8) - dm * cfloat(sc7 shr 4)) * xSrc[eb + laneOff + 224'i32]

# ---------------------------------------------------------------------------
# GPU Kernels — individual launches for correctness verification
# ---------------------------------------------------------------------------

proc quantizeQ8_1Kernel(dst: ptr uint8, src: ptr cfloat, nElems: cint) {.hippoGlobal.} =
  ## Quantize float32 to Q8_1: 1 warp (32 threads) per Q8_1 block (32 elements).
  ## Q8_1 block layout: [d:fp16, s:fp16, qs:int8[32]] = 36 bytes
  let tid = cint(threadIdx.x)
  let warpId = cint(blockIdx.x) * (cint(blockDim.x) div 32'i32) + (tid div 32'i32)
  let laneId = tid and 31'i32
  let elemIdx = warpId * 32'i32 + laneId
  if elemIdx >= nElems: return

  let srcArr = cast[ptr UncheckedArray[cfloat]](src)
  let dstArr = cast[ptr UncheckedArray[uint8]](dst)
  let val = srcArr[elemIdx]

  # Warp reduce to find max absolute value
  var amax = hippoFabsf(val)
  warpReduceMax(amax)
  amax = hippoShfl(amax, 0)

  let d = amax / 127.0f
  let id = if amax > 0.0f: 127.0f / amax else: 0.0f
  let qi = cint(hippoRoundf(val * id))
  let qByte = cast[int8](max(-128'i32, min(127'i32, qi)))

  # Compute sum(qi * d) for zero-point correction
  var sumQd = cfloat(qi) * d
  warpReduceSum(sumQd)

  let blkBase = warpId * BlockQ8_1Size
  # Lane 0 writes d and s (as fp16 pair)
  if laneId == 0'i32:
    let dH = hippoFloatToHalf(d)
    let sH = hippoFloatToHalf(sumQd)
    dstArr[blkBase + 0] = uint8(dH and 0xFF'u16)
    dstArr[blkBase + 1] = uint8(dH shr 8)
    dstArr[blkBase + 2] = uint8(sH and 0xFF'u16)
    dstArr[blkBase + 3] = uint8(sH shr 8)
  # All lanes write their int8 quant value
  dstArr[blkBase + 4 + laneId] = cast[uint8](qByte)

proc embeddingKernel(dst: ptr cfloat, weight: ptr cfloat,
                     tokenId: cint, nEmb: cint) {.hippoGlobal.} =
  let tid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  if tid < nEmb:
    let w = cast[ptr UncheckedArray[cfloat]](weight)
    let d = cast[ptr UncheckedArray[cfloat]](dst)
    d[tid] = w[cint(tokenId) * nEmb + tid]

proc rmsnormKernel(dst: ptr cfloat, src: ptr cfloat, weight: ptr cfloat,
                   dim: cint, eps: cfloat) {.hippoGlobal.} =
  var warpSums {.hippoShared.}: array[8, cfloat]
  let tid = cint(threadIdx.x)
  let warpId = tid div 32'i32
  let laneId = tid mod 32'i32
  let s = cast[ptr UncheckedArray[cfloat]](src)
  let d = cast[ptr UncheckedArray[cfloat]](dst)
  let wt = cast[ptr UncheckedArray[cfloat]](weight)
  var sumSq: cfloat = 0.0
  var i = tid
  while i < dim:
    sumSq = sumSq + s[i] * s[i]
    i = i + cint(blockDim.x)
  warpReduceSum(sumSq)
  if laneId == 0'i32: warpSums[warpId] = sumSq
  hippoSyncthreads()
  if warpId == 0'i32:
    sumSq = if laneId < 8'i32: warpSums[laneId] else: 0.0f
    sumSq = sumSq + hippoShflDown(sumSq, 4)
    sumSq = sumSq + hippoShflDown(sumSq, 2)
    sumSq = sumSq + hippoShflDown(sumSq, 1)
    if laneId == 0'i32: warpSums[0] = sumSq
  hippoSyncthreads()
  let rms = 1.0f / sqrtf(warpSums[0] / cfloat(dim) + eps)
  i = tid
  while i < dim:
    d[i] = wt[i] * s[i] * rms
    i = i + cint(blockDim.x)

proc residualRmsnormKernel(normOut: ptr cfloat, x: ptr cfloat,
                           residual: ptr cfloat, weight: ptr cfloat,
                           dim: cint, eps: cfloat) {.hippoGlobal.} =
  var warpSums {.hippoShared.}: array[8, cfloat]
  let tid = cint(threadIdx.x)
  let warpId = tid div 32'i32
  let laneId = tid mod 32'i32
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let rArr = cast[ptr UncheckedArray[cfloat]](residual)
  let nArr = cast[ptr UncheckedArray[cfloat]](normOut)
  let wt = cast[ptr UncheckedArray[cfloat]](weight)
  var sumSq: cfloat = 0.0
  var i = tid
  while i < dim:
    let v = xArr[i] + rArr[i]
    xArr[i] = v
    sumSq = sumSq + v * v
    i = i + cint(blockDim.x)
  warpReduceSum(sumSq)
  if laneId == 0'i32: warpSums[warpId] = sumSq
  hippoSyncthreads()
  if warpId == 0'i32:
    sumSq = if laneId < 8'i32: warpSums[laneId] else: 0.0f
    sumSq = sumSq + hippoShflDown(sumSq, 4)
    sumSq = sumSq + hippoShflDown(sumSq, 2)
    sumSq = sumSq + hippoShflDown(sumSq, 1)
    if laneId == 0'i32: warpSums[0] = sumSq
  hippoSyncthreads()
  let rms = 1.0f / sqrtf(warpSums[0] / cfloat(dim) + eps)
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
    let ddm = hippoLoadU32(addr wArr[bs + 80'i32])
    let d = hippoHalfToFloat(uint16(ddm and 0xFFFF'u32))
    let dm = hippoHalfToFloat(uint16(ddm shr 16))
    let qb0 = wArr[bs + qsOff0]
    let qb1 = wArr[bs + qsOff1]
    q2kAccumBlock(acc, wArr, bs, d, dm, qb0, qb1, sub, xArr, eb, laneId)
    blkIdx = blkIdx + 1'i32

  warpReduceSum(acc)
  if laneId == 0'i32:
    outArr[row] = acc

proc linearQ2KDualWarpKernel(dst1: ptr cfloat, dst2: ptr cfloat, x: ptr cfloat,
                             w1: ptr uint8, w2: ptr uint8,
                             inDim: cint, outDim1: cint, outDim2: cint) {.hippoGlobal.} =
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let rawRow = cint(blockIdx.x) * warpsPerBlock + warpId
  let isSecond = rawRow >= outDim1
  let row = if isSecond: rawRow - outDim1 else: rawRow
  let outDim = if isSecond: outDim2 else: outDim1
  if row >= outDim: return
  let wArr = if isSecond: cast[ptr UncheckedArray[uint8]](w2)
             else: cast[ptr UncheckedArray[uint8]](w1)
  let outArr = if isSecond: cast[ptr UncheckedArray[cfloat]](dst2)
               else: cast[ptr UncheckedArray[cfloat]](dst1)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
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
    let ddm = hippoLoadU32(addr wArr[bs + 80'i32])
    let d = hippoHalfToFloat(uint16(ddm and 0xFFFF'u32))
    let dm = hippoHalfToFloat(uint16(ddm shr 16))
    let qb0 = wArr[bs + qsOff0]
    let qb1 = wArr[bs + qsOff1]
    q2kAccumBlock(acc, wArr, bs, d, dm, qb0, qb1, sub, xArr, eb, laneId)
    blkIdx = blkIdx + 1'i32
  warpReduceSum(acc)
  if laneId == 0'i32:
    outArr[row] = acc

proc linearQ2K2RowWarpKernel(dst: ptr cfloat, x: ptr cfloat, w: ptr uint8,
                             inDim: cint, outDim: cint) {.hippoGlobal.} =
  ## 2-row Q2K GEMV: each warp handles 2 output rows, sharing activation loads.
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let row0 = (cint(blockIdx.x) * warpsPerBlock + warpId) * 2'i32
  let row1 = row0 + 1'i32
  if row0 >= outDim: return

  let wArr = cast[ptr UncheckedArray[uint8]](w)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  let nBlocksPerRow = inDim div cint(QK_K)
  let rowSizeBytes = nBlocksPerRow * cint(BlockQ2KSize)
  let rowBase0 = row0 * rowSizeBytes
  let rowBase1 = row1 * rowSizeBytes
  let hasRow1 = row1 < outDim
  let sub = (laneId shr 4'i32) and 1'i32
  let qsOff0 = 16'i32 + sub * 16'i32 + (laneId and 15'i32)
  let qsOff1 = 48'i32 + sub * 16'i32 + (laneId and 15'i32)

  var acc0: cfloat = 0.0
  var acc1: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx < nBlocksPerRow:
    let eb = blkIdx * cint(QK_K)
    let xv0 = xArr[eb + laneId]
    let xv1 = xArr[eb + laneId + 32'i32]
    let xv2 = xArr[eb + laneId + 64'i32]
    let xv3 = xArr[eb + laneId + 96'i32]
    let xv4 = xArr[eb + laneId + 128'i32]
    let xv5 = xArr[eb + laneId + 160'i32]
    let xv6 = xArr[eb + laneId + 192'i32]
    let xv7 = xArr[eb + laneId + 224'i32]
    block:
      let bs = rowBase0 + blkIdx * cint(BlockQ2KSize)
      let ddm = hippoLoadU32(addr wArr[bs + 80'i32])
      let d = hippoHalfToFloat(uint16(ddm and 0xFFFF'u32))
      let dm = hippoHalfToFloat(uint16(ddm shr 16))
      let qb0 = wArr[bs + qsOff0]
      let qb1 = wArr[bs + qsOff1]
      q2kAccumBlockX(acc0, wArr, bs, d, dm, qb0, qb1, sub,
                      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7)
    if hasRow1:
      let bs = rowBase1 + blkIdx * cint(BlockQ2KSize)
      let ddm = hippoLoadU32(addr wArr[bs + 80'i32])
      let d = hippoHalfToFloat(uint16(ddm and 0xFFFF'u32))
      let dm = hippoHalfToFloat(uint16(ddm shr 16))
      let qb0 = wArr[bs + qsOff0]
      let qb1 = wArr[bs + qsOff1]
      q2kAccumBlockX(acc1, wArr, bs, d, dm, qb0, qb1, sub,
                      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7)
    blkIdx = blkIdx + 1'i32

  warpReduceSum(acc0)
  if laneId == 0'i32:
    outArr[row0] = acc0
  if hasRow1:
    warpReduceSum(acc1)
    if laneId == 0'i32:
      outArr[row1] = acc1

proc linearQ2KDual2RowWarpKernel(dst1: ptr cfloat, dst2: ptr cfloat, x: ptr cfloat,
                                  w1: ptr uint8, w2: ptr uint8,
                                  inDim: cint, outDim1: cint, outDim2: cint) {.hippoGlobal.} =
  ## 2-row dual Q2K: maps rawRow pairs across w1 (outDim1 rows) and w2 (outDim2 rows).
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let rawRow = (cint(blockIdx.x) * warpsPerBlock + warpId) * 2'i32
  let totalRows = outDim1 + outDim2
  if rawRow >= totalRows: return

  let isSecond0 = rawRow >= outDim1
  let row0 = if isSecond0: rawRow - outDim1 else: rawRow
  let w0 = if isSecond0: cast[ptr UncheckedArray[uint8]](w2)
           else: cast[ptr UncheckedArray[uint8]](w1)
  let out0 = if isSecond0: cast[ptr UncheckedArray[cfloat]](dst2)
             else: cast[ptr UncheckedArray[cfloat]](dst1)

  let rawRow1 = rawRow + 1'i32
  let hasRow1 = rawRow1 < totalRows
  let isSecond1 = rawRow1 >= outDim1
  let row1 = if isSecond1: rawRow1 - outDim1 else: rawRow1
  let w1Arr = if isSecond1: cast[ptr UncheckedArray[uint8]](w2)
              else: cast[ptr UncheckedArray[uint8]](w1)
  let out1 = if isSecond1: cast[ptr UncheckedArray[cfloat]](dst2)
             else: cast[ptr UncheckedArray[cfloat]](dst1)

  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let nBlocksPerRow = inDim div cint(QK_K)
  let rowSizeBytes = nBlocksPerRow * cint(BlockQ2KSize)
  let rowBase0 = row0 * rowSizeBytes
  let rowBase1 = row1 * rowSizeBytes
  let sub = (laneId shr 4'i32) and 1'i32
  let qsOff0 = 16'i32 + sub * 16'i32 + (laneId and 15'i32)
  let qsOff1 = 48'i32 + sub * 16'i32 + (laneId and 15'i32)

  var acc0: cfloat = 0.0
  var acc1: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx < nBlocksPerRow:
    let eb = blkIdx * cint(QK_K)
    let xv0 = xArr[eb + laneId]
    let xv1 = xArr[eb + laneId + 32'i32]
    let xv2 = xArr[eb + laneId + 64'i32]
    let xv3 = xArr[eb + laneId + 96'i32]
    let xv4 = xArr[eb + laneId + 128'i32]
    let xv5 = xArr[eb + laneId + 160'i32]
    let xv6 = xArr[eb + laneId + 192'i32]
    let xv7 = xArr[eb + laneId + 224'i32]
    block:
      let bs = rowBase0 + blkIdx * cint(BlockQ2KSize)
      let ddm = hippoLoadU32(addr w0[bs + 80'i32])
      let d = hippoHalfToFloat(uint16(ddm and 0xFFFF'u32))
      let dm = hippoHalfToFloat(uint16(ddm shr 16))
      let qb0 = w0[bs + qsOff0]
      let qb1 = w0[bs + qsOff1]
      q2kAccumBlockX(acc0, w0, bs, d, dm, qb0, qb1, sub,
                      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7)
    if hasRow1:
      let bs = rowBase1 + blkIdx * cint(BlockQ2KSize)
      let ddm = hippoLoadU32(addr w1Arr[bs + 80'i32])
      let d = hippoHalfToFloat(uint16(ddm and 0xFFFF'u32))
      let dm = hippoHalfToFloat(uint16(ddm shr 16))
      let qb0 = w1Arr[bs + qsOff0]
      let qb1 = w1Arr[bs + qsOff1]
      q2kAccumBlockX(acc1, w1Arr, bs, d, dm, qb0, qb1, sub,
                      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7)
    blkIdx = blkIdx + 1'i32

  warpReduceSum(acc0)
  if laneId == 0'i32:
    out0[row0] = acc0
  if hasRow1:
    warpReduceSum(acc1)
    if laneId == 0'i32:
      out1[row1] = acc1

proc linearQ3KWarpKernel(dst: ptr cfloat, x: ptr cfloat, w: ptr uint8,
                         inDim: cint, outDim: cint) {.hippoGlobal.} =
  ## Warp-per-row Q3_K GEMV.
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let row = cint(blockIdx.x) * warpsPerBlock + warpId
  let tid = laneId
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
  var acc2: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx + 1'i32 < nBlocksPerRow:
    let bs0 = rowBase + blkIdx * 110'i32
    let bs1 = rowBase + (blkIdx + 1'i32) * 110'i32
    let eb0 = blkIdx * 256'i32
    let eb1 = (blkIdx + 1'i32) * 256'i32
    let dAll0 = readHalf(wArr, bs0 + 108'i32)
    let dAll1 = readHalf(wArr, bs1 + 108'i32)
    let qb0a = wArr[bs0 + qsOff0]
    let qb1a = wArr[bs0 + qsOff1]
    let hmByte0 = cint(wArr[bs0 + hmOff])
    let qb0b = wArr[bs1 + qsOff0]
    let qb1b = wArr[bs1 + qsOff1]
    let hmByte1 = cint(wArr[bs1 + hmOff])
    let r96_0 = hippoLoadU32(addr wArr[bs0 + 96'i32])
    let r100_0 = hippoLoadU32(addr wArr[bs0 + 100'i32])
    let r104_0 = hippoLoadU32(addr wArr[bs0 + 104'i32])
    let r96_1 = hippoLoadU32(addr wArr[bs1 + 96'i32])
    let r100_1 = hippoLoadU32(addr wArr[bs1 + 100'i32])
    let r104_1 = hippoLoadU32(addr wArr[bs1 + 104'i32])

    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, sub,            qb0a, 0, hmByte0, 0, xArr, eb0 + tid)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, sub,            qb0b, 0, hmByte1, 0, xArr, eb1 + tid)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 2'i32 + sub,    qb0a, 2, hmByte0, 1, xArr, eb0 + tid + 32'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 2'i32 + sub,    qb0b, 2, hmByte1, 1, xArr, eb1 + tid + 32'i32)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 4'i32 + sub,    qb0a, 4, hmByte0, 2, xArr, eb0 + tid + 64'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 4'i32 + sub,    qb0b, 4, hmByte1, 2, xArr, eb1 + tid + 64'i32)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 6'i32 + sub,    qb0a, 6, hmByte0, 3, xArr, eb0 + tid + 96'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 6'i32 + sub,    qb0b, 6, hmByte1, 3, xArr, eb1 + tid + 96'i32)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 8'i32 + sub,    qb1a, 0, hmByte0, 4, xArr, eb0 + tid + 128'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 8'i32 + sub,    qb1b, 0, hmByte1, 4, xArr, eb1 + tid + 128'i32)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 10'i32 + sub,   qb1a, 2, hmByte0, 5, xArr, eb0 + tid + 160'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 10'i32 + sub,   qb1b, 2, hmByte1, 5, xArr, eb1 + tid + 160'i32)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 12'i32 + sub,   qb1a, 4, hmByte0, 6, xArr, eb0 + tid + 192'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 12'i32 + sub,   qb1b, 4, hmByte1, 6, xArr, eb1 + tid + 192'i32)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 14'i32 + sub,   qb1a, 6, hmByte0, 7, xArr, eb0 + tid + 224'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 14'i32 + sub,   qb1b, 6, hmByte1, 7, xArr, eb1 + tid + 224'i32)
    blkIdx = blkIdx + 2'i32

  while blkIdx < nBlocksPerRow:
    let bs = rowBase + blkIdx * 110'i32
    let eb = blkIdx * 256'i32
    let dAll = readHalf(wArr, bs + 108'i32)
    let qb0 = wArr[bs + qsOff0]
    let qb1 = wArr[bs + qsOff1]
    let hmByte = cint(wArr[bs + hmOff])
    let r96 = hippoLoadU32(addr wArr[bs + 96'i32])
    let r100 = hippoLoadU32(addr wArr[bs + 100'i32])
    let r104 = hippoLoadU32(addr wArr[bs + 104'i32])

    q3kElemVec(acc, r96, r100, r104, dAll, sub,            qb0, 0, hmByte, 0, xArr, eb + tid)
    q3kElemVec(acc, r96, r100, r104, dAll, 2'i32 + sub,    qb0, 2, hmByte, 1, xArr, eb + tid + 32'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 4'i32 + sub,    qb0, 4, hmByte, 2, xArr, eb + tid + 64'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 6'i32 + sub,    qb0, 6, hmByte, 3, xArr, eb + tid + 96'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 8'i32 + sub,    qb1, 0, hmByte, 4, xArr, eb + tid + 128'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 10'i32 + sub,   qb1, 2, hmByte, 5, xArr, eb + tid + 160'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 12'i32 + sub,   qb1, 4, hmByte, 6, xArr, eb + tid + 192'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 14'i32 + sub,   qb1, 6, hmByte, 7, xArr, eb + tid + 224'i32)
    blkIdx = blkIdx + 1'i32

  acc = acc + acc2
  warpReduceSum(acc)
  if tid == 0'i32:
    outArr[row] = acc

proc linearQ3KDp4aWarpKernel(dst: ptr cfloat, xQ8: ptr uint8, w: ptr uint8,
                              inDim: cint, outDim: cint) {.hippoGlobal.} =
  ## Q3K GEMV with dp4a: warp-per-row, uses pre-quantized Q8_1 activation.
  ## QI3_K=16 threads per Q3K block, QR3_K=4 dp4a per thread, 4 elems per dp4a.
  const QI3_K = 16'i32
  const QR3_K = 4'i32
  const QI8_1 = 8'i32

  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let row = cint(blockIdx.x) * warpsPerBlock + warpId
  let tid = laneId
  if row >= outDim: return

  let wArr = cast[ptr UncheckedArray[uint8]](w)
  let q8Arr = cast[ptr UncheckedArray[uint8]](xQ8)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)

  let nBlocksPerRow = inDim div 256'i32
  let rowSizeBytes = nBlocksPerRow * 110'i32
  let rowBase = row * rowSizeBytes

  let iqs = tid and (QI3_K - 1'i32)
  let bq8Offset = QR3_K * (iqs div (QI3_K div 2'i32))
  let scaleOffset = iqs - (iqs mod QI8_1) + (iqs mod QI8_1) div (QI8_1 div 2'i32)

  var sumf: cfloat = 0.0

  var blkIdx = tid div QI3_K
  while blkIdx < nBlocksPerRow:
    let bs = rowBase + blkIdx * 110'i32

    let d3 = readHalf(wArr, bs + 108'i32)

    let qsBase = bs + 32'i32 + iqs * 4'i32
    let vl = cint(hippoLoadU32(addr wArr[qsBase]))

    let hmBase = bs + (iqs mod (QI3_K div 2'i32)) * 4'i32
    let vhRaw = cint(hippoLoadU32(addr wArr[hmBase]))
    let vh = (not vhRaw) shr bq8Offset

    let q8BlockBase = blkIdx * 8'i32 + bq8Offset

    var blockSumf: cfloat = 0.0
    for i in 0'i32 ..< QR3_K:
      let isc = scaleOffset + 2'i32 * i

      let iscLow = isc mod 8'i32
      let scShiftLow = 4'i32 * (isc div 8'i32)
      let scLow = (cint(wArr[bs + 96'i32 + iscLow]) shr scShiftLow) and 0x0F'i32

      let iscHigh = isc mod 4'i32
      let scShiftHigh = 2'i32 * (isc div 4'i32)
      let scHigh = ((cint(wArr[bs + 96'i32 + 8'i32 + iscHigh]) shr scShiftHigh) and 3'i32) shl 4'i32

      let sc = (scLow or scHigh) - 32'i32

      let vil = (vl shr (2'i32 * i)) and 0x03030303'i32
      let vih = ((vh shr i) shl 2'i32) and 0x04040404'i32
      let vi = hippoVsubss4(vil, vih)

      let q8Blk = q8BlockBase + i
      let q8Base = q8Blk * BlockQ8_1Size
      let d8 = readHalf(q8Arr, q8Base)

      let q8QsBase = q8Base + 4'i32 + (iqs mod QI8_1) * 4'i32
      let u = cint(hippoLoadU32(addr q8Arr[q8QsBase]))

      blockSumf = blockSumf + d8 * cfloat(hippoSdot4(vi, u, 0'i32)) * cfloat(sc)

    sumf = sumf + d3 * blockSumf
    blkIdx = blkIdx + 2'i32

  warpReduceSum(sumf)
  if tid == 0'i32:
    outArr[row] = sumf

proc linearQ3KDualWarpKernel(dst1: ptr cfloat, dst2: ptr cfloat, x: ptr cfloat,
                             w1: ptr uint8, w2: ptr uint8,
                             inDim: cint, outDim: cint) {.hippoGlobal.} =
  ## Dual Q3K warp GEMV: blocks 0..outDim-1 compute dst1 from w1,
  ## Dual Q3K: rows 0..outDim-1 from w1, outDim..2*outDim-1 from w2.
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let rawRow = cint(blockIdx.x) * warpsPerBlock + warpId
  let tid = laneId
  let isSecond = rawRow >= outDim
  let row = if isSecond: rawRow - outDim else: rawRow
  if row >= outDim: return

  let w = if isSecond: cast[ptr UncheckedArray[uint8]](w2)
          else: cast[ptr UncheckedArray[uint8]](w1)
  let outArr = if isSecond: cast[ptr UncheckedArray[cfloat]](dst2)
               else: cast[ptr UncheckedArray[cfloat]](dst1)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let nBlocksPerRow = inDim div 256'i32
  let rowSizeBytes = nBlocksPerRow * 110'i32
  let rowBase = row * rowSizeBytes

  let sub = (tid shr 4'i32) and 1'i32
  let qsOff0 = 32'i32 + sub * 16'i32 + (tid and 15'i32)
  let qsOff1 = 64'i32 + sub * 16'i32 + (tid and 15'i32)
  let hmOff = sub * 16'i32 + (tid and 15'i32)

  var acc: cfloat = 0.0
  var acc2: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx + 1'i32 < nBlocksPerRow:
    let bs0 = rowBase + blkIdx * 110'i32
    let bs1 = rowBase + (blkIdx + 1'i32) * 110'i32
    let eb0 = blkIdx * 256'i32
    let eb1 = (blkIdx + 1'i32) * 256'i32
    let dAll0 = readHalf(w, bs0 + 108'i32)
    let dAll1 = readHalf(w, bs1 + 108'i32)
    let qb0a = w[bs0 + qsOff0]
    let qb1a = w[bs0 + qsOff1]
    let hmByte0 = cint(w[bs0 + hmOff])
    let qb0b = w[bs1 + qsOff0]
    let qb1b = w[bs1 + qsOff1]
    let hmByte1 = cint(w[bs1 + hmOff])
    let r96_0 = hippoLoadU32(addr w[bs0 + 96'i32])
    let r100_0 = hippoLoadU32(addr w[bs0 + 100'i32])
    let r104_0 = hippoLoadU32(addr w[bs0 + 104'i32])
    let r96_1 = hippoLoadU32(addr w[bs1 + 96'i32])
    let r100_1 = hippoLoadU32(addr w[bs1 + 100'i32])
    let r104_1 = hippoLoadU32(addr w[bs1 + 104'i32])

    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, sub,            qb0a, 0, hmByte0, 0, xArr, eb0 + tid)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, sub,            qb0b, 0, hmByte1, 0, xArr, eb1 + tid)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 2'i32 + sub,    qb0a, 2, hmByte0, 1, xArr, eb0 + tid + 32'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 2'i32 + sub,    qb0b, 2, hmByte1, 1, xArr, eb1 + tid + 32'i32)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 4'i32 + sub,    qb0a, 4, hmByte0, 2, xArr, eb0 + tid + 64'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 4'i32 + sub,    qb0b, 4, hmByte1, 2, xArr, eb1 + tid + 64'i32)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 6'i32 + sub,    qb0a, 6, hmByte0, 3, xArr, eb0 + tid + 96'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 6'i32 + sub,    qb0b, 6, hmByte1, 3, xArr, eb1 + tid + 96'i32)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 8'i32 + sub,    qb1a, 0, hmByte0, 4, xArr, eb0 + tid + 128'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 8'i32 + sub,    qb1b, 0, hmByte1, 4, xArr, eb1 + tid + 128'i32)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 10'i32 + sub,   qb1a, 2, hmByte0, 5, xArr, eb0 + tid + 160'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 10'i32 + sub,   qb1b, 2, hmByte1, 5, xArr, eb1 + tid + 160'i32)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 12'i32 + sub,   qb1a, 4, hmByte0, 6, xArr, eb0 + tid + 192'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 12'i32 + sub,   qb1b, 4, hmByte1, 6, xArr, eb1 + tid + 192'i32)
    q3kElemVec(acc,  r96_0, r100_0, r104_0, dAll0, 14'i32 + sub,   qb1a, 6, hmByte0, 7, xArr, eb0 + tid + 224'i32)
    q3kElemVec(acc2, r96_1, r100_1, r104_1, dAll1, 14'i32 + sub,   qb1b, 6, hmByte1, 7, xArr, eb1 + tid + 224'i32)
    blkIdx = blkIdx + 2'i32

  while blkIdx < nBlocksPerRow:
    let bs = rowBase + blkIdx * 110'i32
    let eb = blkIdx * 256'i32
    let dAll = readHalf(w, bs + 108'i32)
    let qb0 = w[bs + qsOff0]
    let qb1 = w[bs + qsOff1]
    let hmByte = cint(w[bs + hmOff])
    let r96 = hippoLoadU32(addr w[bs + 96'i32])
    let r100 = hippoLoadU32(addr w[bs + 100'i32])
    let r104 = hippoLoadU32(addr w[bs + 104'i32])

    q3kElemVec(acc, r96, r100, r104, dAll, sub,            qb0, 0, hmByte, 0, xArr, eb + tid)
    q3kElemVec(acc, r96, r100, r104, dAll, 2'i32 + sub,    qb0, 2, hmByte, 1, xArr, eb + tid + 32'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 4'i32 + sub,    qb0, 4, hmByte, 2, xArr, eb + tid + 64'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 6'i32 + sub,    qb0, 6, hmByte, 3, xArr, eb + tid + 96'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 8'i32 + sub,    qb1, 0, hmByte, 4, xArr, eb + tid + 128'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 10'i32 + sub,   qb1, 2, hmByte, 5, xArr, eb + tid + 160'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 12'i32 + sub,   qb1, 4, hmByte, 6, xArr, eb + tid + 192'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 14'i32 + sub,   qb1, 6, hmByte, 7, xArr, eb + tid + 224'i32)
    blkIdx = blkIdx + 1'i32

  acc = acc + acc2
  warpReduceSum(acc)
  if tid == 0'i32:
    outArr[row] = acc

proc linearQ3K2RowWarpKernel(dst: ptr cfloat, x: ptr cfloat, w: ptr uint8,
                             inDim: cint, outDim: cint) {.hippoGlobal.} =
  ## 2-row Q3K GEMV: each warp handles 2 output rows, sharing activation loads.
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let row0 = (cint(blockIdx.x) * warpsPerBlock + warpId) * 2'i32
  let row1 = row0 + 1'i32
  if row0 >= outDim: return

  let wArr = cast[ptr UncheckedArray[uint8]](w)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  let nBlocksPerRow = inDim div 256'i32
  let rowSizeBytes = nBlocksPerRow * 110'i32
  let rowBase0 = row0 * rowSizeBytes
  let rowBase1 = row1 * rowSizeBytes
  let hasRow1 = row1 < outDim
  let tid = laneId
  let sub = (tid shr 4'i32) and 1'i32
  let qsOff0 = 32'i32 + sub * 16'i32 + (tid and 15'i32)
  let qsOff1 = 64'i32 + sub * 16'i32 + (tid and 15'i32)
  let hmOff = sub * 16'i32 + (tid and 15'i32)

  var acc0: cfloat = 0.0
  var acc1: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx < nBlocksPerRow:
    let eb = blkIdx * 256'i32
    let xv0 = xArr[eb + tid]
    let xv1 = xArr[eb + tid + 32'i32]
    let xv2 = xArr[eb + tid + 64'i32]
    let xv3 = xArr[eb + tid + 96'i32]
    let xv4 = xArr[eb + tid + 128'i32]
    let xv5 = xArr[eb + tid + 160'i32]
    let xv6 = xArr[eb + tid + 192'i32]
    let xv7 = xArr[eb + tid + 224'i32]
    let bs0 = rowBase0 + blkIdx * 110'i32
    q3kAccumBlockVec(acc0, wArr, bs0, sub, qsOff0, qsOff1, hmOff,
                      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7)
    if hasRow1:
      let bs1 = rowBase1 + blkIdx * 110'i32
      q3kAccumBlockVec(acc1, wArr, bs1, sub, qsOff0, qsOff1, hmOff,
                        xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7)
    blkIdx = blkIdx + 1'i32

  warpReduceSum(acc0)
  if tid == 0'i32:
    outArr[row0] = acc0
  if hasRow1:
    warpReduceSum(acc1)
    if tid == 0'i32:
      outArr[row1] = acc1

proc linearQ3KDual2RowWarpKernel(dst1: ptr cfloat, dst2: ptr cfloat, x: ptr cfloat,
                                  w1: ptr uint8, w2: ptr uint8,
                                  inDim: cint, outDim: cint) {.hippoGlobal.} =
  ## Dual Q3K 2-row: rows 0..2*outDim-1 mapped to w1/w2, each warp handles 2 rows.
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let rawRow = (cint(blockIdx.x) * warpsPerBlock + warpId) * 2'i32
  let tid = laneId
  let totalRows = outDim * 2'i32
  if rawRow >= totalRows: return

  let isSecond0 = rawRow >= outDim
  let row0 = if isSecond0: rawRow - outDim else: rawRow
  let w0 = if isSecond0: cast[ptr UncheckedArray[uint8]](w2)
           else: cast[ptr UncheckedArray[uint8]](w1)
  let out0 = if isSecond0: cast[ptr UncheckedArray[cfloat]](dst2)
             else: cast[ptr UncheckedArray[cfloat]](dst1)

  let rawRow1 = rawRow + 1'i32
  let hasRow1 = rawRow1 < totalRows
  let isSecond1 = rawRow1 >= outDim
  let row1 = if isSecond1: rawRow1 - outDim else: rawRow1
  let w1Arr = if isSecond1: cast[ptr UncheckedArray[uint8]](w2)
              else: cast[ptr UncheckedArray[uint8]](w1)
  let out1 = if isSecond1: cast[ptr UncheckedArray[cfloat]](dst2)
             else: cast[ptr UncheckedArray[cfloat]](dst1)

  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let nBlocksPerRow = inDim div 256'i32
  let rowSizeBytes = nBlocksPerRow * 110'i32
  let rowBase0 = row0 * rowSizeBytes
  let rowBase1 = row1 * rowSizeBytes
  let sub = (tid shr 4'i32) and 1'i32
  let qsOff0 = 32'i32 + sub * 16'i32 + (tid and 15'i32)
  let qsOff1 = 64'i32 + sub * 16'i32 + (tid and 15'i32)
  let hmOff = sub * 16'i32 + (tid and 15'i32)

  var acc0: cfloat = 0.0
  var acc1: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx < nBlocksPerRow:
    let eb = blkIdx * 256'i32
    let xv0 = xArr[eb + tid]
    let xv1 = xArr[eb + tid + 32'i32]
    let xv2 = xArr[eb + tid + 64'i32]
    let xv3 = xArr[eb + tid + 96'i32]
    let xv4 = xArr[eb + tid + 128'i32]
    let xv5 = xArr[eb + tid + 160'i32]
    let xv6 = xArr[eb + tid + 192'i32]
    let xv7 = xArr[eb + tid + 224'i32]
    let bs0 = rowBase0 + blkIdx * 110'i32
    q3kAccumBlockVec(acc0, w0, bs0, sub, qsOff0, qsOff1, hmOff,
                      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7)
    if hasRow1:
      let bs1 = rowBase1 + blkIdx * 110'i32
      q3kAccumBlockVec(acc1, w1Arr, bs1, sub, qsOff0, qsOff1, hmOff,
                        xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7)
    blkIdx = blkIdx + 1'i32

  warpReduceSum(acc0)
  if tid == 0'i32:
    out0[row0] = acc0
  if hasRow1:
    warpReduceSum(acc1)
    if tid == 0'i32:
      out1[row1] = acc1

proc linearQ3KDualSiluWarpKernel(dst: ptr cfloat, x: ptr cfloat,
                                  wGate: ptr uint8, wUp: ptr uint8,
                                  inDim: cint, outDim: cint) {.hippoGlobal.} =
  ## Fused dual Q3K GEMV + SiLU×mul: computes dst[row] = silu(gate·x) * (up·x).
  ## Each warp processes one output row across both gate and up weight matrices.
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let row = cint(blockIdx.x) * warpsPerBlock + warpId
  let tid = laneId
  if row >= outDim: return

  let gArr = cast[ptr UncheckedArray[uint8]](wGate)
  let uArr = cast[ptr UncheckedArray[uint8]](wUp)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  let nBlocksPerRow = inDim div 256'i32
  let rowSizeBytes = nBlocksPerRow * 110'i32
  let rowBase = row * rowSizeBytes
  let sub = (tid shr 4'i32) and 1'i32
  let qsOff0 = 32'i32 + sub * 16'i32 + (tid and 15'i32)
  let qsOff1 = 64'i32 + sub * 16'i32 + (tid and 15'i32)
  let hmOff = sub * 16'i32 + (tid and 15'i32)

  var accG: cfloat = 0.0
  var accG2: cfloat = 0.0
  var accU: cfloat = 0.0
  var accU2: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx + 1'i32 < nBlocksPerRow:
    let bs0g = rowBase + blkIdx * 110'i32
    let bs1g = rowBase + (blkIdx + 1'i32) * 110'i32
    let bs0u = rowBase + blkIdx * 110'i32
    let bs1u = rowBase + (blkIdx + 1'i32) * 110'i32
    let eb0 = blkIdx * 256'i32
    let eb1 = (blkIdx + 1'i32) * 256'i32

    let dAll0g = readHalf(gArr, bs0g + 108'i32)
    let dAll1g = readHalf(gArr, bs1g + 108'i32)
    let qb0ag = gArr[bs0g + qsOff0]; let qb1ag = gArr[bs0g + qsOff1]
    let hmByte0g = cint(gArr[bs0g + hmOff])
    let qb0bg = gArr[bs1g + qsOff0]; let qb1bg = gArr[bs1g + qsOff1]
    let hmByte1g = cint(gArr[bs1g + hmOff])
    let r96_0g = hippoLoadU32(addr gArr[bs0g + 96'i32])
    let r100_0g = hippoLoadU32(addr gArr[bs0g + 100'i32])
    let r104_0g = hippoLoadU32(addr gArr[bs0g + 104'i32])
    let r96_1g = hippoLoadU32(addr gArr[bs1g + 96'i32])
    let r100_1g = hippoLoadU32(addr gArr[bs1g + 100'i32])
    let r104_1g = hippoLoadU32(addr gArr[bs1g + 104'i32])

    let dAll0u = readHalf(uArr, bs0u + 108'i32)
    let dAll1u = readHalf(uArr, bs1u + 108'i32)
    let qb0au = uArr[bs0u + qsOff0]; let qb1au = uArr[bs0u + qsOff1]
    let hmByte0u = cint(uArr[bs0u + hmOff])
    let qb0bu = uArr[bs1u + qsOff0]; let qb1bu = uArr[bs1u + qsOff1]
    let hmByte1u = cint(uArr[bs1u + hmOff])
    let r96_0u = hippoLoadU32(addr uArr[bs0u + 96'i32])
    let r100_0u = hippoLoadU32(addr uArr[bs0u + 100'i32])
    let r104_0u = hippoLoadU32(addr uArr[bs0u + 104'i32])
    let r96_1u = hippoLoadU32(addr uArr[bs1u + 96'i32])
    let r100_1u = hippoLoadU32(addr uArr[bs1u + 100'i32])
    let r104_1u = hippoLoadU32(addr uArr[bs1u + 104'i32])

    q3kElemVec(accG,  r96_0g, r100_0g, r104_0g, dAll0g, sub,          qb0ag, 0, hmByte0g, 0, xArr, eb0 + tid)
    q3kElemVec(accU,  r96_0u, r100_0u, r104_0u, dAll0u, sub,          qb0au, 0, hmByte0u, 0, xArr, eb0 + tid)
    q3kElemVec(accG2, r96_1g, r100_1g, r104_1g, dAll1g, sub,          qb0bg, 0, hmByte1g, 0, xArr, eb1 + tid)
    q3kElemVec(accU2, r96_1u, r100_1u, r104_1u, dAll1u, sub,          qb0bu, 0, hmByte1u, 0, xArr, eb1 + tid)
    q3kElemVec(accG,  r96_0g, r100_0g, r104_0g, dAll0g, 2'i32 + sub,  qb0ag, 2, hmByte0g, 1, xArr, eb0 + tid + 32'i32)
    q3kElemVec(accU,  r96_0u, r100_0u, r104_0u, dAll0u, 2'i32 + sub,  qb0au, 2, hmByte0u, 1, xArr, eb0 + tid + 32'i32)
    q3kElemVec(accG2, r96_1g, r100_1g, r104_1g, dAll1g, 2'i32 + sub,  qb0bg, 2, hmByte1g, 1, xArr, eb1 + tid + 32'i32)
    q3kElemVec(accU2, r96_1u, r100_1u, r104_1u, dAll1u, 2'i32 + sub,  qb0bu, 2, hmByte1u, 1, xArr, eb1 + tid + 32'i32)
    q3kElemVec(accG,  r96_0g, r100_0g, r104_0g, dAll0g, 4'i32 + sub,  qb0ag, 4, hmByte0g, 2, xArr, eb0 + tid + 64'i32)
    q3kElemVec(accU,  r96_0u, r100_0u, r104_0u, dAll0u, 4'i32 + sub,  qb0au, 4, hmByte0u, 2, xArr, eb0 + tid + 64'i32)
    q3kElemVec(accG2, r96_1g, r100_1g, r104_1g, dAll1g, 4'i32 + sub,  qb0bg, 4, hmByte1g, 2, xArr, eb1 + tid + 64'i32)
    q3kElemVec(accU2, r96_1u, r100_1u, r104_1u, dAll1u, 4'i32 + sub,  qb0bu, 4, hmByte1u, 2, xArr, eb1 + tid + 64'i32)
    q3kElemVec(accG,  r96_0g, r100_0g, r104_0g, dAll0g, 6'i32 + sub,  qb0ag, 6, hmByte0g, 3, xArr, eb0 + tid + 96'i32)
    q3kElemVec(accU,  r96_0u, r100_0u, r104_0u, dAll0u, 6'i32 + sub,  qb0au, 6, hmByte0u, 3, xArr, eb0 + tid + 96'i32)
    q3kElemVec(accG2, r96_1g, r100_1g, r104_1g, dAll1g, 6'i32 + sub,  qb0bg, 6, hmByte1g, 3, xArr, eb1 + tid + 96'i32)
    q3kElemVec(accU2, r96_1u, r100_1u, r104_1u, dAll1u, 6'i32 + sub,  qb0bu, 6, hmByte1u, 3, xArr, eb1 + tid + 96'i32)
    q3kElemVec(accG,  r96_0g, r100_0g, r104_0g, dAll0g, 8'i32 + sub,  qb1ag, 0, hmByte0g, 4, xArr, eb0 + tid + 128'i32)
    q3kElemVec(accU,  r96_0u, r100_0u, r104_0u, dAll0u, 8'i32 + sub,  qb1au, 0, hmByte0u, 4, xArr, eb0 + tid + 128'i32)
    q3kElemVec(accG2, r96_1g, r100_1g, r104_1g, dAll1g, 8'i32 + sub,  qb1bg, 0, hmByte1g, 4, xArr, eb1 + tid + 128'i32)
    q3kElemVec(accU2, r96_1u, r100_1u, r104_1u, dAll1u, 8'i32 + sub,  qb1bu, 0, hmByte1u, 4, xArr, eb1 + tid + 128'i32)
    q3kElemVec(accG,  r96_0g, r100_0g, r104_0g, dAll0g, 10'i32 + sub, qb1ag, 2, hmByte0g, 5, xArr, eb0 + tid + 160'i32)
    q3kElemVec(accU,  r96_0u, r100_0u, r104_0u, dAll0u, 10'i32 + sub, qb1au, 2, hmByte0u, 5, xArr, eb0 + tid + 160'i32)
    q3kElemVec(accG2, r96_1g, r100_1g, r104_1g, dAll1g, 10'i32 + sub, qb1bg, 2, hmByte1g, 5, xArr, eb1 + tid + 160'i32)
    q3kElemVec(accU2, r96_1u, r100_1u, r104_1u, dAll1u, 10'i32 + sub, qb1bu, 2, hmByte1u, 5, xArr, eb1 + tid + 160'i32)
    q3kElemVec(accG,  r96_0g, r100_0g, r104_0g, dAll0g, 12'i32 + sub, qb1ag, 4, hmByte0g, 6, xArr, eb0 + tid + 192'i32)
    q3kElemVec(accU,  r96_0u, r100_0u, r104_0u, dAll0u, 12'i32 + sub, qb1au, 4, hmByte0u, 6, xArr, eb0 + tid + 192'i32)
    q3kElemVec(accG2, r96_1g, r100_1g, r104_1g, dAll1g, 12'i32 + sub, qb1bg, 4, hmByte1g, 6, xArr, eb1 + tid + 192'i32)
    q3kElemVec(accU2, r96_1u, r100_1u, r104_1u, dAll1u, 12'i32 + sub, qb1bu, 4, hmByte1u, 6, xArr, eb1 + tid + 192'i32)
    q3kElemVec(accG,  r96_0g, r100_0g, r104_0g, dAll0g, 14'i32 + sub, qb1ag, 6, hmByte0g, 7, xArr, eb0 + tid + 224'i32)
    q3kElemVec(accU,  r96_0u, r100_0u, r104_0u, dAll0u, 14'i32 + sub, qb1au, 6, hmByte0u, 7, xArr, eb0 + tid + 224'i32)
    q3kElemVec(accG2, r96_1g, r100_1g, r104_1g, dAll1g, 14'i32 + sub, qb1bg, 6, hmByte1g, 7, xArr, eb1 + tid + 224'i32)
    q3kElemVec(accU2, r96_1u, r100_1u, r104_1u, dAll1u, 14'i32 + sub, qb1bu, 6, hmByte1u, 7, xArr, eb1 + tid + 224'i32)
    blkIdx = blkIdx + 2'i32

  while blkIdx < nBlocksPerRow:
    let bsg = rowBase + blkIdx * 110'i32
    let bsu = rowBase + blkIdx * 110'i32
    let eb = blkIdx * 256'i32
    let dAllg = readHalf(gArr, bsg + 108'i32)
    let qb0g = gArr[bsg + qsOff0]; let qb1g = gArr[bsg + qsOff1]
    let hmByteg = cint(gArr[bsg + hmOff])
    let r96g = hippoLoadU32(addr gArr[bsg + 96'i32])
    let r100g = hippoLoadU32(addr gArr[bsg + 100'i32])
    let r104g = hippoLoadU32(addr gArr[bsg + 104'i32])
    let dAllu = readHalf(uArr, bsu + 108'i32)
    let qb0u = uArr[bsu + qsOff0]; let qb1u = uArr[bsu + qsOff1]
    let hmByteu = cint(uArr[bsu + hmOff])
    let r96u = hippoLoadU32(addr uArr[bsu + 96'i32])
    let r100u = hippoLoadU32(addr uArr[bsu + 100'i32])
    let r104u = hippoLoadU32(addr uArr[bsu + 104'i32])

    q3kElemVec(accG, r96g, r100g, r104g, dAllg, sub,          qb0g, 0, hmByteg, 0, xArr, eb + tid)
    q3kElemVec(accU, r96u, r100u, r104u, dAllu, sub,          qb0u, 0, hmByteu, 0, xArr, eb + tid)
    q3kElemVec(accG, r96g, r100g, r104g, dAllg, 2'i32 + sub,  qb0g, 2, hmByteg, 1, xArr, eb + tid + 32'i32)
    q3kElemVec(accU, r96u, r100u, r104u, dAllu, 2'i32 + sub,  qb0u, 2, hmByteu, 1, xArr, eb + tid + 32'i32)
    q3kElemVec(accG, r96g, r100g, r104g, dAllg, 4'i32 + sub,  qb0g, 4, hmByteg, 2, xArr, eb + tid + 64'i32)
    q3kElemVec(accU, r96u, r100u, r104u, dAllu, 4'i32 + sub,  qb0u, 4, hmByteu, 2, xArr, eb + tid + 64'i32)
    q3kElemVec(accG, r96g, r100g, r104g, dAllg, 6'i32 + sub,  qb0g, 6, hmByteg, 3, xArr, eb + tid + 96'i32)
    q3kElemVec(accU, r96u, r100u, r104u, dAllu, 6'i32 + sub,  qb0u, 6, hmByteu, 3, xArr, eb + tid + 96'i32)
    q3kElemVec(accG, r96g, r100g, r104g, dAllg, 8'i32 + sub,  qb1g, 0, hmByteg, 4, xArr, eb + tid + 128'i32)
    q3kElemVec(accU, r96u, r100u, r104u, dAllu, 8'i32 + sub,  qb1u, 0, hmByteu, 4, xArr, eb + tid + 128'i32)
    q3kElemVec(accG, r96g, r100g, r104g, dAllg, 10'i32 + sub, qb1g, 2, hmByteg, 5, xArr, eb + tid + 160'i32)
    q3kElemVec(accU, r96u, r100u, r104u, dAllu, 10'i32 + sub, qb1u, 2, hmByteu, 5, xArr, eb + tid + 160'i32)
    q3kElemVec(accG, r96g, r100g, r104g, dAllg, 12'i32 + sub, qb1g, 4, hmByteg, 6, xArr, eb + tid + 192'i32)
    q3kElemVec(accU, r96u, r100u, r104u, dAllu, 12'i32 + sub, qb1u, 4, hmByteu, 6, xArr, eb + tid + 192'i32)
    q3kElemVec(accG, r96g, r100g, r104g, dAllg, 14'i32 + sub, qb1g, 6, hmByteg, 7, xArr, eb + tid + 224'i32)
    q3kElemVec(accU, r96u, r100u, r104u, dAllu, 14'i32 + sub, qb1u, 6, hmByteu, 7, xArr, eb + tid + 224'i32)
    blkIdx = blkIdx + 1'i32

  accG = accG + accG2
  accU = accU + accU2
  warpReduceSum(accG)
  warpReduceSum(accU)
  if tid == 0'i32:
    let g = accG
    outArr[row] = (g / (1.0f + expf(-g))) * accU

proc linearQ3KDualSiluSimpleWarpKernel(dst: ptr cfloat, x: ptr cfloat,
                                        wGate: ptr uint8, wUp: ptr uint8,
                                        inDim: cint, outDim: cint) {.hippoGlobal.} =
  ## Fused dual Q3K GEMV + SiLU×mul without ILP unroll: 2 accumulators only.
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let row = cint(blockIdx.x) * warpsPerBlock + warpId
  let tid = laneId
  if row >= outDim: return

  let gArr = cast[ptr UncheckedArray[uint8]](wGate)
  let uArr = cast[ptr UncheckedArray[uint8]](wUp)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  let nBlocksPerRow = inDim div 256'i32
  let rowSizeBytes = nBlocksPerRow * 110'i32
  let rowBase = row * rowSizeBytes
  let sub = (tid shr 4'i32) and 1'i32
  let qsOff0 = 32'i32 + sub * 16'i32 + (tid and 15'i32)
  let qsOff1 = 64'i32 + sub * 16'i32 + (tid and 15'i32)
  let hmOff = sub * 16'i32 + (tid and 15'i32)

  var accG: cfloat = 0.0
  var accU: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx < nBlocksPerRow:
    let bs = rowBase + blkIdx * 110'i32
    let eb = blkIdx * 256'i32
    let xv0 = xArr[eb + tid]
    let xv1 = xArr[eb + tid + 32'i32]
    let xv2 = xArr[eb + tid + 64'i32]
    let xv3 = xArr[eb + tid + 96'i32]
    let xv4 = xArr[eb + tid + 128'i32]
    let xv5 = xArr[eb + tid + 160'i32]
    let xv6 = xArr[eb + tid + 192'i32]
    let xv7 = xArr[eb + tid + 224'i32]
    q3kAccumBlockVec(accG, gArr, bs, sub, qsOff0, qsOff1, hmOff,
                      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7)
    q3kAccumBlockVec(accU, uArr, bs, sub, qsOff0, qsOff1, hmOff,
                      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7)
    blkIdx = blkIdx + 1'i32

  warpReduceSum(accG)
  warpReduceSum(accU)
  if tid == 0'i32:
    let g = accG
    outArr[row] = (g / (1.0f + expf(-g))) * accU

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

proc ropeAndKVStoreKernel(q: ptr cfloat, k: ptr cfloat,
                          theta: ptr cfloat,
                          kCache: ptr cfloat, vSrc: ptr cfloat,
                          vCache: ptr cfloat,
                          nHeadQ: cint, nHeadK: cint, headDim: cint,
                          ropeDim: cint, kvDim: cint,
                          pos: cint, cacheCols: cint) {.hippoGlobal.} =
  let tid = cint(threadIdx.x)
  let thetaArr = cast[ptr UncheckedArray[cfloat]](theta)
  let halfRope = ropeDim div 2
  let totalPairs = (nHeadQ + nHeadK) * halfRope
  var i = tid
  while i < totalPairs:
    let isK = i >= nHeadQ * halfRope
    let head = if isK: (i - nHeadQ * halfRope) div halfRope
               else: i div halfRope
    let pairIdx = if isK: (i - nHeadQ * halfRope) mod halfRope
                  else: i mod halfRope
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
    i = i + cint(blockDim.x)
  hippoSyncthreads()
  i = tid
  while i < kvDim:
    let kc = cast[ptr UncheckedArray[cfloat]](kCache)
    let vc = cast[ptr UncheckedArray[cfloat]](vCache)
    let ks = cast[ptr UncheckedArray[cfloat]](k)
    let vs = cast[ptr UncheckedArray[cfloat]](vSrc)
    kc[i * cacheCols + pos] = ks[i]
    vc[i * cacheCols + pos] = vs[i]
    i = i + cint(blockDim.x)

proc attentionDecodeKernel(dst: ptr cfloat, q: ptr cfloat,
                           kCache: ptr cfloat, vCache: ptr cfloat,
                           nHead: cint, nHeadKv: cint, headDim: cint,
                           curLen: cint, cacheCols: cint) {.hippoGlobal.} =
  ## Warp-per-head attention: 1 block per head, warp 0 (32 lanes) does work.
  ## headDim=64, lane i handles dims i and i+32.
  let head = cint(blockIdx.x)
  if head >= nHead: return
  let tid = cint(threadIdx.x)
  if tid >= cint(mkc.WarpSize): return

  let qArr = cast[ptr UncheckedArray[cfloat]](q)
  let kArr = cast[ptr UncheckedArray[cfloat]](kCache)
  let vArr = cast[ptr UncheckedArray[cfloat]](vCache)
  let dArr = cast[ptr UncheckedArray[cfloat]](dst)

  let qOff = head * headDim
  let kvHead = head div (nHead div nHeadKv)
  let kvOff = kvHead * headDim
  let scale = 1.0f / sqrtf(cfloat(headDim))

  let q0 = qArr[qOff + tid]
  let q1 = qArr[qOff + tid + 32]

  var acc0: cfloat = 0.0
  var acc1: cfloat = 0.0
  var mS: cfloat = -1e30f
  var sE: cfloat = 0.0f

  var p: cint = 0
  while p < curLen:
    var partial = q0 * kArr[(kvOff + tid) * cacheCols + p] +
                  q1 * kArr[(kvOff + tid + 32) * cacheCols + p]
    warpReduceSum(partial)
    let sc = hippoShfl(partial, 0) * scale

    if sc > mS:
      let corr = expf(mS - sc)
      acc0 = acc0 * corr + vArr[(kvOff + tid) * cacheCols + p]
      acc1 = acc1 * corr + vArr[(kvOff + tid + 32) * cacheCols + p]
      sE = sE * corr + 1.0f
      mS = sc
    else:
      let w = expf(sc - mS)
      acc0 = acc0 + w * vArr[(kvOff + tid) * cacheCols + p]
      acc1 = acc1 + w * vArr[(kvOff + tid + 32) * cacheCols + p]
      sE = sE + w
    p = p + 1

  let invSum = 1.0f / sE
  dArr[qOff + tid] = acc0 * invSum
  dArr[qOff + tid + 32] = acc1 * invSum

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
  warpReduceSum(acc)
  if tid == 0'i32:
    outArr[row] = acc

proc linearQ8_0WarpKernel(dst: ptr cfloat, x: ptr cfloat, w: ptr uint8,
                          inDim: cint, outDim: cint) {.hippoGlobal.} =
  ## Warp-per-row Q8_0 GEMV.
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let row = cint(blockIdx.x) * warpsPerBlock + warpId
  if row >= outDim: return
  let wArr = cast[ptr UncheckedArray[uint8]](w)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  let nBlocksPerRow = inDim div 32'i32
  let rowSizeBytes = nBlocksPerRow * cint(BlockQ8_0Size)
  let rowBase = row * rowSizeBytes
  var acc: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx < nBlocksPerRow:
    let bs = rowBase + blkIdx * cint(BlockQ8_0Size)
    let eb = blkIdx * 32'i32
    let d = readHalf(wArr, bs)
    let qVal = cast[int8](wArr[bs + 2'i32 + laneId])
    acc = acc + d * cfloat(qVal) * xArr[eb + laneId]
    blkIdx = blkIdx + 1'i32
  warpReduceSum(acc)
  if laneId == 0'i32:
    outArr[row] = acc

proc linearQ2KLdsKernel(dst: ptr cfloat, x: ptr cfloat, w: ptr uint8,
                        inDim: cint, outDim: cint) {.hippoGlobal.} =
  var sAct {.hippoShared.}: array[2048, cfloat]
  let tid = cint(threadIdx.x)
  let warpId = tid div cint(mkc.WarpSize)
  let laneId = tid mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  var i = tid
  while i < inDim:
    sAct[i] = xArr[i]
    i = i + cint(blockDim.x)
  hippoSyncthreads()
  let row = cint(blockIdx.x) * warpsPerBlock + warpId
  if row >= outDim: return
  let wArr = cast[ptr UncheckedArray[uint8]](w)
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
    let ddm = hippoLoadU32(addr wArr[bs + 80'i32])
    let d = hippoHalfToFloat(uint16(ddm and 0xFFFF'u32))
    let dm = hippoHalfToFloat(uint16(ddm shr 16))
    let qb0 = wArr[bs + qsOff0]
    let qb1 = wArr[bs + qsOff1]
    q2kAccumBlock(acc, wArr, bs, d, dm, qb0, qb1, sub, sAct, eb, laneId)
    blkIdx = blkIdx + 1'i32
  warpReduceSum(acc)
  if laneId == 0'i32:
    outArr[row] = acc

proc linearQ2KDualLdsKernel(dst1: ptr cfloat, dst2: ptr cfloat,
                            x: ptr cfloat, w1: ptr uint8, w2: ptr uint8,
                            inDim: cint, outDim: cint) {.hippoGlobal.} =
  var sAct {.hippoShared.}: array[2048, cfloat]
  let tid = cint(threadIdx.x)
  let warpId = tid div cint(mkc.WarpSize)
  let laneId = tid mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  var i = tid
  while i < inDim:
    sAct[i] = xArr[i]
    i = i + cint(blockDim.x)
  hippoSyncthreads()
  let row = cint(blockIdx.x) * warpsPerBlock + warpId
  if row >= outDim: return
  let outArr1 = cast[ptr UncheckedArray[cfloat]](dst1)
  let outArr2 = cast[ptr UncheckedArray[cfloat]](dst2)
  let nBlocksPerRow = inDim div cint(QK_K)
  let rowSizeBytes = nBlocksPerRow * cint(BlockQ2KSize)
  let rowBase = row * rowSizeBytes
  let sub = (laneId shr 4'i32) and 1'i32
  let qsOff0 = 16'i32 + sub * 16'i32 + (laneId and 15'i32)
  let qsOff1 = 48'i32 + sub * 16'i32 + (laneId and 15'i32)
  var acc1, acc2: cfloat = 0.0
  var blkIdx: cint = 0
  let wArr1 = cast[ptr UncheckedArray[uint8]](w1)
  let wArr2 = cast[ptr UncheckedArray[uint8]](w2)
  while blkIdx < nBlocksPerRow:
    let eb = blkIdx * cint(QK_K)
    block:
      let bs = rowBase + blkIdx * cint(BlockQ2KSize)
      let ddm = hippoLoadU32(addr wArr1[bs + 80'i32])
      let d = hippoHalfToFloat(uint16(ddm and 0xFFFF'u32))
      let dm = hippoHalfToFloat(uint16(ddm shr 16))
      let qb0 = wArr1[bs + qsOff0]
      let qb1 = wArr1[bs + qsOff1]
      q2kAccumBlock(acc1, wArr1, bs, d, dm, qb0, qb1, sub, sAct, eb, laneId)
    block:
      let bs = rowBase + blkIdx * cint(BlockQ2KSize)
      let ddm = hippoLoadU32(addr wArr2[bs + 80'i32])
      let d = hippoHalfToFloat(uint16(ddm and 0xFFFF'u32))
      let dm = hippoHalfToFloat(uint16(ddm shr 16))
      let qb0 = wArr2[bs + qsOff0]
      let qb1 = wArr2[bs + qsOff1]
      q2kAccumBlock(acc2, wArr2, bs, d, dm, qb0, qb1, sub, sAct, eb, laneId)
    blkIdx = blkIdx + 1'i32
  warpReduceSum(acc1)
  warpReduceSum(acc2)
  if laneId == 0'i32:
    outArr1[row] = acc1
    outArr2[row] = acc2

proc linearQ3KLdsKernel(dst: ptr cfloat, x: ptr cfloat, w: ptr uint8,
                        inDim: cint, outDim: cint) {.hippoGlobal.} =
  var sAct {.hippoShared.}: array[2048, cfloat]
  let tid = cint(threadIdx.x)
  let warpId = tid div cint(mkc.WarpSize)
  let laneId = tid mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  var i = tid
  while i < inDim:
    sAct[i] = xArr[i]
    i = i + cint(blockDim.x)
  hippoSyncthreads()
  let row = cint(blockIdx.x) * warpsPerBlock + warpId
  if row >= outDim: return
  let wArr = cast[ptr UncheckedArray[uint8]](w)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  let nBlocksPerRow = inDim div 256'i32
  let rowSizeBytes = nBlocksPerRow * 110'i32
  let rowBase = row * rowSizeBytes
  let sub = (laneId shr 4'i32) and 1'i32
  let qsOff0 = 32'i32 + sub * 16'i32 + (laneId and 15'i32)
  let qsOff1 = 64'i32 + sub * 16'i32 + (laneId and 15'i32)
  let hmOff = sub * 16'i32 + (laneId and 15'i32)
  var acc: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx < nBlocksPerRow:
    let bs = rowBase + blkIdx * 110'i32
    let eb = blkIdx * 256'i32
    let dAll = readHalf(wArr, bs + 108'i32)
    let qb0 = wArr[bs + qsOff0]
    let qb1 = wArr[bs + qsOff1]
    let hmByte = cint(wArr[bs + hmOff])
    let r96 = hippoLoadU32(addr wArr[bs + 96'i32])
    let r100 = hippoLoadU32(addr wArr[bs + 100'i32])
    let r104 = hippoLoadU32(addr wArr[bs + 104'i32])
    q3kElemVec(acc, r96, r100, r104, dAll, sub,            qb0, 0, hmByte, 0, sAct, eb + laneId)
    q3kElemVec(acc, r96, r100, r104, dAll, 2'i32 + sub,    qb0, 2, hmByte, 1, sAct, eb + laneId + 32'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 4'i32 + sub,    qb0, 4, hmByte, 2, sAct, eb + laneId + 64'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 6'i32 + sub,    qb0, 6, hmByte, 3, sAct, eb + laneId + 96'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 8'i32 + sub,    qb1, 0, hmByte, 4, sAct, eb + laneId + 128'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 10'i32 + sub,   qb1, 2, hmByte, 5, sAct, eb + laneId + 160'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 12'i32 + sub,   qb1, 4, hmByte, 6, sAct, eb + laneId + 192'i32)
    q3kElemVec(acc, r96, r100, r104, dAll, 14'i32 + sub,   qb1, 6, hmByte, 7, sAct, eb + laneId + 224'i32)
    blkIdx = blkIdx + 1'i32
  warpReduceSum(acc)
  if laneId == 0'i32:
    outArr[row] = acc

proc linearQ8_0LdsKernel(dst: ptr cfloat, x: ptr cfloat, w: ptr uint8,
                          inDim: cint, outDim: cint) {.hippoGlobal.} =
  var sAct {.hippoShared.}: array[2048, cfloat]
  let tid = cint(threadIdx.x)
  let warpId = tid div cint(mkc.WarpSize)
  let laneId = tid mod cint(mkc.WarpSize)
  let warpsPerBlock = cint(blockDim.x) div cint(mkc.WarpSize)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  var i = tid
  while i < inDim:
    sAct[i] = xArr[i]
    i = i + cint(blockDim.x)
  hippoSyncthreads()
  let row = cint(blockIdx.x) * warpsPerBlock + warpId
  if row >= outDim: return
  let wArr = cast[ptr UncheckedArray[uint8]](w)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  let nBlocksPerRow = inDim div 32'i32
  let rowSizeBytes = nBlocksPerRow * cint(BlockQ8_0Size)
  let rowBase = row * rowSizeBytes
  var acc: cfloat = 0.0
  var blkIdx: cint = 0
  while blkIdx < nBlocksPerRow:
    let bs = rowBase + blkIdx * cint(BlockQ8_0Size)
    let eb = blkIdx * 32'i32
    let d = readHalf(wArr, bs)
    let qVal = cast[int8](wArr[bs + 2'i32 + laneId])
    acc = acc + d * cfloat(qVal) * sAct[eb + laneId]
    blkIdx = blkIdx + 1'i32
  warpReduceSum(acc)
  if laneId == 0'i32:
    outArr[row] = acc

proc siluMulKernel(gate: ptr cfloat, up: ptr cfloat,
                   dim: cint) {.hippoGlobal.} =
  let tid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  if tid < dim:
    let g = cast[ptr UncheckedArray[cfloat]](gate)
    let u = cast[ptr UncheckedArray[cfloat]](up)
    let x = g[tid]
    g[tid] = (x / (1.0f + expf(-x))) * u[tid]

template warpReduceMaxIdx(val: var cfloat, idx: var cint) {.dirty.} =
  block:
    var ov: cfloat
    var oi: cint
    ov = hippoShflDown(val, 16); oi = hippoShflDown(idx, 16)
    if ov > val: val = ov; idx = oi
    ov = hippoShflDown(val, 8); oi = hippoShflDown(idx, 8)
    if ov > val: val = ov; idx = oi
    ov = hippoShflDown(val, 4); oi = hippoShflDown(idx, 4)
    if ov > val: val = ov; idx = oi
    ov = hippoShflDown(val, 2); oi = hippoShflDown(idx, 2)
    if ov > val: val = ov; idx = oi
    ov = hippoShflDown(val, 1); oi = hippoShflDown(idx, 1)
    if ov > val: val = ov; idx = oi

proc argmaxKernel(resultPtr: ptr cint, src: ptr cfloat, n: cint) {.hippoGlobal.} =
  var sMaxVal {.hippoShared.}: array[8, cfloat]
  var sMaxIdx {.hippoShared.}: array[8, cint]
  let tid = cint(threadIdx.x)
  let warpId = tid div 32'i32
  let laneId = tid mod 32'i32
  let arr = cast[ptr UncheckedArray[cfloat]](src)
  var bestVal: cfloat = -1e30f
  var bestIdx: cint = 0
  var i = tid
  while i < n:
    let v = arr[i]
    if v > bestVal:
      bestVal = v
      bestIdx = i
    i = i + cint(blockDim.x)
  warpReduceMaxIdx(bestVal, bestIdx)
  if laneId == 0'i32:
    sMaxVal[warpId] = bestVal
    sMaxIdx[warpId] = bestIdx
  hippoSyncthreads()
  if warpId == 0'i32 and laneId < (cint(blockDim.x) div 32'i32):
    bestVal = sMaxVal[laneId]
    bestIdx = sMaxIdx[laneId]
    var ov: cfloat
    var oi: cint
    ov = hippoShflDown(bestVal, 4); oi = hippoShflDown(bestIdx, 4)
    if ov > bestVal: bestVal = ov; bestIdx = oi
    ov = hippoShflDown(bestVal, 2); oi = hippoShflDown(bestIdx, 2)
    if ov > bestVal: bestVal = ov; bestIdx = oi
    ov = hippoShflDown(bestVal, 1); oi = hippoShflDown(bestIdx, 1)
    if ov > bestVal: bestVal = ov; bestIdx = oi
    if laneId == 0'i32:
      cast[ptr cint](resultPtr)[] = bestIdx

# ---------------------------------------------------------------------------
# Graph-compatible kernel variants (read variable args from device config)
# ---------------------------------------------------------------------------

proc embeddingKernelG(dst: ptr cfloat, weight: ptr cfloat,
                      cfg: ptr DecodeStepConfig, nEmb: cint) {.hippoGlobal.} =
  let tid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  if tid < nEmb:
    let w = cast[ptr UncheckedArray[cfloat]](weight)
    let d = cast[ptr UncheckedArray[cfloat]](dst)
    d[tid] = w[cfg.tokenId * nEmb + tid]

proc ropeQKDecodeKernelG(q: ptr cfloat, k: ptr cfloat,
                         theta: ptr cfloat,
                         cfg: ptr DecodeStepConfig,
                         nHeadQ: cint, nHeadK: cint, headDim: cint,
                         ropeDim: cint) {.hippoGlobal.} =
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
  let freq = thetaArr[pairIdx] * cfloat(cfg.pos)
  let cosVal = cosf(freq)
  let sinVal = sinf(freq)
  let v0 = basePtr[offset]
  let v1 = basePtr[offset + halfRope]
  basePtr[offset] = v0 * cosVal - v1 * sinVal
  basePtr[offset + halfRope] = v0 * sinVal + v1 * cosVal

proc storeKVPairKernelG(kCache: ptr cfloat, kSrc: ptr cfloat,
                        vCache: ptr cfloat, vSrc: ptr cfloat,
                        kvDim: cint, cacheCols: cint,
                        cfg: ptr DecodeStepConfig) {.hippoGlobal.} =
  let tid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  if tid < kvDim:
    let kc = cast[ptr UncheckedArray[cfloat]](kCache)
    let vc = cast[ptr UncheckedArray[cfloat]](vCache)
    let ks = cast[ptr UncheckedArray[cfloat]](kSrc)
    let vs = cast[ptr UncheckedArray[cfloat]](vSrc)
    kc[tid * cacheCols + cfg.pos] = ks[tid]
    vc[tid * cacheCols + cfg.pos] = vs[tid]

proc attentionDecodeKernelG(dst: ptr cfloat, q: ptr cfloat,
                            kCache: ptr cfloat, vCache: ptr cfloat,
                            nHead: cint, nHeadKv: cint, headDim: cint,
                            cfg: ptr DecodeStepConfig,
                            cacheCols: cint) {.hippoGlobal.} =
  let head = cint(blockIdx.x)
  if head >= nHead: return
  let tid = cint(threadIdx.x)
  if tid >= cint(mkc.WarpSize): return
  let qArr = cast[ptr UncheckedArray[cfloat]](q)
  let kArr = cast[ptr UncheckedArray[cfloat]](kCache)
  let vArr = cast[ptr UncheckedArray[cfloat]](vCache)
  let dArr = cast[ptr UncheckedArray[cfloat]](dst)
  let qOff = head * headDim
  let kvHead = head div (nHead div nHeadKv)
  let kvOff = kvHead * headDim
  let scale = 1.0f / sqrtf(cfloat(headDim))
  let q0 = qArr[qOff + tid]
  let q1 = qArr[qOff + tid + 32]
  var acc0: cfloat = 0.0
  var acc1: cfloat = 0.0
  var mS: cfloat = -1e30f
  var sE: cfloat = 0.0f
  var p: cint = 0
  while p < cfg.seqLen:
    var partial = q0 * kArr[(kvOff + tid) * cacheCols + p] +
                  q1 * kArr[(kvOff + tid + 32) * cacheCols + p]
    warpReduceSum(partial)
    let sc = hippoShfl(partial, 0) * scale
    if sc > mS:
      let corr = expf(mS - sc)
      acc0 = acc0 * corr + vArr[(kvOff + tid) * cacheCols + p]
      acc1 = acc1 * corr + vArr[(kvOff + tid + 32) * cacheCols + p]
      sE = sE * corr + 1.0f
      mS = sc
    else:
      let w = expf(sc - mS)
      acc0 = acc0 + w * vArr[(kvOff + tid) * cacheCols + p]
      acc1 = acc1 + w * vArr[(kvOff + tid + 32) * cacheCols + p]
      sE = sE + w
    p = p + 1
  let invSum = 1.0f / sE
  dArr[qOff + tid] = acc0 * invSum
  dArr[qOff + tid + 32] = acc1 * invSum

# ---------------------------------------------------------------------------
# Device-side phase functions for persistent cooperative kernel
# ---------------------------------------------------------------------------
# Grid is fixed at NumBlocks × BlockSize (16 × 256 = 4096 threads).
# No grid sync inside phase functions — the persistent kernel manages barriers.

proc embeddingPhase(dst: ptr cfloat, weight: ptr cfloat,
                    tokenId: cint, nEmb: cint) {.hippoDevice, used.} =
  let globalTid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  let stride = cint(gridDim.x) * cint(blockDim.x)
  let w = cast[ptr UncheckedArray[cfloat]](weight)
  let d = cast[ptr UncheckedArray[cfloat]](dst)
  var i = globalTid
  while i < nEmb:
    d[i] = w[tokenId * nEmb + i]
    i = i + stride

proc rmsnormPhase(dst: ptr cfloat, src: ptr cfloat, weight: ptr cfloat,
                  dim: cint, eps: cfloat) {.hippoDevice, used.} =
  if blockIdx.x != 0'u32: return
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

proc addPhase(dst: ptr cfloat, src: ptr cfloat, dim: cint) {.hippoDevice, used.} =
  let globalTid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  let stride = cint(gridDim.x) * cint(blockDim.x)
  let d = cast[ptr UncheckedArray[cfloat]](dst)
  let s = cast[ptr UncheckedArray[cfloat]](src)
  var i = globalTid
  while i < dim:
    d[i] = d[i] + s[i]
    i = i + stride

proc linearQ2KPhase(dst: ptr cfloat, x: ptr cfloat, w: ptr uint8,
                    inDim: cint, outDim: cint) {.hippoDevice, used.} =
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let globalWarpId = cint(blockIdx.x) * cint(WarpsPerBlock) + warpId
  let wArr = cast[ptr UncheckedArray[uint8]](w)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  let nBlocksPerRow = inDim div cint(QK_K)
  let rowSizeBytes = nBlocksPerRow * cint(BlockQ2KSize)
  let sub = (laneId shr 4'i32) and 1'i32
  let qsOff0 = 16'i32 + sub * 16'i32 + (laneId and 15'i32)
  let qsOff1 = 48'i32 + sub * 16'i32 + (laneId and 15'i32)
  var row = globalWarpId
  while row < outDim:
    let rowBase = row * rowSizeBytes
    var acc: cfloat = 0.0
    var blkIdx: cint = 0
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * cint(BlockQ2KSize)
      let eb = blkIdx * cint(QK_K)
      let ddm = hippoLoadU32(addr wArr[bs + 80'i32])
      let d = hippoHalfToFloat(uint16(ddm and 0xFFFF'u32))
      let dm = hippoHalfToFloat(uint16(ddm shr 16))
      let qb0 = wArr[bs + qsOff0]
      let qb1 = wArr[bs + qsOff1]
      q2kAccumBlock(acc, wArr, bs, d, dm, qb0, qb1, sub, xArr, eb, laneId)
      blkIdx = blkIdx + 1'i32
    warpReduceSum(acc)
    if laneId == 0'i32:
      outArr[row] = acc
    row = row + cint(TotalWarps)

proc linearQ3KPhase(dst: ptr cfloat, x: ptr cfloat, w: ptr uint8,
                    inDim: cint, outDim: cint) {.hippoDevice, used.} =
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let globalWarpId = cint(blockIdx.x) * cint(WarpsPerBlock) + warpId
  let wArr = cast[ptr UncheckedArray[uint8]](w)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  let nBlocksPerRow = inDim div 256'i32
  let rowSizeBytes = nBlocksPerRow * 110'i32
  let sub = (laneId shr 4'i32) and 1'i32
  let qsOff0 = 32'i32 + sub * 16'i32 + (laneId and 15'i32)
  let qsOff1 = 64'i32 + sub * 16'i32 + (laneId and 15'i32)
  let hmOff = sub * 16'i32 + (laneId and 15'i32)
  var row = globalWarpId
  while row < outDim:
    let rowBase = row * rowSizeBytes
    var acc: cfloat = 0.0
    var blkIdx: cint = 0
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * 110'i32
      let eb = blkIdx * 256'i32
      let dAll = readHalf(wArr, bs + 108'i32)
      let qb0 = wArr[bs + qsOff0]
      let qb1 = wArr[bs + qsOff1]
      let hmByte = cint(wArr[bs + hmOff])
      let r96 = hippoLoadU32(addr wArr[bs + 96'i32])
      let r100 = hippoLoadU32(addr wArr[bs + 100'i32])
      let r104 = hippoLoadU32(addr wArr[bs + 104'i32])
      q3kElemVec(acc, r96, r100, r104, dAll, sub,            qb0, 0, hmByte, 0, xArr, eb + laneId)
      q3kElemVec(acc, r96, r100, r104, dAll, 2'i32 + sub,    qb0, 2, hmByte, 1, xArr, eb + laneId + 32'i32)
      q3kElemVec(acc, r96, r100, r104, dAll, 4'i32 + sub,    qb0, 4, hmByte, 2, xArr, eb + laneId + 64'i32)
      q3kElemVec(acc, r96, r100, r104, dAll, 6'i32 + sub,    qb0, 6, hmByte, 3, xArr, eb + laneId + 96'i32)
      q3kElemVec(acc, r96, r100, r104, dAll, 8'i32 + sub,    qb1, 0, hmByte, 4, xArr, eb + laneId + 128'i32)
      q3kElemVec(acc, r96, r100, r104, dAll, 10'i32 + sub,   qb1, 2, hmByte, 5, xArr, eb + laneId + 160'i32)
      q3kElemVec(acc, r96, r100, r104, dAll, 12'i32 + sub,   qb1, 4, hmByte, 6, xArr, eb + laneId + 192'i32)
      q3kElemVec(acc, r96, r100, r104, dAll, 14'i32 + sub,   qb1, 6, hmByte, 7, xArr, eb + laneId + 224'i32)
      blkIdx = blkIdx + 1'i32
    warpReduceSum(acc)
    if laneId == 0'i32:
      outArr[row] = acc
    row = row + cint(TotalWarps)

proc linearF32Phase(dst: ptr cfloat, x: ptr cfloat, w: ptr cfloat,
                    inDim: cint, outDim: cint) {.hippoDevice, used.} =
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let globalWarpId = cint(blockIdx.x) * cint(WarpsPerBlock) + warpId
  let wArr = cast[ptr UncheckedArray[cfloat]](w)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  var row = globalWarpId
  while row < outDim:
    var acc: cfloat = 0.0
    var i = laneId
    while i < inDim:
      acc = acc + wArr[row * inDim + i] * xArr[i]
      i = i + cint(mkc.WarpSize)
    warpReduceSum(acc)
    if laneId == 0'i32:
      outArr[row] = acc
    row = row + cint(TotalWarps)

proc linearQ8_0Phase(dst: ptr cfloat, x: ptr cfloat, w: ptr uint8,
                     inDim: cint, outDim: cint) {.hippoDevice, used.} =
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let globalWarpId = cint(blockIdx.x) * cint(WarpsPerBlock) + warpId
  let wArr = cast[ptr UncheckedArray[uint8]](w)
  let xArr = cast[ptr UncheckedArray[cfloat]](x)
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  let nBlocksPerRow = inDim div 32'i32
  let rowSizeBytes = nBlocksPerRow * cint(BlockQ8_0Size)
  var row = globalWarpId
  while row < outDim:
    let rowBase = row * rowSizeBytes
    var acc: cfloat = 0.0
    var blkIdx: cint = 0
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * cint(BlockQ8_0Size)
      let eb = blkIdx * 32'i32
      let d = readHalf(wArr, bs)
      let qVal = cast[int8](wArr[bs + 2'i32 + laneId])
      acc = acc + d * cfloat(qVal) * xArr[eb + laneId]
      blkIdx = blkIdx + 1'i32
    warpReduceSum(acc)
    if laneId == 0'i32:
      outArr[row] = acc
    row = row + cint(TotalWarps)

proc linearPhase(dst: ptr cfloat, x: ptr cfloat, w: MkWeight,
                 inDim: cint, outDim: cint) {.hippoDevice, used.} =
  if w.qtype == 10'i32:
    linearQ2KPhase(dst, x, cast[ptr uint8](w.p), inDim, outDim)
  elif w.qtype == 11'i32:
    linearQ3KPhase(dst, x, cast[ptr uint8](w.p), inDim, outDim)
  elif w.qtype == 0'i32:
    linearF32Phase(dst, x, cast[ptr cfloat](w.p), inDim, outDim)
  elif w.qtype == 8'i32:
    linearQ8_0Phase(dst, x, cast[ptr uint8](w.p), inDim, outDim)

proc ropePhase(q: ptr cfloat, k: ptr cfloat, theta: ptr cfloat,
               nHeadQ: cint, nHeadK: cint, headDim: cint,
               ropeDim: cint, pos: cint) {.hippoDevice, used.} =
  let globalTid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  let stride = cint(gridDim.x) * cint(blockDim.x)
  let halfRope = ropeDim div 2
  let totalPairs = (nHeadQ + nHeadK) * halfRope
  let thetaArr = cast[ptr UncheckedArray[cfloat]](theta)
  var tid = globalTid
  while tid < totalPairs:
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
    tid = tid + stride

proc storeKVPhase(kCache: ptr cfloat, kSrc: ptr cfloat,
                  vCache: ptr cfloat, vSrc: ptr cfloat,
                  kvDim: cint, cacheCols: cint, pos: cint) {.hippoDevice, used.} =
  let globalTid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  let stride = cint(gridDim.x) * cint(blockDim.x)
  let kc = cast[ptr UncheckedArray[cfloat]](kCache)
  let vc = cast[ptr UncheckedArray[cfloat]](vCache)
  let ks = cast[ptr UncheckedArray[cfloat]](kSrc)
  let vs = cast[ptr UncheckedArray[cfloat]](vSrc)
  var i = globalTid
  while i < kvDim:
    kc[i * cacheCols + pos] = ks[i]
    vc[i * cacheCols + pos] = vs[i]
    i = i + stride

proc attentionPhase(dst: ptr cfloat, q: ptr cfloat,
                    kCache: ptr cfloat, vCache: ptr cfloat,
                    nHead: cint, nHeadKv: cint, headDim: cint,
                    curLen: cint, cacheCols: cint) {.hippoDevice, used.} =
  ## Warp-per-head attention: 32 lanes × 2 dims each = 64 dims (headDim).
  ## 128 total warps, 32 heads → each warp handles 1 head (warps 32-127 idle).
  let warpId = cint(threadIdx.x) div cint(mkc.WarpSize)
  let laneId = cint(threadIdx.x) mod cint(mkc.WarpSize)
  let globalWarpId = cint(blockIdx.x) * cint(WarpsPerBlock) + warpId
  let head = globalWarpId
  if head >= nHead: return

  let qArr = cast[ptr UncheckedArray[cfloat]](q)
  let kArr = cast[ptr UncheckedArray[cfloat]](kCache)
  let vArr = cast[ptr UncheckedArray[cfloat]](vCache)
  let dArr = cast[ptr UncheckedArray[cfloat]](dst)

  let qOff = head * headDim
  let kvHead = head div (nHead div nHeadKv)
  let kvOff = kvHead * headDim
  let scale = 1.0f / sqrtf(cfloat(headDim))

  # Each lane loads its 2 Q values (lane i handles dims i and i+32)
  let q0 = qArr[qOff + laneId]
  let q1 = qArr[qOff + laneId + 32]

  # Accumulator for V weighted sum (2 dims per lane)
  var acc0: cfloat = 0.0
  var acc1: cfloat = 0.0
  var mS: cfloat = -1e30f
  var sE: cfloat = 0.0f

  var p: cint = 0
  while p < curLen:
    # Q·K dot product: each lane computes partial for its 2 dims
    var partial = q0 * kArr[(kvOff + laneId) * cacheCols + p] +
                  q1 * kArr[(kvOff + laneId + 32) * cacheCols + p]
    # Warp reduce to get full dot product
    warpReduceSum(partial)
    # Broadcast score from lane 0
    let sc = hippoShfl(partial, 0) * scale

    # Online softmax + V accumulation
    if sc > mS:
      let corr = expf(mS - sc)
      acc0 = acc0 * corr + vArr[(kvOff + laneId) * cacheCols + p]
      acc1 = acc1 * corr + vArr[(kvOff + laneId + 32) * cacheCols + p]
      sE = sE * corr + 1.0f
      mS = sc
    else:
      let w = expf(sc - mS)
      acc0 = acc0 + w * vArr[(kvOff + laneId) * cacheCols + p]
      acc1 = acc1 + w * vArr[(kvOff + laneId + 32) * cacheCols + p]
      sE = sE + w
    p = p + 1

  let invSum = 1.0f / sE
  dArr[qOff + laneId] = acc0 * invSum
  dArr[qOff + laneId + 32] = acc1 * invSum

proc siluMulPhase(gate: ptr cfloat, up: ptr cfloat, dim: cint) {.hippoDevice, used.} =
  let globalTid = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  let stride = cint(gridDim.x) * cint(blockDim.x)
  let g = cast[ptr UncheckedArray[cfloat]](gate)
  let u = cast[ptr UncheckedArray[cfloat]](up)
  var i = globalTid
  while i < dim:
    let x = g[i]
    g[i] = (x / (1.0f + expf(-x))) * u[i]
    i = i + stride

# ---------------------------------------------------------------------------
# Persistent cooperative kernel (disabled — grid.sync() wrong primitive on gfx1151)
# ---------------------------------------------------------------------------

when not defined(useIndividualLaunches):
  proc megakernelDecode(
    weights: ptr MkModelWeights, bufs: ptr MkBuffers,
    tokenId: cint, curLen: cint, cacheCols: cint
  ) {.hippoGlobal.} =
    let act0 = cast[ptr cfloat](bufs.act0)
    let act1 = cast[ptr cfloat](bufs.act1)
    let s0 = cast[ptr cfloat](bufs.scratch0)
    let s1 = cast[ptr cfloat](bufs.scratch1)
    let s2 = cast[ptr cfloat](bufs.scratch2)
    let logitsP = cast[ptr cfloat](bufs.logits)
    let nEmb = cint(ModelCfg.nEmb)
    let eps = cfloat(ModelCfg.rmsEps)

    embeddingPhase(act0, cast[ptr cfloat](weights.tokEmb), tokenId, nEmb)
    gridSync()

    rmsnormPhase(act1, act0, cast[ptr cfloat](weights.layers[0].attnNorm), nEmb, eps)
    gridSync()

    var layer: cint = 0
    while layer < cint(ModelCfg.nLayers):
      let kvK = cast[ptr cfloat](bufs.kvK[layer])
      let kvV = cast[ptr cfloat](bufs.kvV[layer])

      linearPhase(s0, act1, weights.layers[layer].wq, nEmb, cint(QDim))
      linearPhase(s1, act1, weights.layers[layer].wk, nEmb, cint(KvDim))
      linearPhase(s2, act1, weights.layers[layer].wv, nEmb, cint(KvDim))
      gridSync()

      ropePhase(s0, s1, cast[ptr cfloat](weights.ropeTheta),
                cint(ModelCfg.nHead), cint(ModelCfg.nHeadKv), cint(ModelCfg.headDim),
                cint(ModelCfg.ropeDim), curLen)
      gridSync()
      storeKVPhase(kvK, s1, kvV, s2, cint(KvDim), cacheCols, curLen)
      gridSync()

      attentionPhase(s1, s0, kvK, kvV,
                     cint(ModelCfg.nHead), cint(ModelCfg.nHeadKv), cint(ModelCfg.headDim),
                     curLen + 1, cacheCols)
      gridSync()

      linearPhase(s0, s1, weights.layers[layer].wo, cint(QDim), nEmb)
      gridSync()

      addPhase(act0, s0, nEmb)
      gridSync()

      rmsnormPhase(act1, act0, cast[ptr cfloat](weights.layers[layer].ffnNorm), nEmb, eps)
      gridSync()

      linearPhase(s0, act1, weights.layers[layer].wGate, nEmb, cint(ModelCfg.ffnDim))
      linearPhase(s1, act1, weights.layers[layer].wUp, nEmb, cint(ModelCfg.ffnDim))
      gridSync()

      siluMulPhase(s0, s1, cint(ModelCfg.ffnDim))
      gridSync()

      linearPhase(s1, s0, weights.layers[layer].wDown, cint(ModelCfg.ffnDim), nEmb)
      gridSync()

      addPhase(act0, s1, nEmb)
      gridSync()

      if layer < cint(ModelCfg.nLayers) - 1:
        rmsnormPhase(act1, act0,
                     cast[ptr cfloat](weights.layers[layer + 1].attnNorm), nEmb, eps)
        gridSync()

      layer = layer + 1

    rmsnormPhase(act1, act0, cast[ptr cfloat](weights.outputNorm), nEmb, eps)
    gridSync()
    linearPhase(logitsP, act1, weights.outputWeight, nEmb, cint(ModelCfg.nVocab))

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

proc gpuRopeAndKVStore(q, k, theta, kCache, vSrc, vCache: pointer, pos, cacheCols: int) =
  var qP = cast[ptr cfloat](q)
  var kP = cast[ptr cfloat](k)
  var tP = cast[ptr cfloat](theta)
  var kcP = cast[ptr cfloat](kCache)
  var vsP = cast[ptr cfloat](vSrc)
  var vcP = cast[ptr cfloat](vCache)
  var nHQ = cint(ModelCfg.nHead)
  var nHK = cint(ModelCfg.nHeadKv)
  var hDim = cint(ModelCfg.headDim)
  var rDim = cint(ModelCfg.ropeDim)
  var kvd = cint(KvDim)
  var p = cint(pos)
  var cc = cint(cacheCols)
  hippoLaunchKernel(ropeAndKVStoreKernel,
    gridDim = newDim3(1), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(qP, kP, tP, kcP, vsP, vcP, nHQ, nHK, hDim, rDim, kvd, p, cc))

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

proc gpuArgmax(result, src: pointer, n: int) =
  var rP = cast[ptr cint](result)
  var sP = cast[ptr cfloat](src)
  var nC = cint(n)
  hippoLaunchKernel(argmaxKernel,
    gridDim = newDim3(1'u32), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(rP, sP, nC))

proc gpuAdd(dst, src: pointer, dim: int) =
  var dstP = cast[ptr cfloat](dst)
  var srcP = cast[ptr cfloat](src)
  var d = cint(dim)
  hippoLaunchKernel(addKernel,
    gridDim = grid1d(dim), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(dstP, srcP, d))

# Graph-mode dispatch wrappers (pass device-side config pointer)
proc gpuEmbeddingG(dst, weight: pointer) =
  var dstP = cast[ptr cfloat](dst)
  var wP = cast[ptr cfloat](weight)
  var cfgP = cast[ptr DecodeStepConfig](mkStepConfigDev)
  var nEmb = cint(ModelCfg.nEmb)
  hippoLaunchKernel(embeddingKernelG,
    gridDim = grid1d(ModelCfg.nEmb), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(dstP, wP, cfgP, nEmb))

proc gpuRopeQKDecodeG(q, k, theta: pointer) =
  var qP = cast[ptr cfloat](q)
  var kP = cast[ptr cfloat](k)
  var tP = cast[ptr cfloat](theta)
  var cfgP = cast[ptr DecodeStepConfig](mkStepConfigDev)
  var nHQ = cint(ModelCfg.nHead)
  var nHK = cint(ModelCfg.nHeadKv)
  var hDim = cint(ModelCfg.headDim)
  var rDim = cint(ModelCfg.ropeDim)
  let halfRope = ModelCfg.ropeDim div 2
  let totalPairs = (ModelCfg.nHead + ModelCfg.nHeadKv) * halfRope
  hippoLaunchKernel(ropeQKDecodeKernelG,
    gridDim = grid1d(totalPairs), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(qP, kP, tP, cfgP, nHQ, nHK, hDim, rDim))

proc gpuStoreKVPairG(kCache, kSrc, vCache, vSrc: pointer, cacheCols: int) =
  var kcP = cast[ptr cfloat](kCache)
  var ksP = cast[ptr cfloat](kSrc)
  var vcP = cast[ptr cfloat](vCache)
  var vsP = cast[ptr cfloat](vSrc)
  var kvd = cint(KvDim)
  var cc = cint(cacheCols)
  var cfgP = cast[ptr DecodeStepConfig](mkStepConfigDev)
  hippoLaunchKernel(storeKVPairKernelG,
    gridDim = grid1d(KvDim), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(kcP, ksP, vcP, vsP, kvd, cc, cfgP))

proc gpuAttentionDecodeG(dst, q, kCache, vCache: pointer, cacheCols: int) =
  var dstP = cast[ptr cfloat](dst)
  var qP = cast[ptr cfloat](q)
  var kcP = cast[ptr cfloat](kCache)
  var vcP = cast[ptr cfloat](vCache)
  var nH = cint(ModelCfg.nHead)
  var nHKv = cint(ModelCfg.nHeadKv)
  var hDim = cint(ModelCfg.headDim)
  var cfgP = cast[ptr DecodeStepConfig](mkStepConfigDev)
  var cc = cint(cacheCols)
  hippoLaunchKernel(attentionDecodeKernelG,
    gridDim = newDim3(ModelCfg.nHead.uint32), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(dstP, qP, kcP, vcP, nH, nHKv, hDim, cfgP, cc))

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

const WarpsPerLaunchBlock = 8'u32

proc gpuLinearQ2K(dst, x, w: pointer, inDim, outDim: int) =
  var dstP = cast[ptr cfloat](dst)
  var xP = cast[ptr cfloat](x)
  var wP = cast[ptr uint8](w)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  let nBlocks = (outDim.uint32 + WarpsPerLaunchBlock - 1) div WarpsPerLaunchBlock
  hippoLaunchKernel(linearQ2KWarpKernel,
    gridDim = newDim3(nBlocks), blockDim = newDim3(WarpsPerLaunchBlock * mkc.WarpSize.uint32),
    stream = mkStream,
    args = hippoArgs(dstP, xP, wP, iDim, oDim))

proc gpuLinearQ3K(dst, x, w: pointer, inDim, outDim: int) =
  var dstP = cast[ptr cfloat](dst)
  var xP = cast[ptr cfloat](x)
  var wP = cast[ptr uint8](w)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  let nBlocks = (outDim.uint32 + WarpsPerLaunchBlock - 1) div WarpsPerLaunchBlock
  hippoLaunchKernel(linearQ3KWarpKernel,
    gridDim = newDim3(nBlocks), blockDim = newDim3(WarpsPerLaunchBlock * mkc.WarpSize.uint32),
    stream = mkStream,
    args = hippoArgs(dstP, xP, wP, iDim, oDim))

proc gpuQuantizeQ8_1(dst, src: pointer, nElems: int) =
  var dstP = cast[ptr uint8](dst)
  var srcP = cast[ptr cfloat](src)
  var n = cint(nElems)
  let nWarps = (nElems + 31) div 32
  let nBlocks = (nWarps + 7) div 8 # 8 warps per block of 256 threads
  hippoLaunchKernel(quantizeQ8_1Kernel,
    gridDim = newDim3(nBlocks.uint32), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(dstP, srcP, n))

proc gpuLinearQ3KDp4a(dst, xQ8, w: pointer, inDim, outDim: int) =
  var dstP = cast[ptr cfloat](dst)
  var xQ8P = cast[ptr uint8](xQ8)
  var wP = cast[ptr uint8](w)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  let nBlocks = (outDim.uint32 + WarpsPerLaunchBlock - 1) div WarpsPerLaunchBlock
  hippoLaunchKernel(linearQ3KDp4aWarpKernel,
    gridDim = newDim3(nBlocks), blockDim = newDim3(WarpsPerLaunchBlock * mkc.WarpSize.uint32),
    stream = mkStream,
    args = hippoArgs(dstP, xQ8P, wP, iDim, oDim))

proc gpuLinearQ2KDual(dst1, dst2, x: pointer, w1, w2: MkWeight, inDim, outDim1, outDim2: int) =
  var dst1P = cast[ptr cfloat](dst1)
  var dst2P = cast[ptr cfloat](dst2)
  var xP = cast[ptr cfloat](x)
  var w1P = cast[ptr uint8](w1.p)
  var w2P = cast[ptr uint8](w2.p)
  var iDim = cint(inDim)
  var oDim1 = cint(outDim1)
  var oDim2 = cint(outDim2)
  let totalRows = (outDim1 + outDim2).uint32
  let nBlocks = (totalRows + WarpsPerLaunchBlock - 1) div WarpsPerLaunchBlock
  hippoLaunchKernel(linearQ2KDualWarpKernel,
    gridDim = newDim3(nBlocks), blockDim = newDim3(WarpsPerLaunchBlock * mkc.WarpSize.uint32),
    stream = mkStream,
    args = hippoArgs(dst1P, dst2P, xP, w1P, w2P, iDim, oDim1, oDim2))

proc gpuLinearQ3KDual(dst1, dst2, x: pointer, w1, w2: MkWeight, inDim, outDim: int) =
  var dst1P = cast[ptr cfloat](dst1)
  var dst2P = cast[ptr cfloat](dst2)
  var xP = cast[ptr cfloat](x)
  var w1P = cast[ptr uint8](w1.p)
  var w2P = cast[ptr uint8](w2.p)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  let totalRows = (outDim * 2).uint32
  let nBlocks = (totalRows + WarpsPerLaunchBlock - 1) div WarpsPerLaunchBlock
  hippoLaunchKernel(linearQ3KDualWarpKernel,
    gridDim = newDim3(nBlocks), blockDim = newDim3(WarpsPerLaunchBlock * mkc.WarpSize.uint32),
    stream = mkStream,
    args = hippoArgs(dst1P, dst2P, xP, w1P, w2P, iDim, oDim))

proc gpuLinearQ3KDualSilu(dst, x: pointer, wGate, wUp: MkWeight, inDim, outDim: int) =
  var dstP = cast[ptr cfloat](dst)
  var xP = cast[ptr cfloat](x)
  var wGateP = cast[ptr uint8](wGate.p)
  var wUpP = cast[ptr uint8](wUp.p)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  let nBlocks = (outDim.uint32 + WarpsPerLaunchBlock - 1) div WarpsPerLaunchBlock
  hippoLaunchKernel(linearQ3KDualSiluSimpleWarpKernel,
    gridDim = newDim3(nBlocks), blockDim = newDim3(WarpsPerLaunchBlock * mkc.WarpSize.uint32),
    stream = mkStream,
    args = hippoArgs(dstP, xP, wGateP, wUpP, iDim, oDim))

proc gpuLinearQ8_0(dst, x, w: pointer, inDim, outDim: int) =
  var dstP = cast[ptr cfloat](dst)
  var xP = cast[ptr cfloat](x)
  var wP = cast[ptr uint8](w)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  let nBlocks = (outDim.uint32 + WarpsPerLaunchBlock - 1) div WarpsPerLaunchBlock
  hippoLaunchKernel(linearQ8_0WarpKernel,
    gridDim = newDim3(nBlocks), blockDim = newDim3(WarpsPerLaunchBlock * mkc.WarpSize.uint32),
    stream = mkStream,
    args = hippoArgs(dstP, xP, wP, iDim, oDim))

proc gpuLinearQ2KLds(dst, x, w: pointer, inDim, outDim: int) =
  var dstP = cast[ptr cfloat](dst)
  var xP = cast[ptr cfloat](x)
  var wP = cast[ptr uint8](w)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  let warpsPerBlock = BlockSize div mkc.WarpSize
  let gridBlocks = (outDim + warpsPerBlock - 1) div warpsPerBlock
  hippoLaunchKernel(linearQ2KLdsKernel,
    gridDim = newDim3(gridBlocks.uint32), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(dstP, xP, wP, iDim, oDim))

proc gpuLinearQ2KDualLds(dst1, dst2, x, w1, w2: pointer, inDim, outDim: int) =
  var dst1P = cast[ptr cfloat](dst1)
  var dst2P = cast[ptr cfloat](dst2)
  var xP = cast[ptr cfloat](x)
  var w1P = cast[ptr uint8](w1)
  var w2P = cast[ptr uint8](w2)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  let warpsPerBlock = BlockSize div mkc.WarpSize
  let gridBlocks = (outDim + warpsPerBlock - 1) div warpsPerBlock
  hippoLaunchKernel(linearQ2KDualLdsKernel,
    gridDim = newDim3(gridBlocks.uint32), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(dst1P, dst2P, xP, w1P, w2P, iDim, oDim))

proc gpuLinearQ3KLds(dst, x, w: pointer, inDim, outDim: int) =
  var dstP = cast[ptr cfloat](dst)
  var xP = cast[ptr cfloat](x)
  var wP = cast[ptr uint8](w)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  let warpsPerBlock = BlockSize div mkc.WarpSize
  let gridBlocks = (outDim + warpsPerBlock - 1) div warpsPerBlock
  hippoLaunchKernel(linearQ3KLdsKernel,
    gridDim = newDim3(gridBlocks.uint32), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(dstP, xP, wP, iDim, oDim))

proc gpuLinearQ8_0Lds(dst, x, w: pointer, inDim, outDim: int) =
  var dstP = cast[ptr cfloat](dst)
  var xP = cast[ptr cfloat](x)
  var wP = cast[ptr uint8](w)
  var iDim = cint(inDim)
  var oDim = cint(outDim)
  let warpsPerBlock = BlockSize div mkc.WarpSize
  let gridBlocks = (outDim + warpsPerBlock - 1) div warpsPerBlock
  hippoLaunchKernel(linearQ8_0LdsKernel,
    gridDim = newDim3(gridBlocks.uint32), blockDim = block1d(),
    stream = mkStream,
    args = hippoArgs(dstP, xP, wP, iDim, oDim))

proc gpuLinear(dst, x: pointer, w: MkWeight, inDim, outDim: int) =
  if inDim <= ModelCfg.nEmb:
    case w.qtype
    of GgmlTypeQ2K.int32: gpuLinearQ2K(dst, x, w.p, inDim, outDim)
    of GgmlTypeQ3K.int32: gpuLinearQ3K(dst, x, w.p, inDim, outDim)
    of GgmlTypeQ8_0.int32: gpuLinearQ8_0Lds(dst, x, w.p, inDim, outDim)
    of GgmlTypeF32.int32: gpuLinearF32(dst, x, w.p, inDim, outDim)
    else: raise newException(ValueError, "unsupported qtype for gpuLinear: " & $w.qtype)
  else:
    case w.qtype
    of GgmlTypeF32.int32: gpuLinearF32(dst, x, w.p, inDim, outDim)
    of GgmlTypeQ2K.int32: gpuLinearQ2K(dst, x, w.p, inDim, outDim)
    of GgmlTypeQ3K.int32: gpuLinearQ3K(dst, x, w.p, inDim, outDim)
    of GgmlTypeQ8_0.int32: gpuLinearQ8_0(dst, x, w.p, inDim, outDim)
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

proc uploadAsQ8_0(m: var Model, tensorName: string, nCols, nRows: int): MkWeight =
  let t = m.getTensor(tensorName)
  let rowSize = rowSizeQ8_0(nCols)
  let totalBytes = rowSize * nRows
  var q8Buf = newSeq[byte](totalBytes)
  for r in 0 ..< nRows:
    let srcRow = cast[ptr UncheckedArray[float32]](addr t.data[r * nCols])
    let dstRow = cast[ptr UncheckedArray[byte]](addr q8Buf[r * rowSize])
    quantizeRowQ8_0(srcRow, dstRow, nCols)
  let alloc = hippoMalloc(totalBytes)
  hippoMemcpyAsync(alloc.p, addr q8Buf[0], totalBytes, HippoMemcpyHostToDevice, mkStream)
  mkWeightAllocs.add(alloc)
  MkWeight(p: alloc.p, qtype: GgmlTypeQ8_0.int32)

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

  # Output — requantize to Q8_0 for ~4x bandwidth reduction vs F32
  mkWeights.outputNorm = loadF32Ptr(m, "output_norm.weight")
  let outName = if m.infos.hasKey("output.weight"): "output.weight"
                else: "token_embd.weight"
  mkWeights.outputWeight = uploadAsQ8_0(m, outName, ModelCfg.nEmb, ModelCfg.nVocab)

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
  const Q8_1BlockSize = 36 # 4 bytes (d,s as fp16 pair) + 32 bytes (qs)
  const MaxQ8Blocks = (MaxDim + 31) div 32
  for alloc in [hippoMalloc(MaxDim * sizeof(float32)),
                hippoMalloc(MaxDim * sizeof(float32)),
                hippoMalloc(MaxDim * sizeof(float32)),
                hippoMalloc(MaxDim * sizeof(float32)),
                hippoMalloc(MaxDim * sizeof(float32)),
                hippoMalloc(MaxQ8Blocks * Q8_1BlockSize),
                hippoMalloc(ModelCfg.nVocab * sizeof(float32))]:
    mkBufAllocs.add(alloc)
  let argmaxAlloc = hippoMalloc(sizeof(cint))
  mkBufAllocs.add(argmaxAlloc)
  mkBuf.act0 = mkBufAllocs[^8].p
  mkBuf.act1 = mkBufAllocs[^7].p
  mkBuf.scratch0 = mkBufAllocs[^6].p
  mkBuf.scratch1 = mkBufAllocs[^5].p
  mkBuf.scratch2 = mkBufAllocs[^4].p
  mkBuf.actQ8 = mkBufAllocs[^3].p
  mkBuf.logits = mkBufAllocs[^2].p
  mkBuf.argmaxResult = mkBufAllocs[^1].p

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

  # Step config for graph capture
  mkStepConfigAlloc = hippoMalloc(sizeof(DecodeStepConfig))
  mkBufAllocs.add(mkStepConfigAlloc)
  mkStepConfigDev = mkStepConfigAlloc.p
  mkGraphCaptured = false

  hippoStreamSynchronize(mkStream)

  # Upload weight/buffer structs to device for persistent kernel
  let wStructAlloc = hippoMalloc(sizeof(MkModelWeights))
  hippoMemcpy(wStructAlloc.p, addr mkWeights, sizeof(MkModelWeights), HippoMemcpyHostToDevice)
  mkBufAllocs.add(wStructAlloc)
  devWeightsPtr = wStructAlloc.p

  let bStructAlloc = hippoMalloc(sizeof(MkBuffers))
  hippoMemcpy(bStructAlloc.p, addr mkBuf, sizeof(MkBuffers), HippoMemcpyHostToDevice)
  mkBufAllocs.add(bStructAlloc)
  devBufsPtr = bStructAlloc.p

  mkInitialized = true
  echo &"[megakernel] Loaded {ModelCfg.nLayers} layers, nEmb={ModelCfg.nEmb}, Q2K"
  echo &"[megakernel] Machine: {Machine.hostname}, {Machine.gpu.cuCount} CU"

proc unloadModelBackend*() =
  if mkInitialized:
    if mkGraphCaptured:
      hippoGraphExecDestroy(mkGraphExec)
      mkGraphCaptured = false
    mkWeightAllocs = @[]
    mkBufAllocs = @[]
    hippoStreamDestroy(mkStream)
    mkInitialized = false

# ---------------------------------------------------------------------------
# Forward decode (individual kernel launches — correctness path)
# ---------------------------------------------------------------------------

when not defined(useIndividualLaunches):
  proc forwardDecodeCooperative(token: int32, curLen: int): seq[float32] =
    var wp = devWeightsPtr
    var bp = devBufsPtr
    var tid = cint(token)
    var cLen = cint(curLen)
    var cc = cint(mkMaxLen)
    hippoLaunchCooperative(megakernelDecode,
      gridDim = newDim3(NumBlocks.uint32),
      blockDim = newDim3(BlockSize.uint32),
      stream = mkStream,
      args = hippoArgs(wp, bp, tid, cLen, cc))
    hippoStreamSynchronize(mkStream)
    result = newSeq[float32](ModelCfg.nVocab)
    hippoMemcpy(addr result[0], mkBuf.logits, ModelCfg.nVocab * sizeof(float32),
                HippoMemcpyDeviceToHost)

proc forwardDecodeGpu(token: int32, curLen: int) =
  ## Dispatch all GPU kernels for one decode step. Does NOT sync or copy logits.
  let cacheCols = mkMaxLen

  when defined(debugMegakernel):
    echo "[mk] forward token=", token, " curLen=", curLen
  gpuEmbedding(mkBuf.act0, mkWeights.tokEmb, token)

  when defined(profileMegakernel):
    var tWq, tWk, tWv, tRope, tKV, tAttn, tWo, tNorm, tWgate, tWup, tSilu, tWdown, tResid, tOutput: float32
    let profE0 = hippoEventCreate()
    let profE1 = hippoEventCreate()

  for layer in 0 ..< ModelCfg.nLayers:
    let lw = mkWeights.layers[layer]
    let kvK = mkBuf.kvK[layer]
    let kvV = mkBuf.kvV[layer]

    if layer == 0:
      gpuRmsNorm(mkBuf.act1, mkBuf.act0, lw.attnNorm)

    when defined(profileMegakernel):
      template prof(accum: var float32, body: untyped) =
        hippoEventRecord(profE0, mkStream)
        body
        hippoEventRecord(profE1, mkStream)
        hippoEventSynchronize(profE1)
        accum += hippoEventElapsedTime(profE0, profE1)
      prof(tWq): gpuLinear(mkBuf.scratch0, mkBuf.act1, lw.wq, ModelCfg.nEmb, QDim)
      prof(tWk): gpuLinear(mkBuf.scratch1, mkBuf.act1, lw.wk, ModelCfg.nEmb, KvDim)
      prof(tWv): gpuLinear(mkBuf.scratch2, mkBuf.act1, lw.wv, ModelCfg.nEmb, KvDim)
      prof(tRope): gpuRopeQKDecode(mkBuf.scratch0, mkBuf.scratch1, mkWeights.ropeTheta, curLen)
      prof(tKV): gpuStoreKVPair(kvK, mkBuf.scratch1, kvV, mkBuf.scratch2, curLen, cacheCols)
      prof(tAttn): gpuAttentionDecode(mkBuf.scratch1, mkBuf.scratch0, kvK, kvV, curLen + 1, cacheCols)
      prof(tWo): gpuLinear(mkBuf.scratch0, mkBuf.scratch1, lw.wo, QDim, ModelCfg.nEmb)
      prof(tNorm): gpuResidualRmsNorm(mkBuf.act1, mkBuf.act0, mkBuf.scratch0, lw.ffnNorm)
      prof(tWgate): gpuLinear(mkBuf.scratch0, mkBuf.act1, lw.wGate, ModelCfg.nEmb, ModelCfg.ffnDim)
      prof(tWup): gpuLinear(mkBuf.scratch1, mkBuf.act1, lw.wUp, ModelCfg.nEmb, ModelCfg.ffnDim)
      prof(tSilu): gpuSiluMul(mkBuf.scratch0, mkBuf.scratch1, ModelCfg.ffnDim)
      prof(tWdown): gpuLinear(mkBuf.scratch1, mkBuf.scratch0, lw.wDown, ModelCfg.ffnDim, ModelCfg.nEmb)
      prof(tResid):
        if layer < ModelCfg.nLayers - 1:
          gpuResidualRmsNorm(mkBuf.act1, mkBuf.act0, mkBuf.scratch1,
                             mkWeights.layers[layer + 1].attnNorm)
        else:
          gpuAdd(mkBuf.act0, mkBuf.scratch1, ModelCfg.nEmb)
    else:
      when defined(useDp4a):
        gpuQuantizeQ8_1(mkBuf.actQ8, mkBuf.act1, ModelCfg.nEmb)
        if lw.wq.qtype == GgmlTypeQ3K.int32:
          gpuLinearQ3KDp4a(mkBuf.scratch0, mkBuf.actQ8, lw.wq.p, ModelCfg.nEmb, QDim)
        else:
          gpuLinear(mkBuf.scratch0, mkBuf.act1, lw.wq, ModelCfg.nEmb, QDim)
        if lw.wk.qtype == GgmlTypeQ3K.int32:
          gpuLinearQ3KDp4a(mkBuf.scratch1, mkBuf.actQ8, lw.wk.p, ModelCfg.nEmb, KvDim)
        else:
          gpuLinear(mkBuf.scratch1, mkBuf.act1, lw.wk, ModelCfg.nEmb, KvDim)
        if lw.wv.qtype == GgmlTypeQ3K.int32:
          gpuLinearQ3KDp4a(mkBuf.scratch2, mkBuf.actQ8, lw.wv.p, ModelCfg.nEmb, KvDim)
        else:
          gpuLinear(mkBuf.scratch2, mkBuf.act1, lw.wv, ModelCfg.nEmb, KvDim)
      else:
        if lw.wq.qtype == GgmlTypeQ2K.int32 and lw.wk.qtype == GgmlTypeQ2K.int32:
          gpuLinearQ2KDual(mkBuf.scratch0, mkBuf.scratch1, mkBuf.act1,
                           lw.wq, lw.wk, ModelCfg.nEmb, QDim, KvDim)
        else:
          gpuLinear(mkBuf.scratch0, mkBuf.act1, lw.wq, ModelCfg.nEmb, QDim)
          gpuLinear(mkBuf.scratch1, mkBuf.act1, lw.wk, ModelCfg.nEmb, KvDim)
        gpuLinear(mkBuf.scratch2, mkBuf.act1, lw.wv, ModelCfg.nEmb, KvDim)
      gpuRopeAndKVStore(mkBuf.scratch0, mkBuf.scratch1, mkWeights.ropeTheta,
                        kvK, mkBuf.scratch2, kvV, curLen, cacheCols)
      gpuAttentionDecode(mkBuf.scratch1, mkBuf.scratch0, kvK, kvV, curLen + 1, cacheCols)
      gpuLinear(mkBuf.scratch0, mkBuf.scratch1, lw.wo, QDim, ModelCfg.nEmb)

      gpuResidualRmsNorm(mkBuf.act1, mkBuf.act0, mkBuf.scratch0, lw.ffnNorm)
      when defined(useDp4a):
        gpuQuantizeQ8_1(mkBuf.actQ8, mkBuf.act1, ModelCfg.nEmb)
        if lw.wGate.qtype == GgmlTypeQ3K.int32:
          gpuLinearQ3KDp4a(mkBuf.scratch0, mkBuf.actQ8, lw.wGate.p, ModelCfg.nEmb, ModelCfg.ffnDim)
        else:
          gpuLinear(mkBuf.scratch0, mkBuf.act1, lw.wGate, ModelCfg.nEmb, ModelCfg.ffnDim)
        if lw.wUp.qtype == GgmlTypeQ3K.int32:
          gpuLinearQ3KDp4a(mkBuf.scratch1, mkBuf.actQ8, lw.wUp.p, ModelCfg.nEmb, ModelCfg.ffnDim)
        else:
          gpuLinear(mkBuf.scratch1, mkBuf.act1, lw.wUp, ModelCfg.nEmb, ModelCfg.ffnDim)
      else:
        if lw.wGate.qtype == GgmlTypeQ3K.int32 and lw.wUp.qtype == GgmlTypeQ3K.int32:
          gpuLinearQ3KDualSilu(mkBuf.scratch0, mkBuf.act1, lw.wGate, lw.wUp, ModelCfg.nEmb, ModelCfg.ffnDim)
        else:
          gpuLinear(mkBuf.scratch0, mkBuf.act1, lw.wGate, ModelCfg.nEmb, ModelCfg.ffnDim)
          gpuLinear(mkBuf.scratch1, mkBuf.act1, lw.wUp, ModelCfg.nEmb, ModelCfg.ffnDim)
          gpuSiluMul(mkBuf.scratch0, mkBuf.scratch1, ModelCfg.ffnDim)
      when defined(useDp4a):
        gpuQuantizeQ8_1(mkBuf.actQ8, mkBuf.scratch0, ModelCfg.ffnDim)
        if lw.wDown.qtype == GgmlTypeQ3K.int32:
          gpuLinearQ3KDp4a(mkBuf.scratch1, mkBuf.actQ8, lw.wDown.p, ModelCfg.ffnDim, ModelCfg.nEmb)
        else:
          gpuLinear(mkBuf.scratch1, mkBuf.scratch0, lw.wDown, ModelCfg.ffnDim, ModelCfg.nEmb)
      else:
        gpuLinear(mkBuf.scratch1, mkBuf.scratch0, lw.wDown, ModelCfg.ffnDim, ModelCfg.nEmb)

      if layer < ModelCfg.nLayers - 1:
        gpuResidualRmsNorm(mkBuf.act1, mkBuf.act0, mkBuf.scratch1,
                           mkWeights.layers[layer + 1].attnNorm)
      else:
        gpuAdd(mkBuf.act0, mkBuf.scratch1, ModelCfg.nEmb)

  when defined(profileMegakernel):
    var profStart = hippoEventCreate()
    var profEnd = hippoEventCreate()
    hippoEventRecord(profStart, mkStream)
  gpuRmsNorm(mkBuf.act1, mkBuf.act0, mkWeights.outputNorm)
  gpuLinear(mkBuf.logits, mkBuf.act1, mkWeights.outputWeight,
            ModelCfg.nEmb, ModelCfg.nVocab)
  when defined(profileMegakernel):
    hippoEventRecord(profEnd, mkStream)
    hippoEventSynchronize(profEnd)
    tOutput = hippoEventElapsedTime(profStart, profEnd)
    hippoEventDestroy(profStart)
    hippoEventDestroy(profEnd)

  when defined(profileMegakernel):
    hippoStreamSynchronize(mkStream)
    let total = tWq + tWk + tWv + tRope + tKV + tAttn + tWo + tNorm + tWgate + tWup + tSilu + tWdown + tResid + tOutput
    echo &"[profile] wq={tWq:.2f} wk={tWk:.2f} wv={tWv:.2f} rope={tRope:.2f} kv={tKV:.2f} attn={tAttn:.2f} wo={tWo:.2f} norm={tNorm:.2f} wGate={tWgate:.2f} wUp={tWup:.2f} silu={tSilu:.2f} wDown={tWdown:.2f} resid={tResid:.2f} output={tOutput:.2f} total={total:.2f}ms"
    hippoEventDestroy(profE0)
    hippoEventDestroy(profE1)

proc forwardDecodeIndividual(token: int32, curLen: int): seq[float32] =
  forwardDecodeGpu(token, curLen)
  hippoStreamSynchronize(mkStream)
  result = newSeq[float32](ModelCfg.nVocab)
  hippoMemcpy(addr result[0], mkBuf.logits, ModelCfg.nVocab * sizeof(float32),
              HippoMemcpyDeviceToHost)

# ---------------------------------------------------------------------------
# Graph capture path: capture once, replay per token
# ---------------------------------------------------------------------------

proc forwardDecodeGraphBody() =
  let cacheCols = mkMaxLen
  gpuEmbeddingG(mkBuf.act0, mkWeights.tokEmb)

  for layer in 0 ..< ModelCfg.nLayers:
    let lw = mkWeights.layers[layer]
    let kvK = mkBuf.kvK[layer]
    let kvV = mkBuf.kvV[layer]

    if layer == 0:
      gpuRmsNorm(mkBuf.act1, mkBuf.act0, lw.attnNorm)

    if lw.wq.qtype == GgmlTypeQ2K.int32 and lw.wk.qtype == GgmlTypeQ2K.int32:
      gpuLinearQ2KDual(mkBuf.scratch0, mkBuf.scratch1, mkBuf.act1,
                       lw.wq, lw.wk, ModelCfg.nEmb, QDim, KvDim)
    else:
      gpuLinear(mkBuf.scratch0, mkBuf.act1, lw.wq, ModelCfg.nEmb, QDim)
      gpuLinear(mkBuf.scratch1, mkBuf.act1, lw.wk, ModelCfg.nEmb, KvDim)
    gpuLinear(mkBuf.scratch2, mkBuf.act1, lw.wv, ModelCfg.nEmb, KvDim)
    gpuRopeQKDecodeG(mkBuf.scratch0, mkBuf.scratch1, mkWeights.ropeTheta)
    gpuStoreKVPairG(kvK, mkBuf.scratch1, kvV, mkBuf.scratch2, cacheCols)
    gpuAttentionDecodeG(mkBuf.scratch1, mkBuf.scratch0, kvK, kvV, cacheCols)
    gpuLinear(mkBuf.scratch0, mkBuf.scratch1, lw.wo, QDim, ModelCfg.nEmb)

    gpuResidualRmsNorm(mkBuf.act1, mkBuf.act0, mkBuf.scratch0, lw.ffnNorm)
    if lw.wGate.qtype == GgmlTypeQ3K.int32 and lw.wUp.qtype == GgmlTypeQ3K.int32:
      gpuLinearQ3KDual(mkBuf.scratch0, mkBuf.scratch1, mkBuf.act1,
                       lw.wGate, lw.wUp, ModelCfg.nEmb, ModelCfg.ffnDim)
    else:
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

proc forwardDecodeGraph(token: int32, curLen: int): seq[float32] =
  mkStepConfig.tokenId = cint(token)
  mkStepConfig.pos = cint(curLen)
  mkStepConfig.seqLen = cint(curLen + 1)
  mkStepConfig.cacheCols = cint(mkMaxLen)
  hippoMemcpyAsync(mkStepConfigDev, cast[pointer](addr mkStepConfig),
                   sizeof(DecodeStepConfig),
                   HippoMemcpyHostToDevice, mkStream)

  if not mkGraphCaptured:
    hippoStreamBeginCapture(mkStream)
    forwardDecodeGraphBody()
    let graph = hippoStreamEndCapture(mkStream)
    mkGraphExec = hippoGraphInstantiate(graph)
    hippoGraphDestroy(graph)
    mkGraphCaptured = true

  hippoGraphLaunch(mkGraphExec, mkStream)
  hippoStreamSynchronize(mkStream)
  result = newSeq[float32](ModelCfg.nVocab)
  hippoMemcpy(addr result[0], mkBuf.logits, ModelCfg.nVocab * sizeof(float32),
              HippoMemcpyDeviceToHost)

# ---------------------------------------------------------------------------
# Backend interface: forward functions
# ---------------------------------------------------------------------------

proc forwardDecodeInternal(token: int32, curLen: int): seq[float32] =
  when defined(useGraphCapture):
    forwardDecodeGraph(token, curLen)
  elif defined(useIndividualLaunches):
    forwardDecodeIndividual(token, curLen)
  else:
    forwardDecodeCooperative(token, curLen)

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
  forwardDecodeGpu(token, cache.curLen)
  gpuArgmax(mkBuf.argmaxResult, mkBuf.logits, ModelCfg.nVocab)
  hippoStreamSynchronize(mkStream)
  var tokenId: cint
  hippoMemcpy(addr tokenId, mkBuf.argmaxResult, sizeof(cint), HippoMemcpyDeviceToHost)
  inc cache.curLen
  result = int32(tokenId)

proc forwardDecodeTokenTimed*(m: var Model, token: int32, cache: var KvCache,
                               gpuMs: var float32): int32 =
  if not mkBenchEventsCreated:
    mkBenchEvent0 = hippoEventCreate()
    mkBenchEvent1 = hippoEventCreate()
    mkBenchEventsCreated = true
  hippoEventRecord(mkBenchEvent0, mkStream)
  forwardDecodeGpu(token, cache.curLen)
  gpuArgmax(mkBuf.argmaxResult, mkBuf.logits, ModelCfg.nVocab)
  hippoEventRecord(mkBenchEvent1, mkStream)
  hippoStreamSynchronize(mkStream)
  gpuMs = hippoEventElapsedTime(mkBenchEvent0, mkBenchEvent1)
  var tokenId: cint
  hippoMemcpy(addr tokenId, mkBuf.argmaxResult, sizeof(cint), HippoMemcpyDeviceToHost)
  inc cache.curLen
  result = int32(tokenId)
