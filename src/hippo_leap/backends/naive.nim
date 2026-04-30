## Naive GPU backend: verbatim port of tinylama's forward_hippo.nim.
##
## All model weights, activations, and KV caches live on GPU.
## Only the initial token IDs are uploaded and final logits downloaded.

import
  std/[tables, math, strformat, strutils],
  hippo,
  ../[backend_types, tensor, model, gguf_loader, quant]

when not defined(cpp):
  {.error: "naive backend requires Nim's C++ backend. Build with `nim cpp`.".}

const
  HippoBlockSize = 256
  HippoBlockSizeX = 16
  HippoBlockSizeY = 16
  HippoDecodeRowsPerBlock = 1
  HippoDecodeDotUnroll = 4
  HippoMaxDecodeCols = 18432

when HippoDecodeDotUnroll != 4:
  {.error: "linearHippoDecodeKernel currently implements a fixed 4-way unroll.".}

template reduceSum256(sdata: var array[HippoBlockSize, float32], tid: int) =
  when HippoBlockSize == 256:
    if tid < 128:
      sdata[tid] = sdata[tid] + sdata[tid + 128]
    hippoSyncthreads()
    when HippoWarpSize == 64:
      if tid < 64:
        sdata[tid] = sdata[tid] + sdata[tid + 64]
      hippoSyncthreads()
      if tid < HippoWarpSize:
        var val = sdata[tid]
        val = val + hippoShflDown(val, 32)
        val = val + hippoShflDown(val, 16)
        val = val + hippoShflDown(val, 8)
        val = val + hippoShflDown(val, 4)
        val = val + hippoShflDown(val, 2)
        val = val + hippoShflDown(val, 1)
        if tid == 0:
          sdata[0] = val
      hippoSyncthreads()
    elif HippoWarpSize == 32:
      if tid < 64:
        sdata[tid] = sdata[tid] + sdata[tid + 64]
      hippoSyncthreads()
      if tid < 32:
        sdata[tid] = sdata[tid] + sdata[tid + 32]
      hippoSyncthreads()
      if tid < HippoWarpSize:
        var val = sdata[tid]
        val = val + hippoShflDown(val, 16)
        val = val + hippoShflDown(val, 8)
        val = val + hippoShflDown(val, 4)
        val = val + hippoShflDown(val, 2)
        val = val + hippoShflDown(val, 1)
        if tid == 0:
          sdata[0] = val
      hippoSyncthreads()
    else:
      if tid < 64:
        sdata[tid] = sdata[tid] + sdata[tid + 64]
      hippoSyncthreads()
      if tid < 32:
        sdata[tid] = sdata[tid] + sdata[tid + 32]
      hippoSyncthreads()
      if tid < 16:
        sdata[tid] = sdata[tid] + sdata[tid + 16]
      hippoSyncthreads()
      if tid < 8:
        sdata[tid] = sdata[tid] + sdata[tid + 8]
      hippoSyncthreads()
      if tid < 4:
        sdata[tid] = sdata[tid] + sdata[tid + 4]
      hippoSyncthreads()
      if tid < 2:
        sdata[tid] = sdata[tid] + sdata[tid + 2]
      hippoSyncthreads()
      if tid < 1:
        sdata[tid] = sdata[tid] + sdata[tid + 1]
      hippoSyncthreads()

template reduceMax256(sdata: var array[HippoBlockSize, float32], tid: int) =
  when HippoBlockSize == 256:
    if tid < 128:
      if sdata[tid + 128] > sdata[tid]:
        sdata[tid] = sdata[tid + 128]
    hippoSyncthreads()
    when HippoWarpSize == 64:
      if tid < 64:
        if sdata[tid + 64] > sdata[tid]:
          sdata[tid] = sdata[tid + 64]
      hippoSyncthreads()
      if tid < HippoWarpSize:
        var val = sdata[tid]
        var other = hippoShflDown(val, 32)
        if other > val: val = other
        other = hippoShflDown(val, 16)
        if other > val: val = other
        other = hippoShflDown(val, 8)
        if other > val: val = other
        other = hippoShflDown(val, 4)
        if other > val: val = other
        other = hippoShflDown(val, 2)
        if other > val: val = other
        other = hippoShflDown(val, 1)
        if other > val: val = other
        if tid == 0:
          sdata[0] = val
      hippoSyncthreads()
    elif HippoWarpSize == 32:
      if tid < 64:
        if sdata[tid + 64] > sdata[tid]:
          sdata[tid] = sdata[tid + 64]
      hippoSyncthreads()
      if tid < 32:
        if sdata[tid + 32] > sdata[tid]:
          sdata[tid] = sdata[tid + 32]
      hippoSyncthreads()
      if tid < HippoWarpSize:
        var val = sdata[tid]
        var other = hippoShflDown(val, 16)
        if other > val: val = other
        other = hippoShflDown(val, 8)
        if other > val: val = other
        other = hippoShflDown(val, 4)
        if other > val: val = other
        other = hippoShflDown(val, 2)
        if other > val: val = other
        other = hippoShflDown(val, 1)
        if other > val: val = other
        if tid == 0:
          sdata[0] = val
      hippoSyncthreads()
    else:
      if tid < 64:
        if sdata[tid + 64] > sdata[tid]:
          sdata[tid] = sdata[tid + 64]
      hippoSyncthreads()
      if tid < 32:
        if sdata[tid + 32] > sdata[tid]:
          sdata[tid] = sdata[tid + 32]
      hippoSyncthreads()
      if tid < 16:
        if sdata[tid + 16] > sdata[tid]:
          sdata[tid] = sdata[tid + 16]
      hippoSyncthreads()
      if tid < 8:
        if sdata[tid + 8] > sdata[tid]:
          sdata[tid] = sdata[tid + 8]
      hippoSyncthreads()
      if tid < 4:
        if sdata[tid + 4] > sdata[tid]:
          sdata[tid] = sdata[tid + 4]
      hippoSyncthreads()
      if tid < 2:
        if sdata[tid + 2] > sdata[tid]:
          sdata[tid] = sdata[tid + 2]
      hippoSyncthreads()
      if tid < 1:
        if sdata[tid + 1] > sdata[tid]:
          sdata[tid] = sdata[tid + 1]
      hippoSyncthreads()

# ---------------------------------------------------------------------------
# GpuTensor utilities
# ---------------------------------------------------------------------------

proc newGpuTensor*(shape: seq[int]): GpuTensor =
  ## Allocate a new GPU tensor with the given shape.
  var total = 1
  for s in shape: total *= s
  let bytes = total * sizeof(float32)
  let alloc = hippoMalloc(bytes)
  result = GpuTensor(
    devicePtr: alloc.p,
    alloc: alloc,
    shape: shape,
    sizeBytes: bytes,
  )

proc numel*(gt: GpuTensor): int =
  result = 1
  for s in gt.shape: result *= s

proc uploadToGpu*(t: Tensor, stream: HippoStream): GpuTensor =
  ## Upload a CPU tensor to a new GPU tensor.
  let bytes = t.data.len * sizeof(float32)
  let alloc = hippoMalloc(bytes)
  hippoMemcpyAsync(alloc.p, unsafeAddr t.data[0], bytes,
                    HippoMemcpyHostToDevice, stream)
  result = GpuTensor(
    devicePtr: alloc.p,
    alloc: alloc,
    shape: t.shape,
    sizeBytes: bytes,
  )

proc downloadToCpu*(gt: GpuTensor, stream: HippoStream): Tensor =
  ## Download a GPU tensor to a new CPU tensor.
  result = newTensor(gt.shape)
  hippoMemcpyAsync(addr result.data[0], gt.devicePtr, gt.sizeBytes,
                    HippoMemcpyDeviceToHost, stream)

proc gpuUploadToDevice*(dst: pointer, src: pointer, bytes: int, stream: HippoStream) =
  ## Upload host data to existing device pointer.
  hippoMemcpyAsync(dst, src, bytes, HippoMemcpyHostToDevice, stream)

proc gpuDownloadFromDevice*(dst: pointer, src: pointer, bytes: int, stream: HippoStream) =
  ## Download device data to host pointer.
  hippoMemcpyAsync(dst, src, bytes, HippoMemcpyDeviceToHost, stream)

proc gpuStreamSync*(stream: HippoStream) =
  ## Synchronize GPU stream.
  hippoStreamSynchronize(stream)

var tokenBufAlloc: HippoAllocRef
var tokenBufCapacity: int = 0

proc gpuUploadInt32Pooled*(data: pointer, count: int, stream: HippoStream): pointer =
  ## Upload int32 data to a reusable GPU buffer.
  let bytes = count * sizeof(int32)
  if bytes > tokenBufCapacity:
    tokenBufAlloc = hippoMalloc(max(bytes, 16))
    tokenBufCapacity = max(bytes, 16)
  hippoMemcpyAsync(tokenBufAlloc.p, data, bytes, HippoMemcpyHostToDevice, stream)
  tokenBufAlloc.p

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

var modelPtrs*: ModelGpuPtrs
var ropeThetaAlloc: HippoAllocRef
var gpuCtx*: GpuContext
var quantWeightCache: Table[string, GpuQuantWeight]
var weightCache: Table[string, GpuTensor]

proc ensureGpuContext*() =
  if not gpuCtx.initialized:
    gpuCtx.stream = hippoStreamCreate()
    gpuCtx.initialized = true

proc ensureActivationBuffers*(nElems: int) =
  ## Ensure ping-pong buffers are large enough.
  let bytes = nElems * sizeof(float32)
  if bytes > gpuCtx.actCapBytes:
    gpuCtx.act0 = newGpuTensor(@[nElems])
    gpuCtx.act1 = newGpuTensor(@[nElems])
    gpuCtx.actCapBytes = bytes

proc ensureScratchBuffers*(nElems: int) =
  ## Ensure scratch buffers are large enough.
  let bytes = nElems * sizeof(float32)
  if bytes > gpuCtx.scratchCapBytes:
    gpuCtx.scratch0 = newGpuTensor(@[nElems])
    gpuCtx.scratch1 = newGpuTensor(@[nElems])
    gpuCtx.scratch2 = newGpuTensor(@[nElems])
    gpuCtx.scratch3 = newGpuTensor(@[nElems])
    gpuCtx.scratchCapBytes = bytes

proc cachedWeight*(name: string, w: Tensor): GpuTensor =
  ## Upload a named weight tensor to GPU once; return cached GpuTensor.
  ensureGpuContext()
  if weightCache.hasKey(name):
    return weightCache[name]
  let gt = uploadToGpu(w, gpuCtx.stream)
  weightCache[name] = gt
  gt

proc cachedWeight*(w: Tensor): GpuTensor =
  ## Uncached fallback — uploads every time.
  ensureGpuContext()
  uploadToGpu(w, gpuCtx.stream)

proc quantRowSize*(nCols: int, elemType: int32): int =
  case elemType
  of GgmlTypeF16: rowSizeF16(nCols)
  of GgmlTypeQ5_0: rowSizeQ5_0(nCols)
  of GgmlTypeQ2K: rowSizeQ2K(nCols)
  of GgmlTypeQ3K: rowSizeQ3K(nCols)
  of GgmlTypeQ4K: rowSizeQ4K(nCols)
  of GgmlTypeQ6K: rowSizeQ6K(nCols)
  of GgmlTypeQ8_0: rowSizeQ8_0(nCols)
  of GgmlTypeQ5_1: rowSizeQ5_1(nCols)
  of GgmlTypeQ5K: rowSizeQ5K(nCols)
  of GgmlTypeIQ4NL: rowSizeIQ4NL(nCols)
  else: raise newException(ValueError, "unsupported quant type for row size: " & $elemType)

proc cachedQuantWeight*(name: string, m: var Model, tensorName: string): GpuQuantWeight =
  ## Upload raw quantized bytes to GPU once; return cached GpuQuantWeight.
  if quantWeightCache.hasKey(name):
    return quantWeightCache[name]
  ensureGpuContext()
  let info = m.infos[tensorName]
  let dataPtr = tensorDataPtr(m.gguf, info)
  let nCols = int(info.ne[0])
  let nRows = tensorElemCount(info) div nCols
  let rowSize = case info.elemType
    of GgmlTypeF16: rowSizeF16(nCols)
    of GgmlTypeQ5_0: rowSizeQ5_0(nCols)
    of GgmlTypeQ2K: rowSizeQ2K(nCols)
    of GgmlTypeQ3K: rowSizeQ3K(nCols)
    of GgmlTypeQ4K: rowSizeQ4K(nCols)
    of GgmlTypeQ6K: rowSizeQ6K(nCols)
    of GgmlTypeQ8_0: rowSizeQ8_0(nCols)
    of GgmlTypeQ5_1: rowSizeQ5_1(nCols)
    of GgmlTypeQ5K: rowSizeQ5K(nCols)
    of GgmlTypeIQ4NL: rowSizeIQ4NL(nCols)
    else: raise newException(ValueError, "unsupported quant type for GPU upload: " & $info.elemType)
  let totalBytes = rowSize * nRows
  let alloc = hippoMalloc(totalBytes)
  hippoMemcpyAsync(alloc.p, dataPtr, totalBytes, HippoMemcpyHostToDevice, gpuCtx.stream)
  result = GpuQuantWeight(devicePtr: alloc.p, alloc: alloc, sizeBytes: totalBytes,
                          nRows: nRows, nCols: nCols)
  quantWeightCache[name] = result

proc dequantF16ToF32Kernel(
  src: ptr uint16, dst: ptr float32, n: cint
) {.hippoGlobal.} =
  let idx = cint(blockIdx.x) * cint(blockDim.x) + cint(threadIdx.x)
  if idx < n:
    let s = cast[ptr UncheckedArray[uint16]](src)
    let d = cast[ptr UncheckedArray[float32]](dst)
    d[idx] = hippoHalfToFloat(s[idx])

proc cachedF16WeightAsF32*(name: string, m: var Model, tensorName: string): GpuTensor =
  if weightCache.hasKey(name):
    return weightCache[name]
  ensureGpuContext()
  let info = m.infos[tensorName]
  let nElems = tensorElemCount(info)
  let dataPtr = tensorDataPtr(m.gguf, info)
  let srcAlloc = hippoMalloc(nElems * 2)
  hippoMemcpyAsync(srcAlloc.p, dataPtr, nElems * 2,
                    HippoMemcpyHostToDevice, gpuCtx.stream)
  let dstAlloc = hippoMalloc(nElems * sizeof(float32))
  let blk = 256'u32
  let grid = ((nElems.uint32 + blk - 1) div blk)
  var srcPtr = srcAlloc.p; var dstPtr = dstAlloc.p; var nArg = nElems.cint
  hippoLaunchKernel(dequantF16ToF32Kernel, gridDim = newDim3(grid),
                    blockDim = newDim3(blk), stream = gpuCtx.stream,
                    args = hippoArgs(srcPtr, dstPtr, nArg))
  gpuStreamSync(gpuCtx.stream)
  var shape = newSeq[int](int(info.nDims))
  for i in 0 ..< int(info.nDims):
    shape[i] = int(info.ne[i])
  result = GpuTensor(devicePtr: dstAlloc.p, alloc: dstAlloc, shape: shape,
                     sizeBytes: nElems * sizeof(float32))
  weightCache[name] = result

# ---------------------------------------------------------------------------
# Kernel: Q2_K GEMV decode
# ---------------------------------------------------------------------------
proc linearQ2KDecodeKernel(
  wData: ptr uint8,
  xData, outData: ptr float32,
  outRows, wCols: cint
) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let tid = int(threadIdx.x)
  let cols = int(wCols)
  let baseRow = int(blockIdx.x) * HippoDecodeRowsPerBlock
  let w = cast[ptr UncheckedArray[uint8]](wData)
  let xArr = cast[ptr UncheckedArray[float32]](xData)
  let outArr = cast[ptr UncheckedArray[float32]](outData)
  let nBlocksPerRow = cols div 256
  let rowSizeBytes = nBlocksPerRow * 84
  let q2_chunk = tid shr 7
  let q2_localElem = tid and 127
  let q2_iteration = q2_localElem shr 5
  let q2_sub = (q2_localElem shr 4) and 1
  let q2_l = q2_localElem and 15
  let q2_scaleIdx = q2_chunk * 8 + q2_iteration * 2 + q2_sub
  let q2_shift = q2_iteration * 2
  let q2_qsByteIdx = q2_chunk * 32 + q2_sub * 16 + q2_l

  for r in 0 ..< HippoDecodeRowsPerBlock:
    let outRow = baseRow + r
    if outRow < int(outRows):
      let rowBase = outRow * rowSizeBytes
      var acc = 0.0'f32
      var blkIdx = 0
      while blkIdx + 3 < nBlocksPerRow:
        let blkStartA = rowBase + blkIdx * 84
        let elemBaseA = blkIdx * 256
        let dRawA = uint16(w[blkStartA + 80]) or (uint16(w[blkStartA + 81]) shl 8)
        let dminRawA = uint16(w[blkStartA + 82]) or (uint16(w[blkStartA + 83]) shl 8)
        let dA = hippoHalfToFloat(dRawA)
        let dminA = hippoHalfToFloat(dminRawA)
        let scA = w[blkStartA + q2_scaleIdx]
        let dlA = dA * cfloat(scA and 0x0F'u8)
        let mlA = dminA * cfloat(scA shr 4)
        let qvalA = cfloat((w[blkStartA + 16 + q2_qsByteIdx] shr q2_shift) and 3)
        acc = acc + (dlA * qvalA - mlA) * xArr[elemBaseA + tid]
        let blkStartB = rowBase + (blkIdx + 1) * 84
        let elemBaseB = (blkIdx + 1) * 256
        let dRawB = uint16(w[blkStartB + 80]) or (uint16(w[blkStartB + 81]) shl 8)
        let dminRawB = uint16(w[blkStartB + 82]) or (uint16(w[blkStartB + 83]) shl 8)
        let dB = hippoHalfToFloat(dRawB)
        let dminB = hippoHalfToFloat(dminRawB)
        let scB = w[blkStartB + q2_scaleIdx]
        let dlB = dB * cfloat(scB and 0x0F'u8)
        let mlB = dminB * cfloat(scB shr 4)
        let qvalB = cfloat((w[blkStartB + 16 + q2_qsByteIdx] shr q2_shift) and 3)
        acc = acc + (dlB * qvalB - mlB) * xArr[elemBaseB + tid]
        let blkStartC = rowBase + (blkIdx + 2) * 84
        let elemBaseC = (blkIdx + 2) * 256
        let dRawC = uint16(w[blkStartC + 80]) or (uint16(w[blkStartC + 81]) shl 8)
        let dminRawC = uint16(w[blkStartC + 82]) or (uint16(w[blkStartC + 83]) shl 8)
        let dC = hippoHalfToFloat(dRawC)
        let dminC = hippoHalfToFloat(dminRawC)
        let scC = w[blkStartC + q2_scaleIdx]
        let dlC = dC * cfloat(scC and 0x0F'u8)
        let mlC = dminC * cfloat(scC shr 4)
        let qvalC = cfloat((w[blkStartC + 16 + q2_qsByteIdx] shr q2_shift) and 3)
        acc = acc + (dlC * qvalC - mlC) * xArr[elemBaseC + tid]
        let blkStartD = rowBase + (blkIdx + 3) * 84
        let elemBaseD = (blkIdx + 3) * 256
        let dRawD = uint16(w[blkStartD + 80]) or (uint16(w[blkStartD + 81]) shl 8)
        let dminRawD = uint16(w[blkStartD + 82]) or (uint16(w[blkStartD + 83]) shl 8)
        let dD = hippoHalfToFloat(dRawD)
        let dminD = hippoHalfToFloat(dminRawD)
        let scD = w[blkStartD + q2_scaleIdx]
        let dlD = dD * cfloat(scD and 0x0F'u8)
        let mlD = dminD * cfloat(scD shr 4)
        let qvalD = cfloat((w[blkStartD + 16 + q2_qsByteIdx] shr q2_shift) and 3)
        acc = acc + (dlD * qvalD - mlD) * xArr[elemBaseD + tid]
        blkIdx = blkIdx + 4
      while blkIdx < nBlocksPerRow:
        let blkStart = rowBase + blkIdx * 84
        let elemBase = blkIdx * 256
        let dRaw = uint16(w[blkStart + 80]) or (uint16(w[blkStart + 81]) shl 8)
        let dminRaw = uint16(w[blkStart + 82]) or (uint16(w[blkStart + 83]) shl 8)
        let d = hippoHalfToFloat(dRaw)
        let dmin = hippoHalfToFloat(dminRaw)
        let sc = w[blkStart + q2_scaleIdx]
        let dl = d * cfloat(sc and 0x0F'u8)
        let ml = dmin * cfloat(sc shr 4)
        let qval = cfloat((w[blkStart + 16 + q2_qsByteIdx] shr q2_shift) and 3)
        acc = acc + (dl * qval - ml) * xArr[elemBase + tid]
        blkIdx = blkIdx + 1
      sdata[tid] = acc
    else:
      sdata[tid] = 0.0'f32
    hippoSyncthreads()
    reduceSum256(sdata, tid)
    if tid == 0 and outRow < int(outRows):
      outArr[outRow] = sdata[0]
    hippoSyncthreads()

when HippoWarpSize == 32:
  proc linearQ2KWarpDecodeKernel(
    wData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    ## Warp-per-row Q2_K GEMV.
    let tid = cint(threadIdx.x)
    let row = cint(blockIdx.x)
    if row >= outRows:
      return
    let w = cast[ptr UncheckedArray[uint8]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    let nBlocksPerRow = wCols div 256'i32
    let rowSizeBytes = nBlocksPerRow * 84'i32
    let rowBase = row * rowSizeBytes
    let sub = (tid shr 4'i32) and 1'i32
    let qsOff0 = 16'i32 + sub * 16'i32 + (tid and 15'i32)
    let qsOff1 = 48'i32 + sub * 16'i32 + (tid and 15'i32)
    var acc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * 84'i32
      let eb = blkIdx * 256'i32
      let dRaw = uint16(w[bs + 80'i32]) or (uint16(w[bs + 81'i32]) shl 8)
      let dmRaw = uint16(w[bs + 82'i32]) or (uint16(w[bs + 83'i32]) shl 8)
      let d = hippoHalfToFloat(dRaw)
      let dm = hippoHalfToFloat(dmRaw)
      let qb0 = w[bs + qsOff0]
      let qb1 = w[bs + qsOff1]
      let sc0 = w[bs + sub]
      acc = acc + (d * cfloat(sc0 and 0x0F) * cfloat(qb0 and 3) - dm * cfloat(sc0 shr 4)) * xArr[eb + tid]
      let sc1 = w[bs + 2'i32 + sub]
      acc = acc + (d * cfloat(sc1 and 0x0F) * cfloat((qb0 shr 2) and 3) - dm * cfloat(sc1 shr 4)) * xArr[eb + tid + 32'i32]
      let sc2 = w[bs + 4'i32 + sub]
      acc = acc + (d * cfloat(sc2 and 0x0F) * cfloat((qb0 shr 4) and 3) - dm * cfloat(sc2 shr 4)) * xArr[eb + tid + 64'i32]
      let sc3 = w[bs + 6'i32 + sub]
      acc = acc + (d * cfloat(sc3 and 0x0F) * cfloat((qb0 shr 6) and 3) - dm * cfloat(sc3 shr 4)) * xArr[eb + tid + 96'i32]
      let sc4 = w[bs + 8'i32 + sub]
      acc = acc + (d * cfloat(sc4 and 0x0F) * cfloat(qb1 and 3) - dm * cfloat(sc4 shr 4)) * xArr[eb + tid + 128'i32]
      let sc5 = w[bs + 10'i32 + sub]
      acc = acc + (d * cfloat(sc5 and 0x0F) * cfloat((qb1 shr 2) and 3) - dm * cfloat(sc5 shr 4)) * xArr[eb + tid + 160'i32]
      let sc6 = w[bs + 12'i32 + sub]
      acc = acc + (d * cfloat(sc6 and 0x0F) * cfloat((qb1 shr 4) and 3) - dm * cfloat(sc6 shr 4)) * xArr[eb + tid + 192'i32]
      let sc7 = w[bs + 14'i32 + sub]
      acc = acc + (d * cfloat(sc7 and 0x0F) * cfloat((qb1 shr 6) and 3) - dm * cfloat(sc7 shr 4)) * xArr[eb + tid + 224'i32]
      blkIdx = blkIdx + 1'i32
    acc = acc + hippoShflDown(acc, 16)
    acc = acc + hippoShflDown(acc, 8)
    acc = acc + hippoShflDown(acc, 4)
    acc = acc + hippoShflDown(acc, 2)
    acc = acc + hippoShflDown(acc, 1)
    if tid == 0'i32:
      outArr[row] = acc

proc gpuLinearColQ2K*(dst, x, wQuant: pointer, wCols, wRows: int,
                       stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoWarpSize.uint32)
    var wPtr = wQuant; var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(linearQ2KWarpDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    if wCols > HippoMaxDecodeCols:
      raise newException(ValueError, "Q2K decode width exceeds limit: " & $wCols)
    let grid = newDim3(((wRows + HippoDecodeRowsPerBlock - 1) div HippoDecodeRowsPerBlock).uint32)
    let blk = newDim3(HippoBlockSize.uint32)
    var wPtr = wQuant; var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(linearQ2KDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))

# ---------------------------------------------------------------------------
# Kernel: Q3_K GEMV decode
# ---------------------------------------------------------------------------
proc linearQ3KDecodeKernel(
  wData: ptr uint8,
  xData, outData: ptr float32,
  outRows, wCols: cint
) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let tid = int(threadIdx.x)
  let cols = int(wCols)
  let baseRow = int(blockIdx.x) * HippoDecodeRowsPerBlock
  let w = cast[ptr UncheckedArray[uint8]](wData)
  let xArr = cast[ptr UncheckedArray[float32]](xData)
  let outArr = cast[ptr UncheckedArray[float32]](outData)
  let nBlocksPerRow = cols div 256
  let rowSizeBytes = nBlocksPerRow * 110
  let chunk = tid shr 7
  let localElem = tid and 127
  let iteration = localElem shr 5
  let sub = (localElem shr 4) and 1
  let l = localElem and 15
  let scaleIdx = chunk * 8 + iteration * 2 + sub
  let shift = iteration * 2
  let qsByteIdx = chunk * 32 + sub * 16 + l
  let hmaskByteOff = qsByteIdx mod 32
  let hmaskBitPos = chunk * 4 + iteration
  let byteInGroup = scaleIdx and 3
  let auxIdx = scaleIdx shr 2
  let sByteOff = 96 + (auxIdx and 1) * 4 + byteInGroup
  let tByteOff = 96 + 8 + byteInGroup
  let useHighShift = (auxIdx shr 1) * 4
  let auxShift = auxIdx * 2

  for r in 0 ..< HippoDecodeRowsPerBlock:
    let outRow = baseRow + r
    if outRow < int(outRows):
      let rowBase = outRow * rowSizeBytes
      var acc = 0.0'f32
      var blkIdx = 0
      while blkIdx + 3 < nBlocksPerRow:
        let blkStartA = rowBase + blkIdx * 110
        let elemBaseA = blkIdx * 256
        let dRawA = uint16(w[blkStartA + 108]) or (uint16(w[blkStartA + 109]) shl 8)
        let dAllA = hippoHalfToFloat(dRawA)
        let sByteA = cint(w[blkStartA + sByteOff])
        let tByteA = cint(w[blkStartA + tByteOff])
        let lowA = (sByteA shr useHighShift) and 0x0F
        let highA = ((tByteA shr auxShift) and 0x03) shl 4
        let scaleByteA = lowA or highA
        let scaleSignedA = ((scaleByteA xor 0x80) - 0x80)
        let dlA = dAllA * cfloat(scaleSignedA - 32)
        let qvalA = cint((w[blkStartA + 32 + qsByteIdx] shr shift) and 3)
        let hmBitA = (cint(w[blkStartA + hmaskByteOff]) shr hmaskBitPos) and 1
        let hmA = 4 - hmBitA * 4
        acc = acc + dlA * cfloat(qvalA - hmA) * xArr[elemBaseA + tid]
        let blkStartB = rowBase + (blkIdx + 1) * 110
        let elemBaseB = (blkIdx + 1) * 256
        let dRawB = uint16(w[blkStartB + 108]) or (uint16(w[blkStartB + 109]) shl 8)
        let dAllB = hippoHalfToFloat(dRawB)
        let sByteB = cint(w[blkStartB + sByteOff])
        let tByteB = cint(w[blkStartB + tByteOff])
        let lowB = (sByteB shr useHighShift) and 0x0F
        let highB = ((tByteB shr auxShift) and 0x03) shl 4
        let scaleByteB = lowB or highB
        let scaleSignedB = ((scaleByteB xor 0x80) - 0x80)
        let dlB = dAllB * cfloat(scaleSignedB - 32)
        let qvalB = cint((w[blkStartB + 32 + qsByteIdx] shr shift) and 3)
        let hmBitB = (cint(w[blkStartB + hmaskByteOff]) shr hmaskBitPos) and 1
        let hmB = 4 - hmBitB * 4
        acc = acc + dlB * cfloat(qvalB - hmB) * xArr[elemBaseB + tid]
        let blkStartC = rowBase + (blkIdx + 2) * 110
        let elemBaseC = (blkIdx + 2) * 256
        let dRawC = uint16(w[blkStartC + 108]) or (uint16(w[blkStartC + 109]) shl 8)
        let dAllC = hippoHalfToFloat(dRawC)
        let sByteC = cint(w[blkStartC + sByteOff])
        let tByteC = cint(w[blkStartC + tByteOff])
        let lowC = (sByteC shr useHighShift) and 0x0F
        let highC = ((tByteC shr auxShift) and 0x03) shl 4
        let scaleByteC = lowC or highC
        let scaleSignedC = ((scaleByteC xor 0x80) - 0x80)
        let dlC = dAllC * cfloat(scaleSignedC - 32)
        let qvalC = cint((w[blkStartC + 32 + qsByteIdx] shr shift) and 3)
        let hmBitC = (cint(w[blkStartC + hmaskByteOff]) shr hmaskBitPos) and 1
        let hmC = 4 - hmBitC * 4
        acc = acc + dlC * cfloat(qvalC - hmC) * xArr[elemBaseC + tid]
        let blkStartD = rowBase + (blkIdx + 3) * 110
        let elemBaseD = (blkIdx + 3) * 256
        let dRawD = uint16(w[blkStartD + 108]) or (uint16(w[blkStartD + 109]) shl 8)
        let dAllD = hippoHalfToFloat(dRawD)
        let sByteD = cint(w[blkStartD + sByteOff])
        let tByteD = cint(w[blkStartD + tByteOff])
        let lowD = (sByteD shr useHighShift) and 0x0F
        let highD = ((tByteD shr auxShift) and 0x03) shl 4
        let scaleByteD = lowD or highD
        let scaleSignedD = ((scaleByteD xor 0x80) - 0x80)
        let dlD = dAllD * cfloat(scaleSignedD - 32)
        let qvalD = cint((w[blkStartD + 32 + qsByteIdx] shr shift) and 3)
        let hmBitD = (cint(w[blkStartD + hmaskByteOff]) shr hmaskBitPos) and 1
        let hmD = 4 - hmBitD * 4
        acc = acc + dlD * cfloat(qvalD - hmD) * xArr[elemBaseD + tid]
        blkIdx = blkIdx + 4
      while blkIdx < nBlocksPerRow:
        let blkStart = rowBase + blkIdx * 110
        let elemBase = blkIdx * 256
        let dRaw = uint16(w[blkStart + 108]) or (uint16(w[blkStart + 109]) shl 8)
        let dAll = hippoHalfToFloat(dRaw)
        let sByte = cint(w[blkStart + sByteOff])
        let tByte = cint(w[blkStart + tByteOff])
        let low = (sByte shr useHighShift) and 0x0F
        let high = ((tByte shr auxShift) and 0x03) shl 4
        let scaleByte = low or high
        let scaleSigned = ((scaleByte xor 0x80) - 0x80)
        let dl = dAll * cfloat(scaleSigned - 32)
        let qval = cint((w[blkStart + 32 + qsByteIdx] shr shift) and 3)
        let hmBit = (cint(w[blkStart + hmaskByteOff]) shr hmaskBitPos) and 1
        let hm = 4 - hmBit * 4
        acc = acc + dl * cfloat(qval - hm) * xArr[elemBase + tid]
        blkIdx = blkIdx + 1
      sdata[tid] = acc
    else:
      sdata[tid] = 0.0'f32
    hippoSyncthreads()
    reduceSum256(sdata, tid)
    if tid == 0 and outRow < int(outRows):
      outArr[outRow] = sdata[0]
    hippoSyncthreads()

when HippoWarpSize == 32:
  proc linearQ3KWarpDecodeKernel(
    wData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    ## Warp-per-row Q3_K GEMV.
    let tid = cint(threadIdx.x)
    let row = cint(blockIdx.x)
    if row >= outRows:
      return
    let w = cast[ptr UncheckedArray[uint8]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    let nBlocksPerRow = wCols div 256'i32
    let rowSizeBytes = nBlocksPerRow * 110'i32
    let rowBase = row * rowSizeBytes
    let sub = (tid shr 4'i32) and 1'i32
    let qsOff0 = 32'i32 + sub * 16'i32 + (tid and 15'i32)
    let qsOff1 = 64'i32 + sub * 16'i32 + (tid and 15'i32)
    let hmOff = sub * 16'i32 + (tid and 15'i32)
    var acc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * 110'i32
      let eb = blkIdx * 256'i32
      let dRaw = uint16(w[bs + 108'i32]) or (uint16(w[bs + 109'i32]) shl 8)
      let dAll = hippoHalfToFloat(dRaw)
      let qb0 = w[bs + qsOff0]
      let qb1 = w[bs + qsOff1]
      let hmByte = cint(w[bs + hmOff])
      template q3kElem(scaleIdx: cint, qByte: untyped, qShift, hmBitPos, xOff: cint) {.dirty.} =
        block:
          let si = scaleIdx
          let big = si and 3'i32
          let ai = si shr 2'i32
          let sByteVal = cint(w[bs + 96'i32 + (ai and 1'i32) * 4'i32 + big])
          let tByteVal = cint(w[bs + 104'i32 + big])
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

proc gpuLinearColQ3K*(dst, x, wQuant: pointer, wCols, wRows: int,
                       stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoWarpSize.uint32)
    var wPtr = wQuant; var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(linearQ3KWarpDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    if wCols > HippoMaxDecodeCols:
      raise newException(ValueError, "Q3K decode width exceeds limit: " & $wCols)
    let grid = newDim3(((wRows + HippoDecodeRowsPerBlock - 1) div HippoDecodeRowsPerBlock).uint32)
    let blk = newDim3(HippoBlockSize.uint32)
    var wPtr = wQuant; var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(linearQ3KDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))

# ---------------------------------------------------------------------------
# Fused K(Q2K)+V(Q3K) GEMV decode
# ---------------------------------------------------------------------------
when HippoWarpSize == 32:
  proc fusedKVQ2KQ3KWarpKernel(
    kData, vData: ptr uint8,
    xData, kOut, vOut: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    ## Fused K(Q2K)+V(Q3K) warp-per-row GEMV.
    let tid = cint(threadIdx.x)
    let row = cint(blockIdx.x)
    if row >= outRows:
      return
    let kw = cast[ptr UncheckedArray[uint8]](kData)
    let vw = cast[ptr UncheckedArray[uint8]](vData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let kArr = cast[ptr UncheckedArray[float32]](kOut)
    let vArr = cast[ptr UncheckedArray[float32]](vOut)
    let nBlocksPerRow = wCols div 256'i32
    let kRowSizeBytes = nBlocksPerRow * 84'i32
    let vRowSizeBytes = nBlocksPerRow * 110'i32
    let kRowBase = row * kRowSizeBytes
    let vRowBase = row * vRowSizeBytes
    let sub = (tid shr 4'i32) and 1'i32
    let kQsOff0 = 16'i32 + sub * 16'i32 + (tid and 15'i32)
    let kQsOff1 = 48'i32 + sub * 16'i32 + (tid and 15'i32)
    let vQsOff0 = 32'i32 + sub * 16'i32 + (tid and 15'i32)
    let vQsOff1 = 64'i32 + sub * 16'i32 + (tid and 15'i32)
    let vHmOff = sub * 16'i32 + (tid and 15'i32)
    var kAcc = 0.0'f32
    var vAcc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let kbs = kRowBase + blkIdx * 84'i32
      let vbs = vRowBase + blkIdx * 110'i32
      let eb = blkIdx * 256'i32
      let x0 = xArr[eb + tid]
      let x1 = xArr[eb + tid + 32'i32]
      let x2 = xArr[eb + tid + 64'i32]
      let x3 = xArr[eb + tid + 96'i32]
      let x4 = xArr[eb + tid + 128'i32]
      let x5 = xArr[eb + tid + 160'i32]
      let x6 = xArr[eb + tid + 192'i32]
      let x7 = xArr[eb + tid + 224'i32]
      let kDRaw = uint16(kw[kbs + 80'i32]) or (uint16(kw[kbs + 81'i32]) shl 8)
      let kDmRaw = uint16(kw[kbs + 82'i32]) or (uint16(kw[kbs + 83'i32]) shl 8)
      let kD = hippoHalfToFloat(kDRaw)
      let kDm = hippoHalfToFloat(kDmRaw)
      let kQb0 = kw[kbs + kQsOff0]
      let kQb1 = kw[kbs + kQsOff1]
      let kSc0 = kw[kbs + sub]
      kAcc = kAcc + (kD * cfloat(kSc0 and 0x0F) * cfloat(kQb0 and 3) - kDm * cfloat(kSc0 shr 4)) * x0
      let kSc1 = kw[kbs + 2'i32 + sub]
      kAcc = kAcc + (kD * cfloat(kSc1 and 0x0F) * cfloat((kQb0 shr 2) and 3) - kDm * cfloat(kSc1 shr 4)) * x1
      let kSc2 = kw[kbs + 4'i32 + sub]
      kAcc = kAcc + (kD * cfloat(kSc2 and 0x0F) * cfloat((kQb0 shr 4) and 3) - kDm * cfloat(kSc2 shr 4)) * x2
      let kSc3 = kw[kbs + 6'i32 + sub]
      kAcc = kAcc + (kD * cfloat(kSc3 and 0x0F) * cfloat((kQb0 shr 6) and 3) - kDm * cfloat(kSc3 shr 4)) * x3
      let kSc4 = kw[kbs + 8'i32 + sub]
      kAcc = kAcc + (kD * cfloat(kSc4 and 0x0F) * cfloat(kQb1 and 3) - kDm * cfloat(kSc4 shr 4)) * x4
      let kSc5 = kw[kbs + 10'i32 + sub]
      kAcc = kAcc + (kD * cfloat(kSc5 and 0x0F) * cfloat((kQb1 shr 2) and 3) - kDm * cfloat(kSc5 shr 4)) * x5
      let kSc6 = kw[kbs + 12'i32 + sub]
      kAcc = kAcc + (kD * cfloat(kSc6 and 0x0F) * cfloat((kQb1 shr 4) and 3) - kDm * cfloat(kSc6 shr 4)) * x6
      let kSc7 = kw[kbs + 14'i32 + sub]
      kAcc = kAcc + (kD * cfloat(kSc7 and 0x0F) * cfloat((kQb1 shr 6) and 3) - kDm * cfloat(kSc7 shr 4)) * x7
      let vDRaw = uint16(vw[vbs + 108'i32]) or (uint16(vw[vbs + 109'i32]) shl 8)
      let vDAll = hippoHalfToFloat(vDRaw)
      let vQb0 = vw[vbs + vQsOff0]
      let vQb1 = vw[vbs + vQsOff1]
      let vHmByte = cint(vw[vbs + vHmOff])
      template fusedQ3KElem(scaleIdx: cint, qByte: untyped, qShift, hmBitPos: cint, xVal: float32) {.dirty.} =
        block:
          let si = scaleIdx
          let big = si and 3'i32
          let ai = si shr 2'i32
          let sByteVal = cint(vw[vbs + 96'i32 + (ai and 1'i32) * 4'i32 + big])
          let tByteVal = cint(vw[vbs + 104'i32 + big])
          let low = (sByteVal shr ((ai shr 1'i32) * 4'i32)) and 0x0F'i32
          let high = ((tByteVal shr (ai * 2'i32)) and 0x03'i32) shl 4'i32
          let scByte = low or high
          let scSigned = (scByte xor 0x80'i32) - 0x80'i32
          let dl = vDAll * cfloat(scSigned - 32'i32)
          let qval = cint((qByte shr qShift) and 3)
          let hm = 4'i32 - ((vHmByte shr hmBitPos) and 1'i32) * 4'i32
          vAcc = vAcc + dl * cfloat(qval - hm) * xVal
      fusedQ3KElem(sub,            vQb0, 0'i32, 0'i32, x0)
      fusedQ3KElem(2'i32 + sub,    vQb0, 2'i32, 1'i32, x1)
      fusedQ3KElem(4'i32 + sub,    vQb0, 4'i32, 2'i32, x2)
      fusedQ3KElem(6'i32 + sub,    vQb0, 6'i32, 3'i32, x3)
      fusedQ3KElem(8'i32 + sub,    vQb1, 0'i32, 4'i32, x4)
      fusedQ3KElem(10'i32 + sub,   vQb1, 2'i32, 5'i32, x5)
      fusedQ3KElem(12'i32 + sub,   vQb1, 4'i32, 6'i32, x6)
      fusedQ3KElem(14'i32 + sub,   vQb1, 6'i32, 7'i32, x7)
      blkIdx = blkIdx + 1'i32
    kAcc = kAcc + hippoShflDown(kAcc, 16)
    kAcc = kAcc + hippoShflDown(kAcc, 8)
    kAcc = kAcc + hippoShflDown(kAcc, 4)
    kAcc = kAcc + hippoShflDown(kAcc, 2)
    kAcc = kAcc + hippoShflDown(kAcc, 1)
    vAcc = vAcc + hippoShflDown(vAcc, 16)
    vAcc = vAcc + hippoShflDown(vAcc, 8)
    vAcc = vAcc + hippoShflDown(vAcc, 4)
    vAcc = vAcc + hippoShflDown(vAcc, 2)
    vAcc = vAcc + hippoShflDown(vAcc, 1)
    if tid == 0'i32:
      kArr[row] = kAcc
      vArr[row] = vAcc

proc gpuFusedKVLinearQ2KQ3K*(kDst, vDst, x, kQuant, vQuant: pointer,
                              wCols, wRows: int, stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoWarpSize.uint32)
    var kPtr = kQuant; var vPtr = vQuant
    var xPtr = x; var kDstP = kDst; var vDstP = vDst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(fusedKVQ2KQ3KWarpKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(kPtr, vPtr, xPtr, kDstP, vDstP, outRowsArg, wColsArg))
  else:
    {.error: "gpuFusedKVLinearQ2KQ3K requires WarpSize == 32".}

# ---------------------------------------------------------------------------
# Q6_K GEMV decode (warp-per-row, WarpSize==32 only)
# ---------------------------------------------------------------------------
when HippoWarpSize == 32:
  proc linearQ6KWarpDecodeKernel(
    wData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    ## Warp-per-row Q6_K GEMV.
    let tid = cint(threadIdx.x)
    let row = cint(blockIdx.x)
    if row >= outRows:
      return
    let w = cast[ptr UncheckedArray[uint8]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    let nBlocksPerRow = wCols div 256'i32
    let rowSizeBytes = nBlocksPerRow * 210'i32
    let rowBase = row * rowSizeBytes
    var acc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * 210'i32
      let eb = blkIdx * 256'i32
      let dRaw = uint16(w[bs + 208'i32]) or (uint16(w[bs + 209'i32]) shl 8)
      let dAll = hippoHalfToFloat(dRaw)
      let scBase = bs + 192'i32
      let scHalf = tid shr 4'i32
      block:
        let qlA = w[bs + tid]
        let qlB = w[bs + 32'i32 + tid]
        let qhByte = w[bs + 128'i32 + tid]
        let sc0 = cast[int8](w[scBase + scHalf])
        let q1 = cint(qlA and 0x0F'u8) or (cint((qhByte shr 0'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc0) * cfloat(q1) * xArr[eb + tid]
        let sc1 = cast[int8](w[scBase + 2'i32 + scHalf])
        let q2 = cint(qlB and 0x0F'u8) or (cint((qhByte shr 2'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc1) * cfloat(q2) * xArr[eb + tid + 32'i32]
        let sc2 = cast[int8](w[scBase + 4'i32 + scHalf])
        let q3 = cint(qlA shr 4'u8) or (cint((qhByte shr 4'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc2) * cfloat(q3) * xArr[eb + tid + 64'i32]
        let sc3 = cast[int8](w[scBase + 6'i32 + scHalf])
        let q4 = cint(qlB shr 4'u8) or (cint((qhByte shr 6'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc3) * cfloat(q4) * xArr[eb + tid + 96'i32]
      block:
        let qlA = w[bs + 64'i32 + tid]
        let qlB = w[bs + 64'i32 + 32'i32 + tid]
        let qhByte = w[bs + 160'i32 + tid]
        let sc0 = cast[int8](w[scBase + 8'i32 + scHalf])
        let q1 = cint(qlA and 0x0F'u8) or (cint((qhByte shr 0'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc0) * cfloat(q1) * xArr[eb + tid + 128'i32]
        let sc1 = cast[int8](w[scBase + 10'i32 + scHalf])
        let q2 = cint(qlB and 0x0F'u8) or (cint((qhByte shr 2'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc1) * cfloat(q2) * xArr[eb + tid + 160'i32]
        let sc2 = cast[int8](w[scBase + 12'i32 + scHalf])
        let q3 = cint(qlA shr 4'u8) or (cint((qhByte shr 4'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc2) * cfloat(q3) * xArr[eb + tid + 192'i32]
        let sc3 = cast[int8](w[scBase + 14'i32 + scHalf])
        let q4 = cint(qlB shr 4'u8) or (cint((qhByte shr 6'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc3) * cfloat(q4) * xArr[eb + tid + 224'i32]
      blkIdx = blkIdx + 1'i32
    acc = acc + hippoShflDown(acc, 16)
    acc = acc + hippoShflDown(acc, 8)
    acc = acc + hippoShflDown(acc, 4)
    acc = acc + hippoShflDown(acc, 2)
    acc = acc + hippoShflDown(acc, 1)
    if tid == 0'i32:
      outArr[row] = acc

  const Q6KBlockRowsPerBlock = 8

  proc linearQ6KBlockDecodeKernel(
    wData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    ## Block-based Q6_K GEMV. 256 threads (8 warps), 8 rows per block.
    let tid = cint(threadIdx.x)
    let warpId = tid shr 5'i32
    let lane = tid and 31'i32
    let baseRow = cint(blockIdx.x) * Q6KBlockRowsPerBlock
    let row = baseRow + warpId
    let w = cast[ptr UncheckedArray[uint8]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    var sX {.hippoShared.}: array[2048, float32]
    var i = tid
    while i < wCols:
      sX[i] = xArr[i]
      i = i + 256'i32
    hippoSyncthreads()
    if row >= outRows:
      return
    let nBlocksPerRow = wCols div 256'i32
    let rowSizeBytes = nBlocksPerRow * 210'i32
    let rowBase = row * rowSizeBytes
    var acc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * 210'i32
      let eb = blkIdx * 256'i32
      let dRaw = uint16(w[bs + 208'i32]) or (uint16(w[bs + 209'i32]) shl 8)
      let dAll = hippoHalfToFloat(dRaw)
      let scBase = bs + 192'i32
      let scHalf = lane shr 4'i32
      block:
        let qlA = w[bs + lane]
        let qlB = w[bs + 32'i32 + lane]
        let qhByte = w[bs + 128'i32 + lane]
        let sc0 = cast[int8](w[scBase + scHalf])
        let q1 = cint(qlA and 0x0F'u8) or (cint((qhByte shr 0'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc0) * cfloat(q1) * sX[eb + lane]
        let sc1 = cast[int8](w[scBase + 2'i32 + scHalf])
        let q2 = cint(qlB and 0x0F'u8) or (cint((qhByte shr 2'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc1) * cfloat(q2) * sX[eb + lane + 32'i32]
        let sc2 = cast[int8](w[scBase + 4'i32 + scHalf])
        let q3 = cint(qlA shr 4'u8) or (cint((qhByte shr 4'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc2) * cfloat(q3) * sX[eb + lane + 64'i32]
        let sc3 = cast[int8](w[scBase + 6'i32 + scHalf])
        let q4 = cint(qlB shr 4'u8) or (cint((qhByte shr 6'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc3) * cfloat(q4) * sX[eb + lane + 96'i32]
      block:
        let qlA = w[bs + 64'i32 + lane]
        let qlB = w[bs + 64'i32 + 32'i32 + lane]
        let qhByte = w[bs + 160'i32 + lane]
        let sc0 = cast[int8](w[scBase + 8'i32 + scHalf])
        let q1 = cint(qlA and 0x0F'u8) or (cint((qhByte shr 0'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc0) * cfloat(q1) * sX[eb + lane + 128'i32]
        let sc1 = cast[int8](w[scBase + 10'i32 + scHalf])
        let q2 = cint(qlB and 0x0F'u8) or (cint((qhByte shr 2'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc1) * cfloat(q2) * sX[eb + lane + 160'i32]
        let sc2 = cast[int8](w[scBase + 12'i32 + scHalf])
        let q3 = cint(qlA shr 4'u8) or (cint((qhByte shr 4'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc2) * cfloat(q3) * sX[eb + lane + 192'i32]
        let sc3 = cast[int8](w[scBase + 14'i32 + scHalf])
        let q4 = cint(qlB shr 4'u8) or (cint((qhByte shr 6'u8) and 3'u8) shl 4'i32) - 32'i32
        acc = acc + dAll * cfloat(sc3) * cfloat(q4) * sX[eb + lane + 224'i32]
      blkIdx = blkIdx + 1'i32
    acc = acc + hippoShflDown(acc, 16)
    acc = acc + hippoShflDown(acc, 8)
    acc = acc + hippoShflDown(acc, 4)
    acc = acc + hippoShflDown(acc, 2)
    acc = acc + hippoShflDown(acc, 1)
    if lane == 0'i32:
      outArr[row] = acc

proc gpuLinearColQ6K*(dst, x, wQuant: pointer, wCols, wRows: int,
                       stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoWarpSize.uint32)
    var wPtr = wQuant; var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(linearQ6KWarpDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    {.error: "gpuLinearColQ6K requires WarpSize == 32".}

# ---------------------------------------------------------------------------
# Q8_0 GEMV decode (warp-per-row, WarpSize==32 only)
# ---------------------------------------------------------------------------
when HippoWarpSize == 32:
  proc linearQ8_0WarpDecodeKernel(
    wData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    let tid = cint(threadIdx.x)
    let row = cint(blockIdx.x)
    if row >= outRows:
      return
    let w = cast[ptr UncheckedArray[uint8]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    let nBlocksPerRow = wCols div 32'i32
    let rowSizeBytes = nBlocksPerRow * 34'i32
    let rowBase = row * rowSizeBytes
    var acc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * 34'i32
      let eb = blkIdx * 32'i32
      let dRaw = uint16(w[bs]) or (uint16(w[bs + 1'i32]) shl 8)
      let d = hippoHalfToFloat(dRaw)
      let qVal = cast[int8](w[bs + 2'i32 + tid])
      acc = acc + d * cfloat(qVal) * xArr[eb + tid]
      blkIdx = blkIdx + 1'i32
    acc = acc + hippoShflDown(acc, 16)
    acc = acc + hippoShflDown(acc, 8)
    acc = acc + hippoShflDown(acc, 4)
    acc = acc + hippoShflDown(acc, 2)
    acc = acc + hippoShflDown(acc, 1)
    if tid == 0'i32:
      outArr[row] = acc

proc gpuLinearColQ8_0*(dst, x, wQuant: pointer, wCols, wRows: int,
                        stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoWarpSize.uint32)
    var wPtr = wQuant; var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(linearQ8_0WarpDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    {.error: "gpuLinearColQ8_0 requires WarpSize == 32".}

# ---------------------------------------------------------------------------
# Q5_0 GEMV decode (warp-per-row, WarpSize==32 only)
# ---------------------------------------------------------------------------
when HippoWarpSize == 32:
  proc linearQ5_0WarpDecodeKernel(
    wData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    let tid = cint(threadIdx.x)
    let row = cint(blockIdx.x)
    if row >= outRows:
      return
    let w = cast[ptr UncheckedArray[uint8]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    let nBlocksPerRow = wCols div 32'i32
    let rowSizeBytes = nBlocksPerRow * 22'i32
    let rowBase = row * rowSizeBytes
    var acc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * 22'i32
      let eb = blkIdx * 32'i32
      let dRaw = uint16(w[bs]) or (uint16(w[bs + 1'i32]) shl 8)
      let d = hippoHalfToFloat(dRaw)
      let qhBits = uint32(w[bs + 2'i32]) or (uint32(w[bs + 3'i32]) shl 8) or
                   (uint32(w[bs + 4'i32]) shl 16) or (uint32(w[bs + 5'i32]) shl 24)
      # GGML Q5_0: elements 0..15 = low nibble of qs[0..15], elements 16..31 = high nibble of qs[0..15]
      let nibbleByte = w[bs + 6'i32 + (tid and 15'i32)]
      let lo = if tid < 16'i32: nibbleByte and 0x0F'u8
               else: nibbleByte shr 4
      let hi = uint8((qhBits shr uint32(tid)) and 1'u32) shl 4
      let qVal = cint(hi or lo) - 16'i32
      acc = acc + d * cfloat(qVal) * xArr[eb + tid]
      blkIdx = blkIdx + 1'i32
    acc = acc + hippoShflDown(acc, 16)
    acc = acc + hippoShflDown(acc, 8)
    acc = acc + hippoShflDown(acc, 4)
    acc = acc + hippoShflDown(acc, 2)
    acc = acc + hippoShflDown(acc, 1)
    if tid == 0'i32:
      outArr[row] = acc

proc gpuLinearColQ5_0*(dst, x, wQuant: pointer, wCols, wRows: int,
                        stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoWarpSize.uint32)
    var wPtr = wQuant; var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(linearQ5_0WarpDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    {.error: "gpuLinearColQ5_0 requires WarpSize == 32".}

# ---------------------------------------------------------------------------
# Q5_1 GEMV decode (warp-per-row, WarpSize==32 only)
# ---------------------------------------------------------------------------
when HippoWarpSize == 32:
  proc linearQ5_1WarpDecodeKernel(
    wData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    let tid = cint(threadIdx.x)
    let row = cint(blockIdx.x)
    if row >= outRows:
      return
    let w = cast[ptr UncheckedArray[uint8]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    let nBlocksPerRow = wCols div 32'i32
    let rowSizeBytes = nBlocksPerRow * 24'i32  # BlockQ5_1Size = 24
    let rowBase = row * rowSizeBytes
    var acc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * 24'i32
      let eb = blkIdx * 32'i32
      let dRaw = uint16(w[bs]) or (uint16(w[bs + 1'i32]) shl 8)
      let mRaw = uint16(w[bs + 2'i32]) or (uint16(w[bs + 3'i32]) shl 8)
      let d = hippoHalfToFloat(dRaw)
      let m = hippoHalfToFloat(mRaw)
      let qhBits = uint32(w[bs + 4'i32]) or (uint32(w[bs + 5'i32]) shl 8) or
                   (uint32(w[bs + 6'i32]) shl 16) or (uint32(w[bs + 7'i32]) shl 24)
      let nibbleByte = w[bs + 8'i32 + (tid and 15'i32)]
      let lo = if tid < 16'i32: nibbleByte and 0x0F'u8
               else: nibbleByte shr 4
      let hi = uint8((qhBits shr uint32(tid)) and 1'u32) shl 4
      let qVal = cfloat(hi or lo)
      acc = acc + (d * qVal + m) * xArr[eb + tid]
      blkIdx = blkIdx + 1'i32
    acc = acc + hippoShflDown(acc, 16)
    acc = acc + hippoShflDown(acc, 8)
    acc = acc + hippoShflDown(acc, 4)
    acc = acc + hippoShflDown(acc, 2)
    acc = acc + hippoShflDown(acc, 1)
    if tid == 0'i32:
      outArr[row] = acc

proc gpuLinearColQ5_1*(dst, x, wQuant: pointer, wCols, wRows: int,
                        stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoWarpSize.uint32)
    var wPtr = wQuant; var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(linearQ5_1WarpDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    {.error: "gpuLinearColQ5_1 requires WarpSize == 32".}

# ---------------------------------------------------------------------------
# IQ4_NL GEMV decode (warp-per-row, WarpSize==32 only)
# ---------------------------------------------------------------------------
when HippoWarpSize == 32:
  proc linearIQ4NLWarpDecodeKernel(
    wData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    let tid = cint(threadIdx.x)
    let row = cint(blockIdx.x)
    if row >= outRows:
      return
    let w = cast[ptr UncheckedArray[uint8]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    let nBlocksPerRow = wCols div 32'i32
    let rowSizeBytes = nBlocksPerRow * 18'i32  # BlockIQ4NLSize = 18
    let rowBase = row * rowSizeBytes
    var acc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * 18'i32
      let eb = blkIdx * 32'i32
      let dRaw = uint16(w[bs]) or (uint16(w[bs + 1'i32]) shl 8)
      let d = hippoHalfToFloat(dRaw)
      let nibbleByte = w[bs + 2'i32 + (tid and 15'i32)]
      let nibble = if tid < 16'i32: nibbleByte and 0x0F'u8
                   else: nibbleByte shr 4
      # IQ4_NL lookup table (compile-time constant in registers)
      var qVal: cint
      case nibble
      of 0'u8: qVal = -127'i32
      of 1'u8: qVal = -104'i32
      of 2'u8: qVal = -83'i32
      of 3'u8: qVal = -65'i32
      of 4'u8: qVal = -49'i32
      of 5'u8: qVal = -35'i32
      of 6'u8: qVal = -22'i32
      of 7'u8: qVal = -10'i32
      of 8'u8: qVal = 1'i32
      of 9'u8: qVal = 13'i32
      of 10'u8: qVal = 25'i32
      of 11'u8: qVal = 38'i32
      of 12'u8: qVal = 53'i32
      of 13'u8: qVal = 69'i32
      of 14'u8: qVal = 89'i32
      else: qVal = 113'i32
      acc = acc + d * cfloat(qVal) * xArr[eb + tid]
      blkIdx = blkIdx + 1'i32
    acc = acc + hippoShflDown(acc, 16)
    acc = acc + hippoShflDown(acc, 8)
    acc = acc + hippoShflDown(acc, 4)
    acc = acc + hippoShflDown(acc, 2)
    acc = acc + hippoShflDown(acc, 1)
    if tid == 0'i32:
      outArr[row] = acc

proc gpuLinearColIQ4NL*(dst, x, wQuant: pointer, wCols, wRows: int,
                         stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoWarpSize.uint32)
    var wPtr = wQuant; var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(linearIQ4NLWarpDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    {.error: "gpuLinearColIQ4NL requires WarpSize == 32".}

# ---------------------------------------------------------------------------
# Q5_K GEMV decode (warp-per-row, WarpSize==32 only)
# ---------------------------------------------------------------------------
when HippoWarpSize == 32:
  proc linearQ5KWarpDecodeKernel(
    wData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    let tid = cint(threadIdx.x)
    let row = cint(blockIdx.x)
    if row >= outRows:
      return
    let w = cast[ptr UncheckedArray[uint8]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    let nBlocksPerRow = wCols div 256'i32
    let blockSizeBytes = 176'i32  # BlockQ5KSize
    let rowBase = row * nBlocksPerRow * blockSizeBytes
    var acc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * blockSizeBytes
      let dRaw = uint16(w[bs]) or (uint16(w[bs + 1'i32]) shl 8)
      let dminRaw = uint16(w[bs + 2'i32]) or (uint16(w[bs + 3'i32]) shl 8)
      let d = hippoHalfToFloat(dRaw)
      let dmin = hippoHalfToFloat(dminRaw)
      let scales = cast[ptr UncheckedArray[uint8]](addr w[bs + 4'i32])
      let qh = cast[ptr UncheckedArray[uint8]](addr w[bs + 16'i32])
      let qs = cast[ptr UncheckedArray[uint8]](addr w[bs + 16'i32 + 32'i32])
      let elemBase = blkIdx * 256'i32
      for subBlock in 0'i32 ..< 8'i32:
        let isIdx = subBlock
        var sc, mn: uint8
        if isIdx < 4'i32:
          sc = scales[isIdx] and 63'u8
          mn = scales[isIdx + 4'i32] and 63'u8
        else:
          sc = (scales[isIdx + 4'i32] and 0x0F'u8) or ((scales[isIdx - 4'i32] shr 6'u8) shl 4'u8)
          mn = (scales[isIdx + 4'i32] shr 4'u8) or ((scales[isIdx] shr 6'u8) shl 4'u8)
        let dl = d * cfloat(sc)
        let ml = dmin * cfloat(mn)
        let subBase = subBlock * 32'i32
        let qOff = (subBlock div 2'i32) * 32'i32
        let qShift = if (subBlock and 1'i32) == 0'i32: 0'i32 else: 4'i32
        let qhBitIdx = subBlock div 2'i32
        if tid < 32'i32:
          let qByte = qs[qOff + tid]
          let nibble = if qShift == 0'i32: qByte and 0x0F'u8
                       else: qByte shr 4
          let hBit = uint8((qh[tid] shr uint8(qhBitIdx * 2'i32 + (subBlock and 1'i32))) and 1'u8) shl 4
          let qVal = cfloat(nibble or hBit)
          acc = acc + (dl * qVal - ml) * xArr[elemBase + subBase + tid]
      blkIdx = blkIdx + 1'i32
    acc = acc + hippoShflDown(acc, 16)
    acc = acc + hippoShflDown(acc, 8)
    acc = acc + hippoShflDown(acc, 4)
    acc = acc + hippoShflDown(acc, 2)
    acc = acc + hippoShflDown(acc, 1)
    if tid == 0'i32:
      outArr[row] = acc

proc gpuLinearColQ5K*(dst, x, wQuant: pointer, wCols, wRows: int,
                       stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoWarpSize.uint32)
    var wPtr = wQuant; var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(linearQ5KWarpDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    {.error: "gpuLinearColQ5K requires WarpSize == 32".}

# ---------------------------------------------------------------------------
# F16 GEMV decode (warp-per-row, WarpSize==32 only)
# ---------------------------------------------------------------------------
when HippoWarpSize == 32:
  proc linearF16WarpDecodeKernel(
    wData: ptr uint16,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    let tid = cint(threadIdx.x)
    let row = cint(blockIdx.x)
    if row >= outRows:
      return
    let w = cast[ptr UncheckedArray[uint16]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    let rowBase = row * wCols
    var acc = 0.0'f32
    var col = tid
    while col < wCols:
      acc = acc + hippoHalfToFloat(w[rowBase + col]) * xArr[col]
      col = col + 32'i32
    acc = acc + hippoShflDown(acc, 16)
    acc = acc + hippoShflDown(acc, 8)
    acc = acc + hippoShflDown(acc, 4)
    acc = acc + hippoShflDown(acc, 2)
    acc = acc + hippoShflDown(acc, 1)
    if tid == 0'i32:
      outArr[row] = acc

proc gpuLinearColF16*(dst, x, wQuant: pointer, wCols, wRows: int,
                      stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoWarpSize.uint32)
    var wPtr = wQuant; var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(linearF16WarpDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    {.error: "gpuLinearColF16 requires WarpSize == 32".}

# ---------------------------------------------------------------------------
# Q4_K GEMV decode (warp-per-row, WarpSize==32 only)
# ---------------------------------------------------------------------------
when HippoWarpSize == 32:
  proc getScaleMinK4Gpu(j: cint, sc: ptr UncheckedArray[uint8], dOut, mOut: var uint8) {.hippoDevice.} =
    if j < 4'i32:
      dOut = sc[j] and 63'u8
      mOut = sc[j + 4'i32] and 63'u8
    else:
      dOut = (sc[j + 4'i32] and 0x0F'u8) or ((sc[j - 4'i32] shr 6'u8) shl 4'u8)
      mOut = (sc[j + 4'i32] shr 4'u8) or ((sc[j] shr 6'u8) shl 4'u8)

  proc linearQ4KWarpDecodeKernel(
    wData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    ## Warp-per-row Q4_K GEMV. Each thread handles 8 elements per block.
    let tid = cint(threadIdx.x)
    let row = cint(blockIdx.x)
    if row >= outRows:
      return
    let w = cast[ptr UncheckedArray[uint8]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    let nBlocksPerRow = wCols div 256'i32
    let rowSizeBytes = nBlocksPerRow * 144'i32
    let rowBase = row * rowSizeBytes
    var acc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * 144'i32
      let eb = blkIdx * 256'i32
      let dRaw = uint16(w[bs]) or (uint16(w[bs + 1'i32]) shl 8)
      let dmRaw = uint16(w[bs + 2'i32]) or (uint16(w[bs + 3'i32]) shl 8)
      let dAll = hippoHalfToFloat(dRaw)
      let dMin = hippoHalfToFloat(dmRaw)
      let scales = cast[ptr UncheckedArray[uint8]](addr w[bs + 4'i32])
      let qs = cast[ptr UncheckedArray[uint8]](addr w[bs + 16'i32])
      var isIdx = 0'i32
      var qOff = 0'i32
      for grp in countup(0'i32, 3'i32):
        var sc0, mn0, sc1, mn1: uint8
        getScaleMinK4Gpu(isIdx, scales, sc0, mn0)
        let d1 = dAll * cfloat(sc0)
        let m1 = dMin * cfloat(mn0)
        getScaleMinK4Gpu(isIdx + 1'i32, scales, sc1, mn1)
        let d2 = dAll * cfloat(sc1)
        let m2 = dMin * cfloat(mn1)
        let qByte = qs[qOff + tid]
        acc = acc + (d1 * cfloat(qByte and 0x0F'u8) - m1) * xArr[eb + grp * 64'i32 + tid]
        acc = acc + (d2 * cfloat(qByte shr 4'u8) - m2) * xArr[eb + grp * 64'i32 + 32'i32 + tid]
        isIdx = isIdx + 2'i32
        qOff = qOff + 32'i32
      blkIdx = blkIdx + 1'i32
    acc = acc + hippoShflDown(acc, 16)
    acc = acc + hippoShflDown(acc, 8)
    acc = acc + hippoShflDown(acc, 4)
    acc = acc + hippoShflDown(acc, 2)
    acc = acc + hippoShflDown(acc, 1)
    if tid == 0'i32:
      outArr[row] = acc

  const Q4KBlockRowsPerBlock = 8

  proc linearQ4KBlockDecodeKernel(
    wData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    ## Block-based Q4_K GEMV. 256 threads (8 warps), 8 rows per block.
    ## Shared memory sized for wCols <= 2048.
    let tid = cint(threadIdx.x)
    let warpId = tid shr 5'i32
    let lane = tid and 31'i32
    let baseRow = cint(blockIdx.x) * Q4KBlockRowsPerBlock
    let row = baseRow + warpId
    let w = cast[ptr UncheckedArray[uint8]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    var sX {.hippoShared.}: array[2048, float32]
    var i = tid
    while i < wCols:
      sX[i] = xArr[i]
      i = i + 256'i32
    hippoSyncthreads()
    if row >= outRows:
      return
    let nBlocksPerRow = wCols div 256'i32
    let rowSizeBytes = nBlocksPerRow * 144'i32
    let rowBase = row * rowSizeBytes
    var acc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * 144'i32
      let eb = blkIdx * 256'i32
      let dRaw = uint16(w[bs]) or (uint16(w[bs + 1'i32]) shl 8)
      let dmRaw = uint16(w[bs + 2'i32]) or (uint16(w[bs + 3'i32]) shl 8)
      let dAll = hippoHalfToFloat(dRaw)
      let dMin = hippoHalfToFloat(dmRaw)
      let scales = cast[ptr UncheckedArray[uint8]](addr w[bs + 4'i32])
      let qs = cast[ptr UncheckedArray[uint8]](addr w[bs + 16'i32])
      var isIdx = 0'i32
      var qOff = 0'i32
      for grp in countup(0'i32, 3'i32):
        var sc0, mn0, sc1, mn1: uint8
        getScaleMinK4Gpu(isIdx, scales, sc0, mn0)
        let d1 = dAll * cfloat(sc0)
        let m1 = dMin * cfloat(mn0)
        getScaleMinK4Gpu(isIdx + 1'i32, scales, sc1, mn1)
        let d2 = dAll * cfloat(sc1)
        let m2 = dMin * cfloat(mn1)
        let qByte = qs[qOff + lane]
        acc = acc + (d1 * cfloat(qByte and 0x0F'u8) - m1) * sX[eb + grp * 64'i32 + lane]
        acc = acc + (d2 * cfloat(qByte shr 4'u8) - m2) * sX[eb + grp * 64'i32 + 32'i32 + lane]
        isIdx = isIdx + 2'i32
        qOff = qOff + 32'i32
      blkIdx = blkIdx + 1'i32
    acc = acc + hippoShflDown(acc, 16)
    acc = acc + hippoShflDown(acc, 8)
    acc = acc + hippoShflDown(acc, 4)
    acc = acc + hippoShflDown(acc, 2)
    acc = acc + hippoShflDown(acc, 1)
    if lane == 0'i32:
      outArr[row] = acc

  proc linearQ4KBlockWideDecodeKernel(
    wData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    ## Block-based Q4_K GEMV for wide inputs (wCols > 2048, e.g. FFN down).
    let tid = cint(threadIdx.x)
    let warpId = tid shr 5'i32
    let lane = tid and 31'i32
    let baseRow = cint(blockIdx.x) * Q4KBlockRowsPerBlock
    let row = baseRow + warpId
    let w = cast[ptr UncheckedArray[uint8]](wData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    var sX {.hippoShared.}: array[8192, float32]
    var i = tid
    while i < wCols:
      sX[i] = xArr[i]
      i = i + 256'i32
    hippoSyncthreads()
    if row >= outRows:
      return
    let nBlocksPerRow = wCols div 256'i32
    let rowSizeBytes = nBlocksPerRow * 144'i32
    let rowBase = row * rowSizeBytes
    var acc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let bs = rowBase + blkIdx * 144'i32
      let eb = blkIdx * 256'i32
      let dRaw = uint16(w[bs]) or (uint16(w[bs + 1'i32]) shl 8)
      let dmRaw = uint16(w[bs + 2'i32]) or (uint16(w[bs + 3'i32]) shl 8)
      let dAll = hippoHalfToFloat(dRaw)
      let dMin = hippoHalfToFloat(dmRaw)
      let scales = cast[ptr UncheckedArray[uint8]](addr w[bs + 4'i32])
      let qs = cast[ptr UncheckedArray[uint8]](addr w[bs + 16'i32])
      var isIdx = 0'i32
      var qOff = 0'i32
      for grp in countup(0'i32, 3'i32):
        var sc0, mn0, sc1, mn1: uint8
        getScaleMinK4Gpu(isIdx, scales, sc0, mn0)
        let d1 = dAll * cfloat(sc0)
        let m1 = dMin * cfloat(mn0)
        getScaleMinK4Gpu(isIdx + 1'i32, scales, sc1, mn1)
        let d2 = dAll * cfloat(sc1)
        let m2 = dMin * cfloat(mn1)
        let qByte = qs[qOff + lane]
        acc = acc + (d1 * cfloat(qByte and 0x0F'u8) - m1) * sX[eb + grp * 64'i32 + lane]
        acc = acc + (d2 * cfloat(qByte shr 4'u8) - m2) * sX[eb + grp * 64'i32 + 32'i32 + lane]
        isIdx = isIdx + 2'i32
        qOff = qOff + 32'i32
      blkIdx = blkIdx + 1'i32
    acc = acc + hippoShflDown(acc, 16)
    acc = acc + hippoShflDown(acc, 8)
    acc = acc + hippoShflDown(acc, 4)
    acc = acc + hippoShflDown(acc, 2)
    acc = acc + hippoShflDown(acc, 1)
    if lane == 0'i32:
      outArr[row] = acc

proc gpuLinearColQ4K*(dst, x, wQuant: pointer, wCols, wRows: int,
                       stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoWarpSize.uint32)
    var wPtr = wQuant; var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(linearQ4KWarpDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    {.error: "gpuLinearColQ4K requires WarpSize == 32".}

proc gpuLinearColQuant*(dst, x, wQuant: pointer, wCols, wRows: int,
                         quantType: int32, stream: HippoStream) =
  ## Dispatch to the appropriate quantized GEMV kernel.
  case quantType
  of GgmlTypeF16: gpuLinearColF16(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ5_0: gpuLinearColQ5_0(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ2K: gpuLinearColQ2K(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ3K: gpuLinearColQ3K(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ4K: gpuLinearColQ4K(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ6K: gpuLinearColQ6K(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ8_0: gpuLinearColQ8_0(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ5_1: gpuLinearColQ5_1(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ5K: gpuLinearColQ5K(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeIQ4NL: gpuLinearColIQ4NL(dst, x, wQuant, wCols, wRows, stream)
  else: raise newException(ValueError, "unsupported quant type for GPU GEMV: " & $quantType)

# ---------------------------------------------------------------------------
# Fused Gate+Up+SiLU for Q3_K (warp-per-row, WarpSize==32 only)
# ---------------------------------------------------------------------------
when HippoWarpSize == 32:
  proc fusedGateUpSiluQ3KWarpKernel(
    gateData, upData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    ## Fused gate+up+silu Q3K GEMV.
    let tid = cint(threadIdx.x)
    let row = cint(blockIdx.x)
    if row >= outRows:
      return
    let gw = cast[ptr UncheckedArray[uint8]](gateData)
    let uw = cast[ptr UncheckedArray[uint8]](upData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    let nBlocksPerRow = wCols div 256'i32
    let rowSizeBytes = nBlocksPerRow * 110'i32
    let rowBase = row * rowSizeBytes
    let sub = (tid shr 4'i32) and 1'i32
    let qsOff0 = 32'i32 + sub * 16'i32 + (tid and 15'i32)
    let qsOff1 = 64'i32 + sub * 16'i32 + (tid and 15'i32)
    let hmOff = sub * 16'i32 + (tid and 15'i32)
    var gateAcc = 0.0'f32
    var upAcc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let gbs = rowBase + blkIdx * 110'i32
      let ubs = rowBase + blkIdx * 110'i32
      let eb = blkIdx * 256'i32
      let x0 = xArr[eb + tid]
      let x1 = xArr[eb + tid + 32'i32]
      let x2 = xArr[eb + tid + 64'i32]
      let x3 = xArr[eb + tid + 96'i32]
      let x4 = xArr[eb + tid + 128'i32]
      let x5 = xArr[eb + tid + 160'i32]
      let x6 = xArr[eb + tid + 192'i32]
      let x7 = xArr[eb + tid + 224'i32]
      let gDRaw = uint16(gw[gbs + 108'i32]) or (uint16(gw[gbs + 109'i32]) shl 8)
      let gDAll = hippoHalfToFloat(gDRaw)
      let gQb0 = gw[gbs + qsOff0]
      let gQb1 = gw[gbs + qsOff1]
      let gHmByte = cint(gw[gbs + hmOff])
      let uDRaw = uint16(uw[ubs + 108'i32]) or (uint16(uw[ubs + 109'i32]) shl 8)
      let uDAll = hippoHalfToFloat(uDRaw)
      let uQb0 = uw[ubs + qsOff0]
      let uQb1 = uw[ubs + qsOff1]
      let uHmByte = cint(uw[ubs + hmOff])
      template fusedElem(scaleIdx: cint, gQByte, uQByte: untyped, qShift, hmBitPos: cint, xVal: float32) {.dirty.} =
        block:
          let si = scaleIdx
          let big = si and 3'i32
          let ai = si shr 2'i32
          let sOff = 96'i32 + (ai and 1'i32) * 4'i32 + big
          let tOff = 104'i32 + big
          let lowShift = (ai shr 1'i32) * 4'i32
          let highShift = ai * 2'i32
          let gLow = (cint(gw[gbs + sOff]) shr lowShift) and 0x0F'i32
          let gHigh = ((cint(gw[gbs + tOff]) shr highShift) and 0x03'i32) shl 4'i32
          let gSc = ((gLow or gHigh) xor 0x80'i32) - 0x80'i32
          let gDl = gDAll * cfloat(gSc - 32'i32)
          let gQval = cint((gQByte shr qShift) and 3)
          let gHm = 4'i32 - ((gHmByte shr hmBitPos) and 1'i32) * 4'i32
          gateAcc = gateAcc + gDl * cfloat(gQval - gHm) * xVal
          let uLow = (cint(uw[ubs + sOff]) shr lowShift) and 0x0F'i32
          let uHigh = ((cint(uw[ubs + tOff]) shr highShift) and 0x03'i32) shl 4'i32
          let uSc = ((uLow or uHigh) xor 0x80'i32) - 0x80'i32
          let uDl = uDAll * cfloat(uSc - 32'i32)
          let uQval = cint((uQByte shr qShift) and 3)
          let uHm = 4'i32 - ((uHmByte shr hmBitPos) and 1'i32) * 4'i32
          upAcc = upAcc + uDl * cfloat(uQval - uHm) * xVal
      fusedElem(sub,            gQb0, uQb0, 0'i32, 0'i32, x0)
      fusedElem(2'i32 + sub,    gQb0, uQb0, 2'i32, 1'i32, x1)
      fusedElem(4'i32 + sub,    gQb0, uQb0, 4'i32, 2'i32, x2)
      fusedElem(6'i32 + sub,    gQb0, uQb0, 6'i32, 3'i32, x3)
      fusedElem(8'i32 + sub,    gQb1, uQb1, 0'i32, 4'i32, x4)
      fusedElem(10'i32 + sub,   gQb1, uQb1, 2'i32, 5'i32, x5)
      fusedElem(12'i32 + sub,   gQb1, uQb1, 4'i32, 6'i32, x6)
      fusedElem(14'i32 + sub,   gQb1, uQb1, 6'i32, 7'i32, x7)
      blkIdx = blkIdx + 1'i32
    gateAcc = gateAcc + hippoShflDown(gateAcc, 16)
    gateAcc = gateAcc + hippoShflDown(gateAcc, 8)
    gateAcc = gateAcc + hippoShflDown(gateAcc, 4)
    gateAcc = gateAcc + hippoShflDown(gateAcc, 2)
    gateAcc = gateAcc + hippoShflDown(gateAcc, 1)
    upAcc = upAcc + hippoShflDown(upAcc, 16)
    upAcc = upAcc + hippoShflDown(upAcc, 8)
    upAcc = upAcc + hippoShflDown(upAcc, 4)
    upAcc = upAcc + hippoShflDown(upAcc, 2)
    upAcc = upAcc + hippoShflDown(upAcc, 1)
    if tid == 0'i32:
      let g = gateAcc
      let sigmoid = 1.0'f32 / (1.0'f32 + expf(-g))
      outArr[row] = g * sigmoid * upAcc

proc gpuFusedGateUpSiluQ3K*(dst, x, gateQuant, upQuant: pointer,
                              wCols, wRows: int, stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoWarpSize.uint32)
    var gPtr = gateQuant; var uPtr = upQuant
    var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(fusedGateUpSiluQ3KWarpKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(gPtr, uPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    {.error: "gpuFusedGateUpSiluQ3K requires WarpSize == 32".}

# ---------------------------------------------------------------------------
# Fused K+V for Q4_K (block-based, 8 rows/block)
# ---------------------------------------------------------------------------
when HippoWarpSize == 32:
  proc fusedKVQ4KBlockKernel(
    kData, vData: ptr uint8,
    xData, kOut, vOut: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    let tid = cint(threadIdx.x)
    let warpId = tid shr 5'i32
    let lane = tid and 31'i32
    let baseRow = cint(blockIdx.x) * Q4KBlockRowsPerBlock
    let row = baseRow + warpId
    let kw = cast[ptr UncheckedArray[uint8]](kData)
    let vw = cast[ptr UncheckedArray[uint8]](vData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let kArr = cast[ptr UncheckedArray[float32]](kOut)
    let vArr = cast[ptr UncheckedArray[float32]](vOut)
    var sX {.hippoShared.}: array[2048, float32]
    var i = tid
    while i < wCols:
      sX[i] = xArr[i]
      i = i + 256'i32
    hippoSyncthreads()
    if row >= outRows:
      return
    let nBlocksPerRow = wCols div 256'i32
    let rowSizeBytes = nBlocksPerRow * 144'i32
    let kRowBase = row * rowSizeBytes
    let vRowBase = row * rowSizeBytes
    var kAcc = 0.0'f32
    var vAcc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let kbs = kRowBase + blkIdx * 144'i32
      let vbs = vRowBase + blkIdx * 144'i32
      let eb = blkIdx * 256'i32
      let kDRaw = uint16(kw[kbs]) or (uint16(kw[kbs + 1'i32]) shl 8)
      let kDmRaw = uint16(kw[kbs + 2'i32]) or (uint16(kw[kbs + 3'i32]) shl 8)
      let kDAll = hippoHalfToFloat(kDRaw)
      let kDMin = hippoHalfToFloat(kDmRaw)
      let kScales = cast[ptr UncheckedArray[uint8]](addr kw[kbs + 4'i32])
      let kQs = cast[ptr UncheckedArray[uint8]](addr kw[kbs + 16'i32])
      let vDRaw = uint16(vw[vbs]) or (uint16(vw[vbs + 1'i32]) shl 8)
      let vDmRaw = uint16(vw[vbs + 2'i32]) or (uint16(vw[vbs + 3'i32]) shl 8)
      let vDAll = hippoHalfToFloat(vDRaw)
      let vDMin = hippoHalfToFloat(vDmRaw)
      let vScales = cast[ptr UncheckedArray[uint8]](addr vw[vbs + 4'i32])
      let vQs = cast[ptr UncheckedArray[uint8]](addr vw[vbs + 16'i32])
      var isIdx = 0'i32
      var qOff = 0'i32
      for grp in countup(0'i32, 3'i32):
        var kSc0, kMn0, kSc1, kMn1: uint8
        getScaleMinK4Gpu(isIdx, kScales, kSc0, kMn0)
        let kD1 = kDAll * cfloat(kSc0)
        let kM1 = kDMin * cfloat(kMn0)
        getScaleMinK4Gpu(isIdx + 1'i32, kScales, kSc1, kMn1)
        let kD2 = kDAll * cfloat(kSc1)
        let kM2 = kDMin * cfloat(kMn1)
        var vSc0, vMn0, vSc1, vMn1: uint8
        getScaleMinK4Gpu(isIdx, vScales, vSc0, vMn0)
        let vD1 = vDAll * cfloat(vSc0)
        let vM1 = vDMin * cfloat(vMn0)
        getScaleMinK4Gpu(isIdx + 1'i32, vScales, vSc1, vMn1)
        let vD2 = vDAll * cfloat(vSc1)
        let vM2 = vDMin * cfloat(vMn1)
        let kQByte = kQs[qOff + lane]
        let vQByte = vQs[qOff + lane]
        let xLo = sX[eb + grp * 64'i32 + lane]
        let xHi = sX[eb + grp * 64'i32 + 32'i32 + lane]
        kAcc = kAcc + (kD1 * cfloat(kQByte and 0x0F'u8) - kM1) * xLo
        kAcc = kAcc + (kD2 * cfloat(kQByte shr 4'u8) - kM2) * xHi
        vAcc = vAcc + (vD1 * cfloat(vQByte and 0x0F'u8) - vM1) * xLo
        vAcc = vAcc + (vD2 * cfloat(vQByte shr 4'u8) - vM2) * xHi
        isIdx = isIdx + 2'i32
        qOff = qOff + 32'i32
      blkIdx = blkIdx + 1'i32
    kAcc = kAcc + hippoShflDown(kAcc, 16)
    kAcc = kAcc + hippoShflDown(kAcc, 8)
    kAcc = kAcc + hippoShflDown(kAcc, 4)
    kAcc = kAcc + hippoShflDown(kAcc, 2)
    kAcc = kAcc + hippoShflDown(kAcc, 1)
    vAcc = vAcc + hippoShflDown(vAcc, 16)
    vAcc = vAcc + hippoShflDown(vAcc, 8)
    vAcc = vAcc + hippoShflDown(vAcc, 4)
    vAcc = vAcc + hippoShflDown(vAcc, 2)
    vAcc = vAcc + hippoShflDown(vAcc, 1)
    if lane == 0'i32:
      kArr[row] = kAcc
      vArr[row] = vAcc

proc gpuFusedKVLinearQ4K*(kDst, vDst, x, kQuant, vQuant: pointer,
                            wCols, wRows: int, stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(((wRows + Q4KBlockRowsPerBlock - 1) div Q4KBlockRowsPerBlock).uint32)
    let blk = newDim3(HippoBlockSize.uint32)
    var kPtr = kQuant; var vPtr = vQuant
    var xPtr = x; var kDstP = kDst; var vDstP = vDst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(fusedKVQ4KBlockKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(kPtr, vPtr, xPtr, kDstP, vDstP, outRowsArg, wColsArg))
  else:
    {.error: "gpuFusedKVLinearQ4K requires WarpSize == 32".}

# ---------------------------------------------------------------------------
# Fused Gate+Up+SiLU for Q4_K (block-based, 8 rows/block)
# ---------------------------------------------------------------------------
when HippoWarpSize == 32:
  proc fusedGateUpSiluQ4KBlockKernel(
    gateData, upData: ptr uint8,
    xData, outData: ptr float32,
    outRows, wCols: cint
  ) {.hippoGlobal.} =
    let tid = cint(threadIdx.x)
    let warpId = tid shr 5'i32
    let lane = tid and 31'i32
    let baseRow = cint(blockIdx.x) * Q4KBlockRowsPerBlock
    let row = baseRow + warpId
    let gw = cast[ptr UncheckedArray[uint8]](gateData)
    let uw = cast[ptr UncheckedArray[uint8]](upData)
    let xArr = cast[ptr UncheckedArray[float32]](xData)
    let outArr = cast[ptr UncheckedArray[float32]](outData)
    var sX {.hippoShared.}: array[2048, float32]
    var i = tid
    while i < wCols:
      sX[i] = xArr[i]
      i = i + 256'i32
    hippoSyncthreads()
    if row >= outRows:
      return
    let nBlocksPerRow = wCols div 256'i32
    let rowSizeBytes = nBlocksPerRow * 144'i32
    let gRowBase = row * rowSizeBytes
    let uRowBase = row * rowSizeBytes
    var gateAcc = 0.0'f32
    var upAcc = 0.0'f32
    var blkIdx = 0'i32
    while blkIdx < nBlocksPerRow:
      let gbs = gRowBase + blkIdx * 144'i32
      let ubs = uRowBase + blkIdx * 144'i32
      let eb = blkIdx * 256'i32
      let gDRaw = uint16(gw[gbs]) or (uint16(gw[gbs + 1'i32]) shl 8)
      let gDmRaw = uint16(gw[gbs + 2'i32]) or (uint16(gw[gbs + 3'i32]) shl 8)
      let gDAll = hippoHalfToFloat(gDRaw)
      let gDMin = hippoHalfToFloat(gDmRaw)
      let gScales = cast[ptr UncheckedArray[uint8]](addr gw[gbs + 4'i32])
      let gQs = cast[ptr UncheckedArray[uint8]](addr gw[gbs + 16'i32])
      let uDRaw = uint16(uw[ubs]) or (uint16(uw[ubs + 1'i32]) shl 8)
      let uDmRaw = uint16(uw[ubs + 2'i32]) or (uint16(uw[ubs + 3'i32]) shl 8)
      let uDAll = hippoHalfToFloat(uDRaw)
      let uDMin = hippoHalfToFloat(uDmRaw)
      let uScales = cast[ptr UncheckedArray[uint8]](addr uw[ubs + 4'i32])
      let uQs = cast[ptr UncheckedArray[uint8]](addr uw[ubs + 16'i32])
      var isIdx = 0'i32
      var qOff = 0'i32
      for grp in countup(0'i32, 3'i32):
        var gSc0, gMn0, gSc1, gMn1: uint8
        getScaleMinK4Gpu(isIdx, gScales, gSc0, gMn0)
        let gD1 = gDAll * cfloat(gSc0)
        let gM1 = gDMin * cfloat(gMn0)
        getScaleMinK4Gpu(isIdx + 1'i32, gScales, gSc1, gMn1)
        let gD2 = gDAll * cfloat(gSc1)
        let gM2 = gDMin * cfloat(gMn1)
        var uSc0, uMn0, uSc1, uMn1: uint8
        getScaleMinK4Gpu(isIdx, uScales, uSc0, uMn0)
        let uD1 = uDAll * cfloat(uSc0)
        let uM1 = uDMin * cfloat(uMn0)
        getScaleMinK4Gpu(isIdx + 1'i32, uScales, uSc1, uMn1)
        let uD2 = uDAll * cfloat(uSc1)
        let uM2 = uDMin * cfloat(uMn1)
        let gQByte = gQs[qOff + lane]
        let uQByte = uQs[qOff + lane]
        let xLo = sX[eb + grp * 64'i32 + lane]
        let xHi = sX[eb + grp * 64'i32 + 32'i32 + lane]
        gateAcc = gateAcc + (gD1 * cfloat(gQByte and 0x0F'u8) - gM1) * xLo
        gateAcc = gateAcc + (gD2 * cfloat(gQByte shr 4'u8) - gM2) * xHi
        upAcc = upAcc + (uD1 * cfloat(uQByte and 0x0F'u8) - uM1) * xLo
        upAcc = upAcc + (uD2 * cfloat(uQByte shr 4'u8) - uM2) * xHi
        isIdx = isIdx + 2'i32
        qOff = qOff + 32'i32
      blkIdx = blkIdx + 1'i32
    gateAcc = gateAcc + hippoShflDown(gateAcc, 16)
    gateAcc = gateAcc + hippoShflDown(gateAcc, 8)
    gateAcc = gateAcc + hippoShflDown(gateAcc, 4)
    gateAcc = gateAcc + hippoShflDown(gateAcc, 2)
    gateAcc = gateAcc + hippoShflDown(gateAcc, 1)
    upAcc = upAcc + hippoShflDown(upAcc, 16)
    upAcc = upAcc + hippoShflDown(upAcc, 8)
    upAcc = upAcc + hippoShflDown(upAcc, 4)
    upAcc = upAcc + hippoShflDown(upAcc, 2)
    upAcc = upAcc + hippoShflDown(upAcc, 1)
    if lane == 0'i32:
      let g = gateAcc
      let sigmoid = 1.0'f32 / (1.0'f32 + expf(-g))
      outArr[row] = g * sigmoid * upAcc

proc gpuFusedGateUpSiluQ4K*(dst, x, gateQuant, upQuant: pointer,
                              wCols, wRows: int, stream: HippoStream) =
  when HippoWarpSize == 32:
    let grid = newDim3(((wRows + Q4KBlockRowsPerBlock - 1) div Q4KBlockRowsPerBlock).uint32)
    let blk = newDim3(HippoBlockSize.uint32)
    var gPtr = gateQuant; var uPtr = upQuant
    var xPtr = x; var dPtr = dst
    var outRowsArg = wRows.cint; var wColsArg = wCols.cint
    hippoLaunchKernel(fusedGateUpSiluQ4KBlockKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(gPtr, uPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    {.error: "gpuFusedGateUpSiluQ4K requires WarpSize == 32".}

# ---------------------------------------------------------------------------
# GPU argmax kernels — fixed 256-block grid so pass 2 fits in one block
# ---------------------------------------------------------------------------
proc argmaxPass1Kernel(
  data: ptr float32,
  scratchVals: ptr float32,
  scratchIdxs: ptr cint,
  n: cint
) {.hippoGlobal.} =
  var sVal {.hippoShared.}: array[HippoBlockSize, float32]
  var sIdx {.hippoShared.}: array[HippoBlockSize, cint]
  let tid = int(threadIdx.x)
  let d = cast[ptr UncheckedArray[float32]](data)
  let stride = int(gridDim.x) * HippoBlockSize
  var gid = int(blockIdx.x) * HippoBlockSize + tid
  var bestVal = -3.4e38'f32
  var bestIdx = -1'i32
  while gid < int(n):
    let v = d[gid]
    if v > bestVal:
      bestVal = v
      bestIdx = cint(gid)
    gid = gid + stride
  sVal[tid] = bestVal
  sIdx[tid] = bestIdx
  hippoSyncthreads()
  when HippoBlockSize == 256:
    if tid < 128:
      if sVal[tid + 128] > sVal[tid]:
        sVal[tid] = sVal[tid + 128]
        sIdx[tid] = sIdx[tid + 128]
    hippoSyncthreads()
    if tid < 64:
      if sVal[tid + 64] > sVal[tid]:
        sVal[tid] = sVal[tid + 64]
        sIdx[tid] = sIdx[tid + 64]
    hippoSyncthreads()
  when HippoWarpSize == 64:
    if tid < HippoWarpSize:
      var val = sVal[tid]
      var idx = sIdx[tid]
      var otherVal = hippoShflDown(val, 32)
      var otherIdx = hippoShflDown(idx, 32)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 16)
      otherIdx = hippoShflDown(idx, 16)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 8)
      otherIdx = hippoShflDown(idx, 8)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 4)
      otherIdx = hippoShflDown(idx, 4)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 2)
      otherIdx = hippoShflDown(idx, 2)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 1)
      otherIdx = hippoShflDown(idx, 1)
      if otherVal > val: val = otherVal; idx = otherIdx
      if tid == 0:
        let sv = cast[ptr UncheckedArray[float32]](scratchVals)
        let si = cast[ptr UncheckedArray[cint]](scratchIdxs)
        sv[int(blockIdx.x)] = val
        si[int(blockIdx.x)] = idx
  elif HippoWarpSize == 32:
    if tid < 32:
      if sVal[tid + 32] > sVal[tid]:
        sVal[tid] = sVal[tid + 32]
        sIdx[tid] = sIdx[tid + 32]
    hippoSyncthreads()
    if tid < HippoWarpSize:
      var val = sVal[tid]
      var idx = sIdx[tid]
      var otherVal = hippoShflDown(val, 16)
      var otherIdx = hippoShflDown(idx, 16)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 8)
      otherIdx = hippoShflDown(idx, 8)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 4)
      otherIdx = hippoShflDown(idx, 4)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 2)
      otherIdx = hippoShflDown(idx, 2)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 1)
      otherIdx = hippoShflDown(idx, 1)
      if otherVal > val: val = otherVal; idx = otherIdx
      if tid == 0:
        let sv = cast[ptr UncheckedArray[float32]](scratchVals)
        let si = cast[ptr UncheckedArray[cint]](scratchIdxs)
        sv[int(blockIdx.x)] = val
        si[int(blockIdx.x)] = idx

proc argmaxPass2Kernel(
  scratchVals: ptr float32,
  scratchIdxs: ptr cint,
  resultIdx: ptr cint,
  nBlocks: cint
) {.hippoGlobal.} =
  var sVal {.hippoShared.}: array[HippoBlockSize, float32]
  var sIdx {.hippoShared.}: array[HippoBlockSize, cint]
  let tid = int(threadIdx.x)
  let sv = cast[ptr UncheckedArray[float32]](scratchVals)
  let si = cast[ptr UncheckedArray[cint]](scratchIdxs)
  if tid < int(nBlocks):
    sVal[tid] = sv[tid]
    sIdx[tid] = si[tid]
  else:
    sVal[tid] = -3.4e38'f32
    sIdx[tid] = -1'i32
  hippoSyncthreads()
  when HippoBlockSize == 256:
    if tid < 128:
      if sVal[tid + 128] > sVal[tid]:
        sVal[tid] = sVal[tid + 128]
        sIdx[tid] = sIdx[tid + 128]
    hippoSyncthreads()
    if tid < 64:
      if sVal[tid + 64] > sVal[tid]:
        sVal[tid] = sVal[tid + 64]
        sIdx[tid] = sIdx[tid + 64]
    hippoSyncthreads()
  when HippoWarpSize == 64:
    if tid < HippoWarpSize:
      var val = sVal[tid]
      var idx = sIdx[tid]
      var otherVal = hippoShflDown(val, 32)
      var otherIdx = hippoShflDown(idx, 32)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 16)
      otherIdx = hippoShflDown(idx, 16)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 8)
      otherIdx = hippoShflDown(idx, 8)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 4)
      otherIdx = hippoShflDown(idx, 4)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 2)
      otherIdx = hippoShflDown(idx, 2)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 1)
      otherIdx = hippoShflDown(idx, 1)
      if otherVal > val: val = otherVal; idx = otherIdx
      if tid == 0:
        cast[ptr cint](resultIdx)[] = idx
  elif HippoWarpSize == 32:
    if tid < 32:
      if sVal[tid + 32] > sVal[tid]:
        sVal[tid] = sVal[tid + 32]
        sIdx[tid] = sIdx[tid + 32]
    hippoSyncthreads()
    if tid < HippoWarpSize:
      var val = sVal[tid]
      var idx = sIdx[tid]
      var otherVal = hippoShflDown(val, 16)
      var otherIdx = hippoShflDown(idx, 16)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 8)
      otherIdx = hippoShflDown(idx, 8)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 4)
      otherIdx = hippoShflDown(idx, 4)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 2)
      otherIdx = hippoShflDown(idx, 2)
      if otherVal > val: val = otherVal; idx = otherIdx
      otherVal = hippoShflDown(val, 1)
      otherIdx = hippoShflDown(idx, 1)
      if otherVal > val: val = otherVal; idx = otherIdx
      if tid == 0:
        cast[ptr cint](resultIdx)[] = idx

const ArgmaxGridBlocks = HippoBlockSize

proc ensureArgmaxBuffers() =
  if gpuCtx.argmaxScratch.sizeBytes < ArgmaxGridBlocks * (sizeof(float32) + sizeof(cint)):
    let nElems = ArgmaxGridBlocks * 2
    gpuCtx.argmaxScratch = newGpuTensor(@[nElems])
  if gpuCtx.argmaxResult.sizeBytes < sizeof(cint):
    let alloc = hippoMalloc(sizeof(cint))
    gpuCtx.argmaxResult = GpuTensor(
      devicePtr: alloc.p, alloc: alloc,
      shape: @[1], sizeBytes: sizeof(cint))

proc gpuArgmax*(logitsPtr: pointer, nVocab: int, stream: HippoStream): int32 =
  ensureArgmaxBuffers()
  let scratchVals = gpuCtx.argmaxScratch.devicePtr
  let scratchIdxs = cast[pointer](cast[uint](scratchVals) + uint(ArgmaxGridBlocks * sizeof(float32)))
  let resultPtr = gpuCtx.argmaxResult.devicePtr
  var grid1 = newDim3(ArgmaxGridBlocks.uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var lPtr = logitsPtr
  var svPtr = scratchVals; var siPtr = scratchIdxs
  var n = nVocab.cint
  hippoLaunchKernel(argmaxPass1Kernel, gridDim = grid1, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(lPtr, svPtr, siPtr, n))
  var grid2 = newDim3(1'u32)
  var rPtr = resultPtr
  var nb = ArgmaxGridBlocks.cint
  hippoLaunchKernel(argmaxPass2Kernel, gridDim = grid2, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(svPtr, siPtr, rPtr, nb))
  var hostResult: cint
  gpuDownloadFromDevice(addr hostResult, resultPtr, sizeof(cint), stream)
  gpuStreamSync(stream)
  return int32(hostResult)

# ---------------------------------------------------------------------------
# Elementwise kernels
# ---------------------------------------------------------------------------
proc addKernel(aData, bData, outData: ptr float32, n: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  if idx < int(n):
    let a = cast[ptr UncheckedArray[float32]](aData)
    let b = cast[ptr UncheckedArray[float32]](bData)
    let o = cast[ptr UncheckedArray[float32]](outData)
    o[idx] = a[idx] + b[idx]

proc gpuAdd*(dst: pointer, a, b: pointer, nElems: int, stream: HippoStream) =
  let grid = newDim3(((nElems + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var aPtr = a; var bPtr = b; var dPtr = dst; var n = nElems.cint
  hippoLaunchKernel(addKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(aPtr, bPtr, dPtr, n))

# Gather column t from column-major matrix [dim, seqLen] into contiguous dst[dim]
proc gatherColKernel(dstData, srcData: ptr float32, dim, seqLen, col: cint) {.hippoGlobal.} =
  let row = int(blockIdx.x * blockDim.x + threadIdx.x)
  if row < int(dim):
    let s = cast[ptr UncheckedArray[float32]](srcData)
    let d = cast[ptr UncheckedArray[float32]](dstData)
    d[row] = s[row * int(seqLen) + int(col)]

proc gpuGatherCol*(dst, src: pointer, dim, seqLen, col: int, stream: HippoStream) =
  let grid = newDim3(((dim + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var dPtr = dst; var sPtr = src
  var dimArg = dim.cint; var slArg = seqLen.cint; var cArg = col.cint
  hippoLaunchKernel(gatherColKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(dPtr, sPtr, dimArg, slArg, cArg))

# Scatter-add contiguous src[dim] into column t of column-major matrix [dim, seqLen]
proc scatterAddColKernel(dstData, srcData: ptr float32, dim, seqLen, col: cint) {.hippoGlobal.} =
  let row = int(blockIdx.x * blockDim.x + threadIdx.x)
  if row < int(dim):
    let d = cast[ptr UncheckedArray[float32]](dstData)
    let s = cast[ptr UncheckedArray[float32]](srcData)
    d[row * int(seqLen) + int(col)] = d[row * int(seqLen) + int(col)] + s[row]

proc gpuScatterAddCol*(dst, src: pointer, dim, seqLen, col: int, stream: HippoStream) =
  let grid = newDim3(((dim + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var dPtr = dst; var sPtr = src
  var dimArg = dim.cint; var slArg = seqLen.cint; var cArg = col.cint
  hippoLaunchKernel(scatterAddColKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(dPtr, sPtr, dimArg, slArg, cArg))

proc siluMulKernel(gateData, upData, outData: ptr float32, n: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  if idx < int(n):
    let g = cast[ptr UncheckedArray[float32]](gateData)
    let u = cast[ptr UncheckedArray[float32]](upData)
    let o = cast[ptr UncheckedArray[float32]](outData)
    let x = g[idx]
    let sigmoid = 1.0'f32 / (1.0'f32 + expf(-x))
    o[idx] = x * sigmoid * u[idx]

proc gpuSiluMul*(dst: pointer, gate, up: pointer, nElems: int, stream: HippoStream) =
  let grid = newDim3(((nElems + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var gPtr = gate; var uPtr = up; var dPtr = dst; var n = nElems.cint
  hippoLaunchKernel(siluMulKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(gPtr, uPtr, dPtr, n))

# ---------------------------------------------------------------------------
# GELU activation kernel
# ---------------------------------------------------------------------------
proc geluKernel(data: ptr float32, n: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  if idx < int(n):
    let d = cast[ptr UncheckedArray[float32]](data)
    let x = d[idx]
    let c = 0.7978845608'f32  # sqrt(2/pi)
    let inner = c * (x + 0.044715'f32 * x * x * x)
    # tanh(x) = 1 - 2/(exp(2x)+1)
    let e2x = expf(2.0'f32 * inner)
    let th = 1.0'f32 - 2.0'f32 / (e2x + 1.0'f32)
    d[idx] = 0.5'f32 * x * (1.0'f32 + th)

proc gpuGelu*(x: pointer, nElems: int, stream: HippoStream) =
  let grid = newDim3(((nElems + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var xPtr = x; var n = nElems.cint
  hippoLaunchKernel(geluKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(xPtr, n))

# ---------------------------------------------------------------------------
# ReLU² activation kernel (relu(x) then square)
# ---------------------------------------------------------------------------
proc reluSqrKernel(data: ptr float32, n: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  if idx < int(n):
    let d = cast[ptr UncheckedArray[float32]](data)
    let x = d[idx]
    let r = if x > 0.0'f32: x else: 0.0'f32
    d[idx] = r * r

proc gpuReluSqr*(x: pointer, nElems: int, stream: HippoStream) =
  let grid = newDim3(((nElems + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var xPtr = x; var n = nElems.cint
  hippoLaunchKernel(reluSqrKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(xPtr, n))

# ---------------------------------------------------------------------------
# SiLU (in-place) kernel
# ---------------------------------------------------------------------------
proc siluKernel(data: ptr float32, n: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  if idx < int(n):
    let d = cast[ptr UncheckedArray[float32]](data)
    let x = d[idx]
    d[idx] = x / (1.0'f32 + expf(-x))

proc gpuSilu*(x: pointer, nElems: int, stream: HippoStream) =
  let grid = newDim3(((nElems + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var xPtr = x; var n = nElems.cint
  hippoLaunchKernel(siluKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(xPtr, n))

# ---------------------------------------------------------------------------
# Elementwise multiply kernel
# ---------------------------------------------------------------------------
proc elemMulKernel(aData, bData: ptr float32, n: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  if idx < int(n):
    let a = cast[ptr UncheckedArray[float32]](aData)
    let b = cast[ptr UncheckedArray[float32]](bData)
    a[idx] = a[idx] * b[idx]

proc gpuElemMul*(a, b: pointer, nElems: int, stream: HippoStream) =
  let grid = newDim3(((nElems + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var aPtr = a; var bPtr = b; var n = nElems.cint
  hippoLaunchKernel(elemMulKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(aPtr, bPtr, n))

# ---------------------------------------------------------------------------
# MoE utility kernels
# ---------------------------------------------------------------------------
proc scaleAddKernel(dstData, srcData: ptr float32, scale: float32, n: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  if idx < int(n):
    let d = cast[ptr UncheckedArray[float32]](dstData)
    let s = cast[ptr UncheckedArray[float32]](srcData)
    d[idx] = d[idx] + scale * s[idx]

proc gpuScaleAdd*(dst, src: pointer, scale: float32, nElems: int, stream: HippoStream) =
  let grid = newDim3(((nElems + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var dPtr = dst; var sPtr = src; var sc = scale; var n = nElems.cint
  hippoLaunchKernel(scaleAddKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(dPtr, sPtr, sc, n))

proc attnGateKernel(dstData, gateData: ptr float32, n: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  if idx < int(n):
    let d = cast[ptr UncheckedArray[float32]](dstData)
    let g = cast[ptr UncheckedArray[float32]](gateData)
    let sig = 1.0'f32 / (1.0'f32 + expf(-g[idx]))
    d[idx] = sig * d[idx]

proc gpuAttnGate*(dst, gate: pointer, nElems: int, stream: HippoStream) =
  let grid = newDim3(((nElems + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var dPtr = dst; var gPtr = gate; var n = nElems.cint
  hippoLaunchKernel(attnGateKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(dPtr, gPtr, n))

proc sharedExpertGateKernel(outputData, inputData, gateWeightData: ptr float32,
                             nElems: cint) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let tid = int(threadIdx.x)
  let g = cast[ptr UncheckedArray[float32]](gateWeightData)
  let x = cast[ptr UncheckedArray[float32]](inputData)
  let o = cast[ptr UncheckedArray[float32]](outputData)
  var acc = 0.0'f32
  var i = tid
  while i < int(nElems):
    acc = acc + g[i] * x[i]
    i = i + int(blockDim.x)
  sdata[tid] = acc
  hippoSyncthreads()
  reduceSum256(sdata, tid)
  var sigVal {.hippoShared.}: array[1, float32]
  if tid == 0:
    sigVal[0] = 1.0'f32 / (1.0'f32 + expf(-sdata[0]))
  hippoSyncthreads()
  let sv = sigVal[0]
  i = tid
  while i < int(nElems):
    o[i] = o[i] * sv
    i = i + int(blockDim.x)

proc gpuSharedExpertGate*(output, input, gateWeight: pointer,
                           nElems: int, stream: HippoStream) =
  let grid = newDim3(1'u32)
  let blk = newDim3(HippoBlockSize.uint32)
  var oPtr = output; var iPtr = input; var gPtr = gateWeight; var n = nElems.cint
  hippoLaunchKernel(sharedExpertGateKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(oPtr, iPtr, gPtr, n))

proc moeTopKKernel(logitsData: ptr float32,
                    expertIndicesData: ptr int32,
                    expertWeightsData: ptr float32,
                    nExperts, K: cint) {.hippoGlobal.} =
  let tid = int(threadIdx.x)
  let logits = cast[ptr UncheckedArray[float32]](logitsData)
  let indices = cast[ptr UncheckedArray[int32]](expertIndicesData)
  let weights = cast[ptr UncheckedArray[float32]](expertWeightsData)
  if tid != 0: return
  # Softmax over all experts first
  var maxVal = logits[0]
  for e in 1 ..< int(nExperts):
    if logits[e] > maxVal: maxVal = logits[e]
  var sumExp = 0.0'f32
  for e in 0 ..< int(nExperts):
    logits[e] = expf(logits[e] - maxVal)
    sumExp = sumExp + logits[e]
  let invSum = 1.0'f32 / sumExp
  for e in 0 ..< int(nExperts):
    logits[e] = logits[e] * invSum
  # Top-K from softmaxed probabilities
  for k in 0 ..< int(K):
    var bestVal = -1e30'f32
    var bestIdx: int32 = 0
    for e in 0 ..< int(nExperts):
      var already = false
      for j in 0 ..< k:
        if indices[j] == int32(e):
          already = true
          break
      if not already and logits[e] > bestVal:
        bestVal = logits[e]
        bestIdx = int32(e)
    indices[k] = bestIdx
    weights[k] = bestVal
  # Renormalize selected weights to sum to 1
  var topSum = 0.0'f32
  for k in 0 ..< int(K):
    topSum = topSum + weights[k]
  let invTopSum = 1.0'f32 / topSum
  for k in 0 ..< int(K):
    weights[k] = weights[k] * invTopSum

proc gpuMoeTopK*(expertIndices, expertWeights, logits: pointer,
                  nExperts, K: int, stream: HippoStream) =
  let grid = newDim3(1'u32)
  let blk = newDim3(1'u32)
  var lPtr = logits; var iPtr = expertIndices; var wPtr = expertWeights
  var ne = nExperts.cint; var k = K.cint
  hippoLaunchKernel(moeTopKKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(lPtr, iPtr, wPtr, ne, k))

proc zeroBufferKernel(data: ptr float32, n: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  if idx < int(n):
    cast[ptr UncheckedArray[float32]](data)[idx] = 0.0'f32

proc gpuZeroBuffer*(dst: pointer, nElems: int, stream: HippoStream) =
  let grid = newDim3(((nElems + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var dPtr = dst; var n = nElems.cint
  hippoLaunchKernel(zeroBufferKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(dPtr, n))

# ---------------------------------------------------------------------------
# De-interleave Q+Gate from fused projection
# Input: [Q0(headDim), G0(headDim), Q1(headDim), G1(headDim), ...]
# Output: q[nHead*headDim], gate[nHead*headDim]
# ---------------------------------------------------------------------------
proc deinterleaveQGateKernel(qOut, gateOut, fused: ptr float32,
                              nHead, headDim: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  let total = int(nHead) * int(headDim)
  if idx >= total: return
  let q = cast[ptr UncheckedArray[float32]](qOut)
  let g = cast[ptr UncheckedArray[float32]](gateOut)
  let f = cast[ptr UncheckedArray[float32]](fused)
  let h = idx div int(headDim)
  let d = idx mod int(headDim)
  q[idx] = f[h * int(headDim) * 2 + d]
  g[idx] = f[h * int(headDim) * 2 + int(headDim) + d]

proc gpuDeinterleaveQGate*(q, gate, fused: pointer, nHead, headDim: int,
                            stream: HippoStream) =
  let total = nHead * headDim
  let grid = newDim3(((total + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var qPtr = q; var gPtr = gate; var fPtr = fused
  var nh = nHead.cint; var hd = headDim.cint
  hippoLaunchKernel(deinterleaveQGateKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(qPtr, gPtr, fPtr, nh, hd))

proc deinterleaveQGatePrefillKernel(qOut, gateOut, fused: ptr float32,
                                     nHead, headDim, seqLen: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  let qDim = int(nHead) * int(headDim)
  let total = qDim * int(seqLen)
  if idx >= total: return
  let q = cast[ptr UncheckedArray[float32]](qOut)
  let g = cast[ptr UncheckedArray[float32]](gateOut)
  let f = cast[ptr UncheckedArray[float32]](fused)
  let elem = idx div int(seqLen)
  let t = idx mod int(seqLen)
  let h = elem div int(headDim)
  let d = elem mod int(headDim)
  let fusedRow = h * int(headDim) * 2 + d
  let fusedGateRow = h * int(headDim) * 2 + int(headDim) + d
  q[elem * int(seqLen) + t] = f[fusedRow * int(seqLen) + t]
  g[elem * int(seqLen) + t] = f[fusedGateRow * int(seqLen) + t]

proc gpuDeinterleaveQGatePrefill*(q, gate, fused: pointer, nHead, headDim, seqLen: int,
                                   stream: HippoStream) =
  let total = nHead * headDim * seqLen
  let grid = newDim3(((total + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var qPtr = q; var gPtr = gate; var fPtr = fused
  var nh = nHead.cint; var hd = headDim.cint; var sl = seqLen.cint
  hippoLaunchKernel(deinterleaveQGatePrefillKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(qPtr, gPtr, fPtr, nh, hd, sl))

# ---------------------------------------------------------------------------
# L2 Norm per head (for Delta Net Q, K normalization)
# ---------------------------------------------------------------------------
proc l2NormPerHeadKernel(data: ptr float32, nHeads, headDim: cint) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let h = int(blockIdx.x)
  let tid = int(threadIdx.x)
  if h >= int(nHeads): return
  let d = cast[ptr UncheckedArray[float32]](data)
  let base = h * int(headDim)
  var sumSq = 0.0'f32
  var i = tid
  while i < int(headDim):
    let v = d[base + i]
    sumSq = sumSq + v * v
    i = i + int(blockDim.x)
  sdata[tid] = sumSq
  hippoSyncthreads()
  reduceSum256(sdata, tid)
  let invNorm = 1.0'f32 / sqrtf(sdata[0] + 1e-6'f32)
  i = tid
  while i < int(headDim):
    d[base + i] = d[base + i] * invNorm
    i = i + int(blockDim.x)

proc gpuL2NormPerHead*(data: pointer, nHeads, headDim: int, stream: HippoStream) =
  let grid = newDim3(nHeads.uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var dPtr = data; var nh = nHeads.cint; var hd = headDim.cint
  hippoLaunchKernel(l2NormPerHeadKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(dPtr, nh, hd))

# ---------------------------------------------------------------------------
# Delta Net single-step recurrence (Gated Delta Net for Qwen3.6 SSM)
# ---------------------------------------------------------------------------
proc deltaNetDecodeKernel(
  stateData: ptr float32,    # [nVHeads, headKDim, headVDim] recurrent state
  qData: ptr float32,        # [nVHeads, headKDim] after L2 norm + head expansion
  kData: ptr float32,        # [nVHeads, headKDim]
  vData: ptr float32,        # [nVHeads, headVDim]
  gateData: ptr float32,     # [nVHeads] decay gate (negative → decay < 1)
  betaData: ptr float32,     # [nVHeads] damping (sigmoid output)
  outputData: ptr float32,   # [nVHeads, headVDim] output
  nVHeads, headKDim, headVDim: cint,
  qScale: float32
) {.hippoGlobal.} =
  # One block per head
  let h = int(blockIdx.x)
  let tid = int(threadIdx.x)
  if h >= int(nVHeads): return
  let S = cast[ptr UncheckedArray[float32]](stateData)
  let q = cast[ptr UncheckedArray[float32]](qData)
  let k = cast[ptr UncheckedArray[float32]](kData)
  let v = cast[ptr UncheckedArray[float32]](vData)
  let gate = cast[ptr UncheckedArray[float32]](gateData)
  let beta = cast[ptr UncheckedArray[float32]](betaData)
  let o = cast[ptr UncheckedArray[float32]](outputData)
  let kd = int(headKDim)
  let vd = int(headVDim)
  let stateBase = h * kd * vd
  let decay = expf(gate[h])
  let b = beta[h]
  # Delta Rule recurrence (per vIdx, parallelized across threads):
  # 1. Decay: S *= decay
  # 2. Retrieve: kvMem = sum_k S[k,v] * K[k]
  # 3. Delta: delta = beta * (V[v] - kvMem)
  # 4. Write: S[k,v] += K[k] * delta
  # 5. Read: output[v] = sum_k Q[k] * S[k,v]
  var vIdx = tid
  while vIdx < vd:
    let vVal = v[h * vd + vIdx]
    var kvMem = 0.0'f32
    for kIdx in 0 ..< kd:
      let sIdx = stateBase + kIdx * vd + vIdx
      let decayed = decay * S[sIdx]
      S[sIdx] = decayed
      kvMem = kvMem + decayed * k[h * kd + kIdx]
    let delta = b * (vVal - kvMem)
    var outVal = 0.0'f32
    for kIdx in 0 ..< kd:
      let sIdx = stateBase + kIdx * vd + vIdx
      let newS = S[sIdx] + k[h * kd + kIdx] * delta
      S[sIdx] = newS
      outVal = outVal + q[h * kd + kIdx] * newS
    o[h * vd + vIdx] = qScale * outVal
    vIdx = vIdx + int(blockDim.x)

proc gpuDeltaNetDecode*(state, q, k, v, gate, beta, output: pointer,
                         nVHeads, headKDim, headVDim: int,
                         stream: HippoStream) =
  let qScale = 1.0'f32 / sqrtf(float32(headKDim))
  let grid = newDim3(nVHeads.uint32)
  let blk = newDim3(min(headVDim, HippoBlockSize).uint32)
  var sPtr = state; var qPtr = q; var kPtr = k; var vPtr = v
  var gPtr = gate; var bPtr = beta; var oPtr = output
  var nvh = nVHeads.cint; var hkd = headKDim.cint; var hvd = headVDim.cint
  var qs = qScale
  hippoLaunchKernel(deltaNetDecodeKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(sPtr, qPtr, kPtr, vPtr, gPtr, bPtr, oPtr, nvh, hkd, hvd, qs))

# ---------------------------------------------------------------------------
# Expand K heads to V heads (repeat interleave for GDN)
# ---------------------------------------------------------------------------
proc expandHeadsKernel(dst, src: ptr float32,
                        nKHeads, nVHeads, headDim: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  let total = int(nVHeads) * int(headDim)
  if idx >= total: return
  let d = cast[ptr UncheckedArray[float32]](dst)
  let s = cast[ptr UncheckedArray[float32]](src)
  let vHead = idx div int(headDim)
  let dim = idx mod int(headDim)
  let kHead = vHead mod int(nKHeads)
  d[idx] = s[kHead * int(headDim) + dim]

proc gpuExpandHeads*(dst, src: pointer, nKHeads, nVHeads, headDim: int,
                      stream: HippoStream) =
  let total = nVHeads * headDim
  let grid = newDim3(((total + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var dPtr = dst; var sPtr = src
  var nk = nKHeads.cint; var nv = nVHeads.cint; var hd = headDim.cint
  hippoLaunchKernel(expandHeadsKernel, gridDim = grid, blockDim = blk,
                    stream = stream, args = hippoArgs(dPtr, sPtr, nk, nv, hd))

# ---------------------------------------------------------------------------
# Compute Delta Net gate: gate[h] = softplus(alpha[h] + dt_bias[h]) * A[h]
# and beta: beta[h] = sigmoid(beta_raw[h])
# ---------------------------------------------------------------------------
proc deltaNetGateKernel(gateOut, betaOut: ptr float32,
                         alphaData, dtBiasData, aData, betaRawData: ptr float32,
                         nHeads: cint) {.hippoGlobal.} =
  let h = int(blockIdx.x * blockDim.x + threadIdx.x)
  if h >= int(nHeads): return
  let g = cast[ptr UncheckedArray[float32]](gateOut)
  let bo = cast[ptr UncheckedArray[float32]](betaOut)
  let alpha = cast[ptr UncheckedArray[float32]](alphaData)
  let dtBias = cast[ptr UncheckedArray[float32]](dtBiasData)
  let a = cast[ptr UncheckedArray[float32]](aData)
  let betaRaw = cast[ptr UncheckedArray[float32]](betaRawData)
  let x = alpha[h] + dtBias[h]
  let sp = if x > 20.0'f32: x else: logf(1.0'f32 + expf(x))
  g[h] = a[h] * sp
  bo[h] = 1.0'f32 / (1.0'f32 + expf(-betaRaw[h]))

proc gpuDeltaNetGate*(gateOut, betaOut, alpha, dtBias, a, betaRaw: pointer,
                       nHeads: int, stream: HippoStream) =
  let grid = newDim3(((nHeads + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(min(nHeads, HippoBlockSize).uint32)
  var goPtr = gateOut; var boPtr = betaOut; var alPtr = alpha
  var dtPtr = dtBias; var aPtr = a; var brPtr = betaRaw
  var nh = nHeads.cint
  hippoLaunchKernel(deltaNetGateKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(goPtr, boPtr, alPtr, dtPtr, aPtr, brPtr, nh))

# ---------------------------------------------------------------------------
# Gated RMS norm with SiLU: out[i] = rms_norm(x[i]) * silu(z[i])
# Per head: normalize each headVDim slice, then multiply by silu(z)
# ---------------------------------------------------------------------------
proc gatedRmsNormKernel(outData, xData, zData, normW: ptr float32,
                         nHeads, headDim: cint, eps: float32) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let h = int(blockIdx.x)
  let tid = int(threadIdx.x)
  if h >= int(nHeads): return
  let o = cast[ptr UncheckedArray[float32]](outData)
  let x = cast[ptr UncheckedArray[float32]](xData)
  let z = cast[ptr UncheckedArray[float32]](zData)
  let w = cast[ptr UncheckedArray[float32]](normW)
  let base = h * int(headDim)
  var sumSq = 0.0'f32
  var i = tid
  while i < int(headDim):
    let v = x[base + i]
    sumSq = sumSq + v * v
    i = i + int(blockDim.x)
  sdata[tid] = sumSq
  hippoSyncthreads()
  reduceSum256(sdata, tid)
  let rms = 1.0'f32 / sqrtf(sdata[0] / float32(headDim) + eps)
  i = tid
  while i < int(headDim):
    let zVal = z[base + i]
    let siluZ = zVal / (1.0'f32 + expf(-zVal))
    o[base + i] = x[base + i] * rms * w[i] * siluZ
    i = i + int(blockDim.x)

proc gpuGatedRmsNorm*(output, x, z, normWeight: pointer,
                       nHeads, headDim: int, eps: float32,
                       stream: HippoStream) =
  let grid = newDim3(nHeads.uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var oPtr = output; var xPtr = x; var zPtr = z; var wPtr = normWeight
  var nh = nHeads.cint; var hd = headDim.cint; var e = eps
  hippoLaunchKernel(gatedRmsNormKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(oPtr, xPtr, zPtr, wPtr, nh, hd, e))

# ---------------------------------------------------------------------------
# Group RMSNorm kernel (for SSM output)
# ---------------------------------------------------------------------------
proc groupRmsNormKernel(
  data, weightData: ptr float32,
  nGroups, groupSize: cint, eps: float32
) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let g = int(blockIdx.x)
  let tid = int(threadIdx.x)
  if g >= int(nGroups): return
  let d = cast[ptr UncheckedArray[float32]](data)
  let w = cast[ptr UncheckedArray[float32]](weightData)
  let base = g * int(groupSize)
  var sumSq = 0.0'f32
  var i = tid
  while i < int(groupSize):
    let v = d[base + i]
    sumSq = sumSq + v * v
    i = i + int(blockDim.x)
  sdata[tid] = sumSq
  hippoSyncthreads()
  reduceSum256(sdata, tid)
  let rms = 1.0'f32 / sqrtf(sdata[0] / float32(groupSize) + eps)
  i = tid
  while i < int(groupSize):
    d[base + i] = d[base + i] * rms * w[base + i]
    i = i + int(blockDim.x)

proc gpuGroupRmsNorm*(x, weight: pointer, nGroups, groupSize: int,
                       eps: float32, stream: HippoStream) =
  let grid = newDim3(nGroups.uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var xPtr = x; var wPtr = weight
  var nGroupsArg = nGroups.cint; var groupSizeArg = groupSize.cint; var epsArg = eps
  hippoLaunchKernel(groupRmsNormKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(xPtr, wPtr, nGroupsArg, groupSizeArg, epsArg))

# ---------------------------------------------------------------------------
# Mamba2 Conv1d decode kernel
# ---------------------------------------------------------------------------
proc conv1dDecodeKernel2(
  convStateData, inputData, weightData, biasData, outputData: ptr float32,
  nChannels, kernelSize: cint
) {.hippoGlobal.} =
  let ch = int(blockIdx.x * blockDim.x + threadIdx.x)
  if ch >= int(nChannels): return
  let cs = cast[ptr UncheckedArray[float32]](convStateData)
  let inp = cast[ptr UncheckedArray[float32]](inputData)
  let w = cast[ptr UncheckedArray[float32]](weightData)
  let b = cast[ptr UncheckedArray[float32]](biasData)
  let o = cast[ptr UncheckedArray[float32]](outputData)
  let convLen = int(kernelSize) - 1
  let base = ch * convLen
  # Compute conv output: weight layout [nChannels, kernelSize] (GGUF ne[0]=kernelSize)
  let wBase = ch * int(kernelSize)
  var acc = if biasData != nil: b[ch] else: 0.0'f32
  for i in 0 ..< convLen:
    acc = acc + cs[base + i] * w[wBase + i]
  acc = acc + inp[ch] * w[wBase + convLen]
  o[ch] = acc
  # Update state: shift left, insert input at end
  for i in 0 ..< convLen - 1:
    cs[base + i] = cs[base + i + 1]
  cs[base + convLen - 1] = inp[ch]

proc gpuConv1dDecode*(convState, input, weight, bias, output: pointer,
                       nChannels, kernelSize: int, stream: HippoStream) =
  let grid = newDim3(((nChannels + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var csPtr = convState; var inpPtr = input; var wPtr = weight
  var bPtr = bias; var oPtr = output
  var nChArg = nChannels.cint; var ksArg = kernelSize.cint
  hippoLaunchKernel(conv1dDecodeKernel2, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(csPtr, inpPtr, wPtr, bPtr, oPtr, nChArg, ksArg))

# ---------------------------------------------------------------------------
# Mamba2 selective scan decode kernel (single timestep)
# ---------------------------------------------------------------------------
# State layout: [nHeads, stateSize, headDim] — one block per head
proc ssmScanDecodeKernel(
  recStateData: ptr float32,      # [nHeads, stateSize, headDim]
  xData: ptr float32,             # [ssmInnerSize] (activated x after conv+silu)
  bData: ptr float32,             # [nGroups * stateSize]
  cData: ptr float32,             # [nGroups * stateSize]
  dtRawData: ptr float32,         # [nHeads]
  dtBiasData: ptr float32,        # [nHeads]
  aData: ptr float32,             # [nHeads]
  dData: ptr float32,             # [nHeads]
  yData: ptr float32,             # [ssmInnerSize] output
  nHeads, headsPerGroup, headDim, stateSize: cint
) {.hippoGlobal.} =
  let h = int(blockIdx.x)
  let tid = int(threadIdx.x)
  if h >= int(nHeads): return
  let rec = cast[ptr UncheckedArray[float32]](recStateData)
  let x = cast[ptr UncheckedArray[float32]](xData)
  let bArr = cast[ptr UncheckedArray[float32]](bData)
  let cArr = cast[ptr UncheckedArray[float32]](cData)
  let dtRaw = cast[ptr UncheckedArray[float32]](dtRawData)
  let dtBias = cast[ptr UncheckedArray[float32]](dtBiasData)
  let aArr = cast[ptr UncheckedArray[float32]](aData)
  let dArr = cast[ptr UncheckedArray[float32]](dData)
  let y = cast[ptr UncheckedArray[float32]](yData)
  let g = h div int(headsPerGroup)
  # dt = softplus(dt_raw + dt_bias)
  let dtVal = dtRaw[h] + dtBias[h]
  let dt = if dtVal > 20.0'f32: dtVal else: logf(1.0'f32 + expf(dtVal))
  let aBar = expf(aArr[h] * dt)
  let dVal = dArr[h]
  let xBase = h * int(headDim)
  let bBase = g * int(stateSize)
  let cBase = g * int(stateSize)
  let stBase = h * int(stateSize) * int(headDim)
  # For each headDim element (parallelized across threads)
  var d = tid
  while d < int(headDim):
    let xVal = x[xBase + d]
    let xDt = xVal * dt
    var yAcc = dVal * xVal
    for s in 0 ..< int(stateSize):
      let idx = stBase + s * int(headDim) + d
      let newState = aBar * rec[idx] + bArr[bBase + s] * xDt
      rec[idx] = newState
      yAcc = yAcc + cArr[cBase + s] * newState
    y[xBase + d] = yAcc
    d = d + int(blockDim.x)

proc gpuSsmScanDecode*(recState, x, b, c, dtRaw, dtBias, a, dParam, y: pointer,
                        nHeads, headsPerGroup, headDim, stateSize: int,
                        stream: HippoStream) =
  let grid = newDim3(nHeads.uint32)
  let blk = newDim3(min(headDim, HippoBlockSize).uint32)
  var rsPtr = recState; var xPtr = x; var bPtr = b; var cPtr = c
  var dtPtr = dtRaw; var dtbPtr = dtBias; var aPtr = a; var dPtr = dParam; var yPtr = y
  var nHArg = nHeads.cint; var hpgArg = headsPerGroup.cint
  var hdArg = headDim.cint; var ssArg = stateSize.cint
  hippoLaunchKernel(ssmScanDecodeKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(rsPtr, xPtr, bPtr, cPtr, dtPtr, dtbPtr,
                                     aPtr, dPtr, yPtr, nHArg, hpgArg, hdArg, ssArg))

# ---------------------------------------------------------------------------
# RMSNorm kernels
# ---------------------------------------------------------------------------
proc rmsnormColsKernel(
  xData, weightData, outData: ptr float32,
  dim, seqLen: cint, eps: float32
) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let col = int(blockIdx.x)
  let tid = int(threadIdx.x)
  if col >= int(seqLen):
    return
  let x = cast[ptr UncheckedArray[float32]](xData)
  let w = cast[ptr UncheckedArray[float32]](weightData)
  let o = cast[ptr UncheckedArray[float32]](outData)
  var ss = 0.0'f32
  var r = tid
  while r < int(dim):
    let v = x[r * int(seqLen) + col]
    ss = ss + v * v
    r = r + int(blockDim.x)
  sdata[tid] = ss
  hippoSyncthreads()
  reduceSum256(sdata, tid)
  var inv {.hippoShared.}: array[1, float32]
  if tid == 0:
    inv[0] = 1.0'f32 / sqrtf(sdata[0] / cfloat(dim) + cfloat(eps))
  hippoSyncthreads()
  let invVal = inv[0]
  r = tid
  while r < int(dim):
    let idx = r * int(seqLen) + col
    o[idx] = x[idx] * invVal * w[r]
    r = r + int(blockDim.x)

proc gpuRmsnormCols*(dst: pointer, x, weight: pointer, dim, seqLen: int,
                      eps: float32, stream: HippoStream) =
  let grid = newDim3(seqLen.uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var xPtr = x; var wPtr = weight; var dPtr = dst
  var dimArg = dim.cint; var seqLenArg = seqLen.cint; var epsArg = eps
  hippoLaunchKernel(rmsnormColsKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(xPtr, wPtr, dPtr, dimArg, seqLenArg, epsArg))

proc residualRmsnormKernel(
  xData, residualData, weightData, outData: ptr float32,
  dim: cint, eps: float32
) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let tid = int(threadIdx.x)
  let x = cast[ptr UncheckedArray[float32]](xData)
  let res = cast[ptr UncheckedArray[float32]](residualData)
  let w = cast[ptr UncheckedArray[float32]](weightData)
  let o = cast[ptr UncheckedArray[float32]](outData)
  var ss = 0.0'f32
  var r = tid
  while r < int(dim):
    let v = x[r] + res[r]
    x[r] = v
    ss = ss + v * v
    r = r + int(blockDim.x)
  sdata[tid] = ss
  hippoSyncthreads()
  reduceSum256(sdata, tid)
  var inv {.hippoShared.}: array[1, float32]
  if tid == 0:
    inv[0] = 1.0'f32 / sqrtf(sdata[0] / cfloat(dim) + cfloat(eps))
  hippoSyncthreads()
  let invVal = inv[0]
  r = tid
  while r < int(dim):
    o[r] = x[r] * invVal * w[r]
    r = r + int(blockDim.x)

proc gpuResidualRmsnorm*(normOut, x, residual, weight: pointer, dim: int,
                          eps: float32, stream: HippoStream) =
  let grid = newDim3(1'u32)
  let blk = newDim3(HippoBlockSize.uint32)
  var xPtr = x; var resPtr = residual; var wPtr = weight; var dPtr = normOut
  var dimArg = dim.cint; var epsArg = eps
  hippoLaunchKernel(residualRmsnormKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(xPtr, resPtr, wPtr, dPtr, dimArg, epsArg))

# ---------------------------------------------------------------------------
# QK normalization (per-head RMSNorm for Qwen3-style models)
# ---------------------------------------------------------------------------
proc qkNormKernel(
  xData, weightData: ptr float32,
  nHead, headDim: cint, eps: float32
) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let h = int(blockIdx.x)
  let tid = int(threadIdx.x)
  if h >= int(nHead):
    return
  let x = cast[ptr UncheckedArray[float32]](xData)
  let w = cast[ptr UncheckedArray[float32]](weightData)
  let hOff = h * int(headDim)
  var ss = 0.0'f32
  var r = tid
  while r < int(headDim):
    let v = x[hOff + r]
    ss = ss + v * v
    r = r + int(blockDim.x)
  sdata[tid] = ss
  hippoSyncthreads()
  reduceSum256(sdata, tid)
  var inv {.hippoShared.}: array[1, float32]
  if tid == 0:
    inv[0] = 1.0'f32 / sqrtf(sdata[0] / cfloat(headDim) + cfloat(eps))
  hippoSyncthreads()
  let invVal = inv[0]
  r = tid
  while r < int(headDim):
    x[hOff + r] = x[hOff + r] * invVal * w[r]
    r = r + int(blockDim.x)

proc gpuQkNorm*(x, weight: pointer, nHead, headDim: int,
                eps: float32, stream: HippoStream) =
  let grid = newDim3(nHead.uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var xPtr = x; var wPtr = weight
  var nHeadArg = nHead.cint; var hdArg = headDim.cint; var epsArg = eps
  hippoLaunchKernel(qkNormKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(xPtr, wPtr, nHeadArg, hdArg, epsArg))

proc qkNormPrefillKernel(
  xData, weightData: ptr float32,
  nHead, headDim, seqLen: cint, eps: float32
) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let blockId = int(blockIdx.x)
  let h = blockId div int(seqLen)
  let col = blockId mod int(seqLen)
  let tid = int(threadIdx.x)
  if h >= int(nHead):
    return
  let x = cast[ptr UncheckedArray[float32]](xData)
  let w = cast[ptr UncheckedArray[float32]](weightData)
  let baseRow = h * int(headDim)
  var ss = 0.0'f32
  var r = tid
  while r < int(headDim):
    let v = x[(baseRow + r) * int(seqLen) + col]
    ss = ss + v * v
    r = r + int(blockDim.x)
  sdata[tid] = ss
  hippoSyncthreads()
  reduceSum256(sdata, tid)
  var inv {.hippoShared.}: array[1, float32]
  if tid == 0:
    inv[0] = 1.0'f32 / sqrtf(sdata[0] / cfloat(headDim) + cfloat(eps))
  hippoSyncthreads()
  let invVal = inv[0]
  r = tid
  while r < int(headDim):
    let idx = (baseRow + r) * int(seqLen) + col
    x[idx] = x[idx] * invVal * w[r]
    r = r + int(blockDim.x)

proc gpuQkNormPrefill*(x, weight: pointer, nHead, headDim, seqLen: int,
                       eps: float32, stream: HippoStream) =
  let grid = newDim3((nHead * seqLen).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var xPtr = x; var wPtr = weight
  var nHeadArg = nHead.cint; var hdArg = headDim.cint
  var seqLenArg = seqLen.cint; var epsArg = eps
  hippoLaunchKernel(qkNormPrefillKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(xPtr, wPtr, nHeadArg, hdArg, seqLenArg, epsArg))

# ---------------------------------------------------------------------------
# Embedding kernel
# ---------------------------------------------------------------------------
proc embeddingKernel(
  weightData, outData: ptr float32,
  tokenIds: ptr int32,
  nEmb, nTokens, nVocab: cint
) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  let totalElems = int(nEmb) * int(nTokens)
  if idx < totalElems:
    let e = idx div int(nTokens)
    let t = idx mod int(nTokens)
    let w = cast[ptr UncheckedArray[float32]](weightData)
    let o = cast[ptr UncheckedArray[float32]](outData)
    let toks = cast[ptr UncheckedArray[int32]](tokenIds)
    let tid = int(toks[t])
    o[e * int(nTokens) + t] = w[tid * int(nEmb) + e]

proc gpuEmbedding*(dst: pointer, weight: pointer, tokenIds: ptr int32,
                    nEmb, nTokens, nVocab: int, stream: HippoStream) =
  let total = nEmb * nTokens
  let grid = newDim3(((total + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var wPtr = weight; var dPtr = dst; var tPtr = cast[pointer](tokenIds)
  var nEmbArg = nEmb.cint; var nTokArg = nTokens.cint; var nVocArg = nVocab.cint
  hippoLaunchKernel(embeddingKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(wPtr, dPtr, tPtr, nEmbArg, nTokArg, nVocArg))

# ---------------------------------------------------------------------------
# RoPE kernels
# ---------------------------------------------------------------------------
proc ropeAtPosKernel(
  xData: ptr float32,
  nHead, headDim, ropeDim: cint,
  ropeBase: float32, pos: cint, seqLen: cint
) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  let halfRope = int(ropeDim) div 2
  let totalPairs = int(nHead) * halfRope
  if idx >= totalPairs:
    return
  let x = cast[ptr UncheckedArray[float32]](xData)
  let h = idx div halfRope
  let i = idx mod halfRope
  let hOffset = h * int(headDim)
  let theta = powf(1.0'f32 / cfloat(ropeBase), cfloat(2 * i) / cfloat(ropeDim))
  let angle = cfloat(pos) * theta
  let c = cosf(angle)
  let s = sinf(angle)
  if int(seqLen) == 1:
    let idx0 = hOffset + i
    let idx1 = hOffset + halfRope + i
    let v0 = x[idx0]
    let v1 = x[idx1]
    x[idx0] = v0 * c - v1 * s
    x[idx1] = v0 * s + v1 * c
  else:
    var p = 0
    while p < int(seqLen):
      let idx0 = (hOffset + i) * int(seqLen) + p
      let idx1 = (hOffset + halfRope + i) * int(seqLen) + p
      let pTheta = powf(1.0'f32 / cfloat(ropeBase), cfloat(2 * i) / cfloat(ropeDim))
      let pAngle = cfloat(p) * pTheta
      let pc = cosf(pAngle)
      let ps = sinf(pAngle)
      let v0 = x[idx0]
      let v1 = x[idx1]
      x[idx0] = v0 * pc - v1 * ps
      x[idx1] = v0 * ps + v1 * pc
      p = p + 1

proc ropeQKDecodeKernel(
  qData, kData: ptr float32,
  thetaData: ptr float32,
  nHeadQ, nHeadK, headDim, halfRope: cint,
  pos: cint
) {.hippoGlobal.} =
  let idx = cint(blockIdx.x * blockDim.x + threadIdx.x)
  let qPairs = nHeadQ * halfRope
  let totalPairs = qPairs + nHeadK * halfRope
  if idx >= totalPairs:
    return
  let thetaArr = cast[ptr UncheckedArray[float32]](thetaData)
  let isK = idx >= qPairs
  let localIdx = if isK: idx - qPairs else: idx
  let x = if isK: cast[ptr UncheckedArray[float32]](kData)
          else: cast[ptr UncheckedArray[float32]](qData)
  let h = localIdx div halfRope
  let i = localIdx mod halfRope
  let hOffset = h * headDim
  let angle = cfloat(pos) * thetaArr[i]
  let c = cosf(angle)
  let s = sinf(angle)
  let idx0 = hOffset + i
  let idx1 = hOffset + halfRope + i
  let v0 = x[idx0]
  let v1 = x[idx1]
  x[idx0] = v0 * c - v1 * s
  x[idx1] = v0 * s + v1 * c

proc gpuRopeQKDecode*(q, k: pointer, nHeadQ, nHeadK, headDim, ropeDim: int,
                       ropeBase: float32, pos: int, stream: HippoStream) =
  let halfRope = ropeDim div 2
  let totalPairs = nHeadQ * halfRope + nHeadK * halfRope
  let grid = newDim3(((totalPairs + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var qPtr = q; var kPtr = k; var thetaPtr = modelPtrs.ropeTheta
  var nHQ = nHeadQ.cint; var nHK = nHeadK.cint
  var hdArg = headDim.cint; var hrArg = halfRope.cint; var posArg = pos.cint
  hippoLaunchKernel(ropeQKDecodeKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(qPtr, kPtr, thetaPtr, nHQ, nHK, hdArg, hrArg, posArg))

proc gpuRopeAtPos*(x: pointer, nHead, headDim, ropeDim: int,
                    ropeBase: float32, pos: int, seqLen: int,
                    stream: HippoStream) =
  let halfRope = ropeDim div 2
  let totalPairs = nHead * halfRope
  let grid = newDim3(((totalPairs + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var xPtr = x; var nHeadArg = nHead.cint; var headDimArg = headDim.cint
  var ropeDimArg = ropeDim.cint; var ropeBaseArg = ropeBase
  var posArg = pos.cint; var seqLenArg = seqLen.cint
  hippoLaunchKernel(ropeAtPosKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(xPtr, nHeadArg, headDimArg, ropeDimArg,
                                     ropeBaseArg, posArg, seqLenArg))

proc ropeAtPosThetaKernel(
  xData, thetaData: ptr float32,
  nHead, headDim, halfRope: cint,
  pos: cint, seqLen: cint
) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  let totalPairs = int(nHead) * int(halfRope)
  if idx >= totalPairs: return
  let x = cast[ptr UncheckedArray[float32]](xData)
  let th = cast[ptr UncheckedArray[float32]](thetaData)
  let h = idx div int(halfRope)
  let i = idx mod int(halfRope)
  let hOffset = h * int(headDim)
  if int(seqLen) == 1:
    let angle = cfloat(pos) * th[i]
    let c = cosf(angle); let s = sinf(angle)
    let idx0 = hOffset + i; let idx1 = hOffset + int(halfRope) + i
    let v0 = x[idx0]; let v1 = x[idx1]
    x[idx0] = v0 * c - v1 * s; x[idx1] = v0 * s + v1 * c
  else:
    var p = 0
    while p < int(seqLen):
      let angle = cfloat(pos + p) * th[i]
      let c = cosf(angle); let s = sinf(angle)
      let idx0 = (hOffset + i) * int(seqLen) + p
      let idx1 = (hOffset + int(halfRope) + i) * int(seqLen) + p
      let v0 = x[idx0]; let v1 = x[idx1]
      x[idx0] = v0 * c - v1 * s; x[idx1] = v0 * s + v1 * c
      p = p + 1

proc gpuRopeAtPosTheta*(x: pointer, nHead, headDim, ropeDim: int,
                          pos, seqLen: int, stream: HippoStream) =
  let halfRope = ropeDim div 2
  let totalPairs = nHead * halfRope
  let grid = newDim3(((totalPairs + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var xPtr = x; var thetaPtr = modelPtrs.ropeTheta
  var nH = nHead.cint; var hd = headDim.cint; var hr = halfRope.cint
  var posArg = pos.cint; var slArg = seqLen.cint
  hippoLaunchKernel(ropeAtPosThetaKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(xPtr, thetaPtr, nH, hd, hr, posArg, slArg))

# ---------------------------------------------------------------------------
# Fused RoPE + KV store kernel
# ---------------------------------------------------------------------------
proc fusedRopeStoreKVKernel(
  qData, kData, vData: ptr float32,
  kCacheData, vCacheData: ptr float32,
  thetaData: ptr float32,
  nHeadQ, nHeadK, headDim, halfRope: cint,
  kvDim, cacheCols, pos: cint
) {.hippoGlobal.} =
  let idx = cint(blockIdx.x * blockDim.x + threadIdx.x)
  let qPairs = nHeadQ * halfRope
  let kPairs = nHeadK * halfRope
  let ropePairs = qPairs + kPairs
  let kvElems = kvDim
  let total = ropePairs + kvElems * 2'i32
  if idx >= total:
    return
  let thetaArr = cast[ptr UncheckedArray[float32]](thetaData)
  if idx < ropePairs:
    let isK = idx >= qPairs
    let localIdx = if isK: idx - qPairs else: idx
    let x = if isK: cast[ptr UncheckedArray[float32]](kData)
            else: cast[ptr UncheckedArray[float32]](qData)
    let h = localIdx div halfRope
    let i = localIdx mod halfRope
    let hOffset = h * headDim
    let angle = cfloat(pos) * thetaArr[i]
    let c = cosf(angle)
    let s = sinf(angle)
    let idx0 = hOffset + i
    let idx1 = hOffset + halfRope + i
    let v0 = x[idx0]
    let v1 = x[idx1]
    x[idx0] = v0 * c - v1 * s
    x[idx1] = v0 * s + v1 * c
    if isK:
      let kc = cast[ptr UncheckedArray[float32]](kCacheData)
      kc[idx0 * cacheCols + pos] = v0 * c - v1 * s
      kc[idx1 * cacheCols + pos] = v0 * s + v1 * c
  else:
    let storeIdx = idx - ropePairs
    let isV = storeIdx >= kvElems
    let localIdx = if isV: storeIdx - kvElems else: storeIdx
    if isV:
      let vc = cast[ptr UncheckedArray[float32]](vCacheData)
      let v = cast[ptr UncheckedArray[float32]](vData)
      vc[localIdx * cacheCols + pos] = v[localIdx]
    else:
      let kc = cast[ptr UncheckedArray[float32]](kCacheData)
      let k = cast[ptr UncheckedArray[float32]](kData)
      let d = localIdx
      let hk = d div headDim
      let di = d mod headDim
      if di >= 2'i32 * halfRope:
        kc[d * cacheCols + pos] = k[d]

proc gpuFusedRopeStoreKV*(q, k, v: pointer, kCache, vCache: pointer,
                            nHeadQ, nHeadK, headDim, ropeDim: int,
                            kvDim, cacheCols, pos: int,
                            stream: HippoStream) =
  let halfRope = ropeDim div 2
  let ropePairs = nHeadQ * halfRope + nHeadK * halfRope
  let kvElems = kvDim
  let total = ropePairs + kvElems * 2
  let grid = newDim3(((total + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var qPtr = q; var kPtr = k; var vPtr = v
  var kcPtr = kCache; var vcPtr = vCache
  var thetaPtr = modelPtrs.ropeTheta
  var nHQ = nHeadQ.cint; var nHK = nHeadK.cint
  var hdArg = headDim.cint; var hrArg = halfRope.cint
  var kvDimArg = kvDim.cint; var cacheColsArg = cacheCols.cint; var posArg = pos.cint
  hippoLaunchKernel(fusedRopeStoreKVKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(qPtr, kPtr, vPtr, kcPtr, vcPtr, thetaPtr,
                                     nHQ, nHK, hdArg, hrArg,
                                     kvDimArg, cacheColsArg, posArg))

# ---------------------------------------------------------------------------
# KV store kernels
# ---------------------------------------------------------------------------
proc storeKVKernel(
  cacheData, srcData: ptr float32,
  rows, srcCols, cacheCols, startPos: cint
) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  let totalElems = int(rows) * int(srcCols)
  if idx < totalElems:
    let r = idx div int(srcCols)
    let c = idx mod int(srcCols)
    let cache = cast[ptr UncheckedArray[float32]](cacheData)
    let src = cast[ptr UncheckedArray[float32]](srcData)
    cache[r * int(cacheCols) + int(startPos) + c] = src[r * int(srcCols) + c]

proc gpuStoreKV*(cache: pointer, src: pointer,
                  rows, srcCols, cacheCols, startPos: int,
                  stream: HippoStream) =
  let total = rows * srcCols
  let grid = newDim3(((total + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var cPtr = cache; var sPtr = src
  var rowsArg = rows.cint; var srcColsArg = srcCols.cint
  var cacheColsArg = cacheCols.cint; var startPosArg = startPos.cint
  hippoLaunchKernel(storeKVKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(cPtr, sPtr, rowsArg, srcColsArg,
                                     cacheColsArg, startPosArg))

proc storeKVPairKernel(
  kCacheData, kSrcData, vCacheData, vSrcData: ptr float32,
  rows, srcCols, cacheCols, startPos: cint
) {.hippoGlobal.} =
  let idx = cint(blockIdx.x * blockDim.x + threadIdx.x)
  let totalPerKV = rows * srcCols
  if idx >= totalPerKV * 2'i32:
    return
  let isV = idx >= totalPerKV
  let localIdx = if isV: idx - totalPerKV else: idx
  let cache = if isV: cast[ptr UncheckedArray[float32]](vCacheData)
              else: cast[ptr UncheckedArray[float32]](kCacheData)
  let src = if isV: cast[ptr UncheckedArray[float32]](vSrcData)
            else: cast[ptr UncheckedArray[float32]](kSrcData)
  let r = localIdx div srcCols
  let c = localIdx mod srcCols
  cache[r * cacheCols + startPos + c] = src[r * srcCols + c]

proc gpuStoreKVPair*(kCache, kSrc, vCache, vSrc: pointer,
                      rows, srcCols, cacheCols, startPos: int,
                      stream: HippoStream) =
  let totalBoth = rows * srcCols * 2
  let grid = newDim3(((totalBoth + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var kcPtr = kCache; var ksPtr = kSrc; var vcPtr = vCache; var vsPtr = vSrc
  var rowsArg = rows.cint; var srcColsArg = srcCols.cint
  var cacheColsArg = cacheCols.cint; var startPosArg = startPos.cint
  hippoLaunchKernel(storeKVPairKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(kcPtr, ksPtr, vcPtr, vsPtr, rowsArg,
                                     srcColsArg, cacheColsArg, startPosArg))

# ---------------------------------------------------------------------------
# Attention kernels
# ---------------------------------------------------------------------------
const AttnTileSize = 256

proc attentionDecodeKernel(
  qData, kCacheData, vCacheData, outData: ptr float32,
  nHead, nHeadKv, headDim, curLen, cacheCols: cint,
  invSqrtHeadDim: float32
) {.hippoGlobal.} =
  ## Tiled online-softmax decode attention — no context length cap.
  ## One block per head, 256 threads. Processes KV positions in tiles.
  let h = int(blockIdx.x)
  let tid = int(threadIdx.x)
  if h >= int(nHead):
    return
  let q = cast[ptr UncheckedArray[float32]](qData)
  let kc = cast[ptr UncheckedArray[float32]](kCacheData)
  let vc = cast[ptr UncheckedArray[float32]](vCacheData)
  let o = cast[ptr UncheckedArray[float32]](outData)
  let group = int(nHead) div int(nHeadKv)
  let kvh = h div group
  let hOff = h * int(headDim)
  let kvhOff = kvh * int(headDim)
  var scores {.hippoShared.}: array[AttnTileSize, float32]
  var sMax {.hippoShared.}: array[HippoBlockSize, float32]
  var sSum {.hippoShared.}: array[HippoBlockSize, float32]
  let numTiles = (int(curLen) + AttnTileSize - 1) div AttnTileSize
  var globalMax = -1e30'f32
  var globalSum = 0.0'f32
  var outAcc = 0.0'f32
  for tile in 0 ..< numTiles:
    let tileStart = tile * AttnTileSize
    let tileEnd = min(tileStart + AttnTileSize, int(curLen))
    let tileLen = tileEnd - tileStart
    var localMax = -1e30'f32
    var j = tid
    while j < tileLen:
      var dot = 0.0'f32
      for d in 0 ..< int(headDim):
        dot = dot + q[hOff + d] * kc[(kvhOff + d) * int(cacheCols) + tileStart + j]
      let score = dot * invSqrtHeadDim
      scores[j] = score
      if score > localMax:
        localMax = score
      j = j + int(blockDim.x)
    sMax[tid] = localMax
    hippoSyncthreads()
    reduceMax256(sMax, tid)
    let tileMax = sMax[0]
    hippoSyncthreads()
    var localSum = 0.0'f32
    j = tid
    while j < tileLen:
      let e = expf(scores[j] - tileMax)
      scores[j] = e
      localSum = localSum + e
      j = j + int(blockDim.x)
    sSum[tid] = localSum
    hippoSyncthreads()
    reduceSum256(sSum, tid)
    let tileSum = sSum[0]
    hippoSyncthreads()
    var scoresCorrection = 1.0'f32
    var outCorrection = 1.0'f32
    if tileMax > globalMax:
      outCorrection = expf(globalMax - tileMax)
      globalSum = globalSum * outCorrection + tileSum
      globalMax = tileMax
    else:
      scoresCorrection = expf(tileMax - globalMax)
      globalSum = globalSum + tileSum * scoresCorrection
    if tid < int(headDim):
      outAcc = outAcc * outCorrection
      var acc = 0.0'f32
      for jj in 0 ..< tileLen:
        acc = acc + scores[jj] * vc[(kvhOff + tid) * int(cacheCols) + tileStart + jj]
      outAcc = outAcc + acc * scoresCorrection
    hippoSyncthreads()
  if tid < int(headDim):
    o[hOff + tid] = outAcc / globalSum

proc gpuAttentionDecode*(dst: pointer, q, kCache, vCache: pointer,
                          nHead, nHeadKv, headDim, curLen, cacheCols: int,
                          stream: HippoStream) =
  let grid = newDim3(nHead.uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  let invSqrt = 1.0'f32 / sqrt(headDim.float32)
  var qPtr = q; var kcPtr = kCache; var vcPtr = vCache; var dPtr = dst
  var nHeadArg = nHead.cint; var nHeadKvArg = nHeadKv.cint
  var headDimArg = headDim.cint; var curLenArg = curLen.cint
  var cacheColsArg = cacheCols.cint; var invSqrtArg = invSqrt
  hippoLaunchKernel(attentionDecodeKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(qPtr, kcPtr, vcPtr, dPtr,
                                     nHeadArg, nHeadKvArg, headDimArg,
                                     curLenArg, cacheColsArg, invSqrtArg))

proc attentionPrefillKernel(
  qData, kData, vData, outData: ptr float32,
  nHead, nHeadKv, headDim, seqLen: cint,
  invSqrtHeadDim: float32
) {.hippoGlobal.} =
  let blockId = int(blockIdx.x)
  let h = blockId div int(seqLen)
  let qi = blockId mod int(seqLen)
  let tid = int(threadIdx.x)
  if h >= int(nHead):
    return
  let qArr = cast[ptr UncheckedArray[float32]](qData)
  let kArr = cast[ptr UncheckedArray[float32]](kData)
  let vArr = cast[ptr UncheckedArray[float32]](vData)
  let oArr = cast[ptr UncheckedArray[float32]](outData)
  let group = int(nHead) div int(nHeadKv)
  let kvh = h div group
  let hOff = h * int(headDim)
  let causalLen = qi + 1
  var scores {.hippoShared.}: array[4096, float32]
  var sMax {.hippoShared.}: array[HippoBlockSize, float32]
  var sSum {.hippoShared.}: array[HippoBlockSize, float32]
  var localMax = -1e30'f32
  var j = tid
  while j < causalLen:
    var dot = 0.0'f32
    for d in 0 ..< int(headDim):
      let qIdx = (hOff + d) * int(seqLen) + qi
      let kIdx = (kvh * int(headDim) + d) * int(seqLen) + j
      dot = dot + qArr[qIdx] * kArr[kIdx]
    let score = dot * invSqrtHeadDim
    scores[j] = score
    if score > localMax:
      localMax = score
    j = j + int(blockDim.x)
  sMax[tid] = localMax
  hippoSyncthreads()
  reduceMax256(sMax, tid)
  let globalMax = sMax[0]
  var localSum = 0.0'f32
  j = tid
  while j < causalLen:
    let e = expf(scores[j] - globalMax)
    scores[j] = e
    localSum = localSum + e
    j = j + int(blockDim.x)
  sSum[tid] = localSum
  hippoSyncthreads()
  reduceSum256(sSum, tid)
  let invSum = 1.0'f32 / sSum[0]
  j = tid
  while j < causalLen:
    scores[j] = scores[j] * invSum
    j = j + int(blockDim.x)
  hippoSyncthreads()
  var d = tid
  while d < int(headDim):
    var acc = 0.0'f32
    for jj in 0 ..< causalLen:
      let vIdx = (kvh * int(headDim) + d) * int(seqLen) + jj
      acc = acc + scores[jj] * vArr[vIdx]
    let outIdx = (hOff + d) * int(seqLen) + qi
    oArr[outIdx] = acc
    d = d + int(blockDim.x)

proc gpuAttentionPrefill*(dst: pointer, q, k, v: pointer,
                           nHead, nHeadKv, headDim, seqLen: int,
                           stream: HippoStream) =
  let nBlocks = nHead * seqLen
  let grid = newDim3(nBlocks.uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  let invSqrt = 1.0'f32 / sqrt(headDim.float32)
  var qPtr = q; var kPtr = k; var vPtr = v; var dPtr = dst
  var nHeadArg = nHead.cint; var nHeadKvArg = nHeadKv.cint
  var headDimArg = headDim.cint; var seqLenArg = seqLen.cint; var invSqrtArg = invSqrt
  hippoLaunchKernel(attentionPrefillKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(qPtr, kPtr, vPtr, dPtr,
                                     nHeadArg, nHeadKvArg, headDimArg,
                                     seqLenArg, invSqrtArg))

# ---------------------------------------------------------------------------
# Float32 GEMM/GEMV kernels
# ---------------------------------------------------------------------------
proc linearHippoKernel(
  wData, xData, outData: ptr float32,
  outRows, wCols, seqLen: cint
) {.hippoGlobal.} =
  let outRow = int(blockIdx.y * blockDim.y + threadIdx.y)
  let seqCol = int(blockIdx.x * blockDim.x + threadIdx.x)
  if outRow < int(outRows) and seqCol < int(seqLen):
    let wArray = cast[ptr UncheckedArray[float32]](wData)
    let xArray = cast[ptr UncheckedArray[float32]](xData)
    let outArray = cast[ptr UncheckedArray[float32]](outData)
    var acc = 0.0'f32
    for k in 0 ..< int(wCols):
      acc = acc + wArray[outRow * int(wCols) + k] * xArray[k * int(seqLen) + seqCol]
    outArray[outRow * int(seqLen) + seqCol] = acc

proc linearHippoDecodeKernel(
  wData, xData, outData: ptr float32,
  outRows, wCols: cint
) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let tid = int(threadIdx.x)
  let blockSize = int(blockDim.x)
  let cols = int(wCols)
  let unrollSpan = HippoDecodeDotUnroll * blockSize
  let baseRow = int(blockIdx.x) * HippoDecodeRowsPerBlock
  let wArray = cast[ptr UncheckedArray[float32]](wData)
  let xArray = cast[ptr UncheckedArray[float32]](xData)
  let outArray = cast[ptr UncheckedArray[float32]](outData)
  for r in 0 ..< HippoDecodeRowsPerBlock:
    let outRow = baseRow + r
    if outRow < int(outRows):
      let rowBase = outRow * cols
      var acc = 0.0'f32
      var k = tid
      while k + (HippoDecodeDotUnroll - 1) * blockSize < cols:
        let k1 = k + blockSize
        let k2 = k1 + blockSize
        let k3 = k2 + blockSize
        acc = acc + wArray[rowBase + k] * xArray[k]
        acc = acc + wArray[rowBase + k1] * xArray[k1]
        acc = acc + wArray[rowBase + k2] * xArray[k2]
        acc = acc + wArray[rowBase + k3] * xArray[k3]
        k = k + unrollSpan
      while k < cols:
        acc = acc + wArray[rowBase + k] * xArray[k]
        k = k + blockSize
      sdata[tid] = acc
    else:
      sdata[tid] = 0.0'f32
    hippoSyncthreads()
    reduceSum256(sdata, tid)
    if tid == 0 and outRow < int(outRows):
      outArray[outRow] = sdata[0]
    hippoSyncthreads()

proc gpuLinearCol*(dst, x, w: pointer, wCols, wRows, seqLen: int,
                   stream: HippoStream) =
  ## Dispatch float32 GEMM (prefill) or GEMV (decode).
  var wPtr = w; var xPtr = x; var dPtr = dst
  var outRowsArg = wRows.cint; var wColsArg = wCols.cint
  if seqLen == 1:
    if wCols > HippoMaxDecodeCols:
      raise newException(ValueError,
        "decode GEMV width exceeds HippoMaxDecodeCols: " & $wCols &
        " > " & $HippoMaxDecodeCols)
    let grid = newDim3(((wRows + HippoDecodeRowsPerBlock - 1) div HippoDecodeRowsPerBlock).uint32)
    let blk = newDim3(HippoBlockSize.uint32)
    hippoLaunchKernel(linearHippoDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    let gridX = (seqLen + HippoBlockSizeX - 1) div HippoBlockSizeX
    let gridY = (wRows + HippoBlockSizeY - 1) div HippoBlockSizeY
    let grid = newDim3(gridX.uint32, gridY.uint32)
    let blk = newDim3(HippoBlockSizeX.uint32, HippoBlockSizeY.uint32)
    var seqLenArg = seqLen.cint
    hippoLaunchKernel(linearHippoKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg, seqLenArg))

# ---------------------------------------------------------------------------
# GPU KV Cache init
# ---------------------------------------------------------------------------
proc initGpuKvCache*(nLayer, nHeadKv, headDim, maxLen: int,
                      layerNHeadKv: seq[int] = @[]): GpuKvCache =
  ensureGpuContext()
  result.maxLen = maxLen
  result.curLen = 0
  result.nHeadKv = nHeadKv
  result.headDim = headDim
  result.k = newSeq[GpuTensor](nLayer)
  result.v = newSeq[GpuTensor](nLayer)
  for i in 0 ..< nLayer:
    let lkv = if layerNHeadKv.len > i and layerNHeadKv[i] > 0: layerNHeadKv[i]
              else: nHeadKv
    if lkv > 0:
      let kvDim = lkv * headDim
      result.k[i] = newGpuTensor(@[kvDim, maxLen])
      result.v[i] = newGpuTensor(@[kvDim, maxLen])

proc initGpuSsmState*(hp: HParams, nLayer: int): SsmGpuState =
  ensureGpuContext()
  if hp.ssmInnerSize <= 0: return
  let isQwen35Moe = hp.arch == "qwen35moe"
  let nVHeads = hp.ssmDtRank
  let headVDim = hp.ssmInnerSize div nVHeads
  # convDim: channels for 1D convolution
  # Nemotron-H Mamba2: ssmInnerSize + 2 * nGroups * stateSize
  # Qwen3.6 Delta Net: 2 * nKHeads * stateSize + ssmInnerSize (Q + K + V)
  let convDim = if isQwen35Moe:
    2 * hp.ssmGroupCount * hp.ssmStateSize + hp.ssmInnerSize
  else:
    hp.ssmInnerSize + 2 * hp.ssmGroupCount * hp.ssmStateSize
  let convLen = hp.ssmConvKernel - 1
  result.ssmLayerMap = newSeq[int](nLayer)
  var ssmIdx = 0
  for i in 0 ..< nLayer:
    result.ssmLayerMap[i] = -1
  result.convState = newSeq[GpuTensor](0)
  result.recState = newSeq[GpuTensor](0)
  for i in 0 ..< nLayer:
    let hasSsm = if isQwen35Moe:
      hp.fullAttnInterval > 0 and (i mod hp.fullAttnInterval) != (hp.fullAttnInterval - 1)
    else:
      hp.layerNHeadKv.len > i and hp.layerNHeadKv[i] == 0 and
      hp.layerNFfn.len > i and hp.layerNFfn[i] == 0
    if hasSsm:
      result.ssmLayerMap[i] = ssmIdx
      let cs = newGpuTensor(@[convDim, convLen])
      var csZeros = newSeq[byte](cs.sizeBytes)
      hippoMemcpy(cs.devicePtr, addr csZeros[0], cs.sizeBytes, HippoMemcpyHostToDevice)
      result.convState.add(cs)
      let rs = newGpuTensor(@[nVHeads * hp.ssmStateSize * headVDim])
      var rsZeros = newSeq[byte](rs.sizeBytes)
      hippoMemcpy(rs.devicePtr, addr rsZeros[0], rs.sizeBytes, HippoMemcpyHostToDevice)
      result.recState.add(rs)
      inc ssmIdx

# ---------------------------------------------------------------------------
# Weight upload and model GPU pointer setup
# ---------------------------------------------------------------------------
proc ensureModelGpuPtrs*(m: var Model, hp: HParams) =
  ## Populate modelPtrs once by caching all weight device pointers.
  if modelPtrs.initialized:
    return
  ensureGpuContext()

  template loadF32Ptr(name: string): pointer =
    if m.infos[name].elemType == GgmlTypeF16:
      cachedF16WeightAsF32(name, m, name).devicePtr
    else:
      cachedWeight(name, m.getTensor(name)).devicePtr

  let tokEmbName = if m.infos.hasKey("token_embd.weight"): "token_embd.weight"
                   else: "tok_embeddings.weight"
  modelPtrs.tokEmb = loadF32Ptr(tokEmbName)

  modelPtrs.layers = newSeq[LayerGpuPtrs](hp.nLayer)
  for layer in 0 ..< hp.nLayer:
    let lp = "blk." & $layer & "."
    var lw: LayerGpuPtrs

    template uploadWeight(fp32Field, quantField, qtypeField: untyped, tensorSuffix: string) =
      let tn = lp & tensorSuffix
      let et = m.infos[tn].elemType.int32
      if et == GgmlTypeF16.int32 or
         et == GgmlTypeQ5_0.int32 or et == GgmlTypeQ5_1.int32 or
         et == GgmlTypeQ2K.int32 or et == GgmlTypeQ3K.int32 or
         et == GgmlTypeQ4K.int32 or et == GgmlTypeQ5K.int32 or
         et == GgmlTypeQ6K.int32 or et == GgmlTypeQ8_0.int32 or
         et == GgmlTypeIQ4NL.int32:
        let qw = cachedQuantWeight(tn, m, tn)
        quantField = qw.devicePtr
        fp32Field = nil
        qtypeField = et
      else:
        fp32Field = cachedWeight(tn, m.getTensor(tn)).devicePtr
        quantField = nil
        qtypeField = 0

    # Classify layer
    if hp.arch == "qwen35moe":
      if m.infos.hasKey(lp & "ssm_out.weight"):
        lw.kind = lkSsmAttnMoe
      else:
        lw.kind = lkAttnMoe
    elif m.infos.hasKey(lp & "ssm_in.weight"):
      lw.kind = lkSsm
    elif m.infos.hasKey(lp & "ffn_gate_inp.weight") and
         not m.infos.hasKey(lp & "attn_q.weight") and
         not m.infos.hasKey(lp & "attn_qkv.weight"):
      lw.kind = lkMoeFfn
    elif hp.layerNHeadKv.len > layer and hp.layerNHeadKv[layer] > 0:
      if m.infos.hasKey(lp & "ffn_gate.weight") or m.infos.hasKey(lp & "ffn_up.weight"):
        lw.kind = lkAttentionFfn
      else:
        lw.kind = lkAttention
    elif hp.layerNFfn.len > layer and hp.layerNFfn[layer] > 0:
      lw.kind = lkFfnOnly
    else:
      lw.kind = lkAttentionFfn

    lw.attnNorm = loadF32Ptr(lp & "attn_norm.weight")
    lw.layerNHeadKv = if hp.layerNHeadKv.len > layer: hp.layerNHeadKv[layer] else: hp.nHeadKv
    lw.layerNFfn = if hp.layerNFfn.len > layer: hp.layerNFfn[layer]
                   else: hp.nFfn

    case lw.kind
    of lkAttentionFfn:
      lw.ffnNorm = loadF32Ptr(lp & "ffn_norm.weight")
      uploadWeight(lw.wq, lw.wqQ, lw.wqQType, "attn_q.weight")
      uploadWeight(lw.wk, lw.wkQ, lw.wkQType, "attn_k.weight")
      uploadWeight(lw.wv, lw.wvQ, lw.wvQType, "attn_v.weight")
      uploadWeight(lw.wo, lw.woQ, lw.woQType, "attn_output.weight")
      uploadWeight(lw.wGate, lw.wGateQ, lw.wGateQType, "ffn_gate.weight")
      uploadWeight(lw.wUp, lw.wUpQ, lw.wUpQType, "ffn_up.weight")
      uploadWeight(lw.wDown, lw.wDownQ, lw.wDownQType, "ffn_down.weight")
      lw.wColsQ = hp.nEmb
      lw.wColsDown = hp.nFfn
      if m.infos.hasKey(lp & "attn_q_norm.weight"):
        lw.attnQNorm = loadF32Ptr(lp & "attn_q_norm.weight")
      if m.infos.hasKey(lp & "attn_k_norm.weight"):
        lw.attnKNorm = loadF32Ptr(lp & "attn_k_norm.weight")
    of lkAttention:
      uploadWeight(lw.wq, lw.wqQ, lw.wqQType, "attn_q.weight")
      uploadWeight(lw.wk, lw.wkQ, lw.wkQType, "attn_k.weight")
      uploadWeight(lw.wv, lw.wvQ, lw.wvQType, "attn_v.weight")
      uploadWeight(lw.wo, lw.woQ, lw.woQType, "attn_output.weight")
      lw.wColsQ = hp.nEmb
    of lkFfnOnly:
      uploadWeight(lw.wUp, lw.wUpQ, lw.wUpQType, "ffn_up.weight")
      uploadWeight(lw.wDown, lw.wDownQ, lw.wDownQType, "ffn_down.weight")
      lw.wColsQ = hp.nEmb
      lw.wColsDown = lw.layerNFfn
    of lkSsm:
      uploadWeight(lw.wUp, lw.ssmInQ, lw.ssmInQType, "ssm_in.weight")
      uploadWeight(lw.wDown, lw.ssmOutQ, lw.ssmOutQType, "ssm_out.weight")
      lw.ssmConv1dW = loadF32Ptr(lp & "ssm_conv1d.weight")
      lw.ssmConv1dBias = loadF32Ptr(lp & "ssm_conv1d.bias")
      lw.ssmDtBias = loadF32Ptr(lp & "ssm_dt.bias")
      lw.ssmA = loadF32Ptr(lp & "ssm_a")
      lw.ssmD = loadF32Ptr(lp & "ssm_d")
      lw.ssmNorm = loadF32Ptr(lp & "ssm_norm.weight")

    of lkMoeFfn:
      if m.infos.hasKey(lp & "ffn_norm.weight"):
        lw.ffnNorm = loadF32Ptr(lp & "ffn_norm.weight")
      lw.moeRouterW = loadF32Ptr(lp & "ffn_gate_inp.weight")
      if m.infos.hasKey(lp & "exp_probs_b.bias"):
        lw.moeExpertBias = loadF32Ptr(lp & "exp_probs_b.bias")
      block:
        let utn = lp & "ffn_up_exps.weight"
        let uInfo = m.infos[utn]
        lw.moeUpExpsQ = cachedQuantWeight(utn, m, utn).devicePtr
        lw.moeUpExpsQType = uInfo.elemType
        lw.moeUpExpsSliceBytes = quantRowSize(int(uInfo.ne[0]), uInfo.elemType) * int(uInfo.ne[1])
        let dtn = lp & "ffn_down_exps.weight"
        let dInfo = m.infos[dtn]
        lw.moeDownExpsQ = cachedQuantWeight(dtn, m, dtn).devicePtr
        lw.moeDownExpsQType = dInfo.elemType
        lw.moeDownExpsSliceBytes = quantRowSize(int(dInfo.ne[0]), dInfo.elemType) * int(dInfo.ne[1])
      uploadWeight(lw.wUp, lw.moeShUpQ, lw.moeShUpQType, "ffn_up_shexp.weight")
      uploadWeight(lw.wDown, lw.moeShDownQ, lw.moeShDownQType, "ffn_down_shexp.weight")

    of lkAttnMoe:
      lw.postAttnNorm = loadF32Ptr(lp & "post_attention_norm.weight")
      uploadWeight(lw.wq, lw.wqQ, lw.wqQType, "attn_q.weight")
      uploadWeight(lw.wk, lw.wkQ, lw.wkQType, "attn_k.weight")
      uploadWeight(lw.wv, lw.wvQ, lw.wvQType, "attn_v.weight")
      uploadWeight(lw.wo, lw.woQ, lw.woQType, "attn_output.weight")
      lw.wColsQ = hp.nEmb
      if m.infos.hasKey(lp & "attn_q_norm.weight"):
        lw.attnQNorm = loadF32Ptr(lp & "attn_q_norm.weight")
      if m.infos.hasKey(lp & "attn_k_norm.weight"):
        lw.attnKNorm = loadF32Ptr(lp & "attn_k_norm.weight")
      # MoE weights
      lw.moeRouterW = loadF32Ptr(lp & "ffn_gate_inp.weight")
      block:
        let gtn = lp & "ffn_gate_exps.weight"
        let gInfo = m.infos[gtn]
        lw.moeGateExpsQ = cachedQuantWeight(gtn, m, gtn).devicePtr
        lw.moeGateExpsQType = gInfo.elemType
        lw.moeGateExpsSliceBytes = quantRowSize(int(gInfo.ne[0]), gInfo.elemType) * int(gInfo.ne[1])
        let utn = lp & "ffn_up_exps.weight"
        let uInfo = m.infos[utn]
        lw.moeUpExpsQ = cachedQuantWeight(utn, m, utn).devicePtr
        lw.moeUpExpsQType = uInfo.elemType
        lw.moeUpExpsSliceBytes = quantRowSize(int(uInfo.ne[0]), uInfo.elemType) * int(uInfo.ne[1])
        let dtn = lp & "ffn_down_exps.weight"
        let dInfo = m.infos[dtn]
        lw.moeDownExpsQ = cachedQuantWeight(dtn, m, dtn).devicePtr
        lw.moeDownExpsQType = dInfo.elemType
        lw.moeDownExpsSliceBytes = quantRowSize(int(dInfo.ne[0]), dInfo.elemType) * int(dInfo.ne[1])
      # Shared expert
      uploadWeight(lw.wGate, lw.moeShGateQ, lw.moeShGateQType, "ffn_gate_shexp.weight")
      uploadWeight(lw.wUp, lw.moeShUpQ, lw.moeShUpQType, "ffn_up_shexp.weight")
      uploadWeight(lw.wDown, lw.moeShDownQ, lw.moeShDownQType, "ffn_down_shexp.weight")
      lw.moeShGateScalar = loadF32Ptr(lp & "ffn_gate_inp_shexp.weight")

    of lkSsmAttnMoe:
      lw.postAttnNorm = loadF32Ptr(lp & "post_attention_norm.weight")
      # Delta Net QKV projection: nEmb → convDim (8192)
      uploadWeight(lw.wq, lw.wqkvQ, lw.wqkvQType, "attn_qkv.weight")
      # Gate projection: nEmb → ssmInnerSize (4096)
      uploadWeight(lw.wo, lw.ssmGateQ, lw.ssmGateQType, "attn_gate.weight")
      lw.wColsQ = hp.nEmb
      # SSM recurrence parameters
      uploadWeight(lw.wk, lw.ssmAlphaQ, lw.ssmAlphaQType, "ssm_alpha.weight")
      uploadWeight(lw.wv, lw.ssmBetaQ, lw.ssmBetaQType, "ssm_beta.weight")
      lw.ssmConv1dW = loadF32Ptr(lp & "ssm_conv1d.weight")
      if m.infos.hasKey(lp & "ssm_conv1d.bias"):
        lw.ssmConv1dBias = loadF32Ptr(lp & "ssm_conv1d.bias")
      lw.ssmDtBias = loadF32Ptr(lp & "ssm_dt.bias")
      lw.ssmA = loadF32Ptr(lp & "ssm_a")
      lw.ssmNorm = loadF32Ptr(lp & "ssm_norm.weight")
      # Output projection: ssmInnerSize → nEmb
      uploadWeight(lw.wGate, lw.ssmOutQ, lw.ssmOutQType, "ssm_out.weight")
      # MoE weights (same as lkAttnMoe)
      lw.moeRouterW = loadF32Ptr(lp & "ffn_gate_inp.weight")
      block:
        let gtn = lp & "ffn_gate_exps.weight"
        let gInfo = m.infos[gtn]
        lw.moeGateExpsQ = cachedQuantWeight(gtn, m, gtn).devicePtr
        lw.moeGateExpsQType = gInfo.elemType
        lw.moeGateExpsSliceBytes = quantRowSize(int(gInfo.ne[0]), gInfo.elemType) * int(gInfo.ne[1])
        let utn = lp & "ffn_up_exps.weight"
        let uInfo = m.infos[utn]
        lw.moeUpExpsQ = cachedQuantWeight(utn, m, utn).devicePtr
        lw.moeUpExpsQType = uInfo.elemType
        lw.moeUpExpsSliceBytes = quantRowSize(int(uInfo.ne[0]), uInfo.elemType) * int(uInfo.ne[1])
        let dtn = lp & "ffn_down_exps.weight"
        let dInfo = m.infos[dtn]
        lw.moeDownExpsQ = cachedQuantWeight(dtn, m, dtn).devicePtr
        lw.moeDownExpsQType = dInfo.elemType
        lw.moeDownExpsSliceBytes = quantRowSize(int(dInfo.ne[0]), dInfo.elemType) * int(dInfo.ne[1])
      uploadWeight(lw.wGate, lw.moeShGateQ, lw.moeShGateQType, "ffn_gate_shexp.weight")
      uploadWeight(lw.wUp, lw.moeShUpQ, lw.moeShUpQType, "ffn_up_shexp.weight")
      uploadWeight(lw.wDown, lw.moeShDownQ, lw.moeShDownQType, "ffn_down_shexp.weight")
      lw.moeShGateScalar = loadF32Ptr(lp & "ffn_gate_inp_shexp.weight")

    modelPtrs.layers[layer] = lw

  let normName = if m.infos.hasKey("output_norm.weight"): "output_norm.weight"
                 else: "norm.weight"
  modelPtrs.normWeight = loadF32Ptr(normName)

  let outTensorName = if m.infos.hasKey("output.weight"): "output.weight"
                      else: "token_embd.weight"
  let outElemType = m.infos[outTensorName].elemType.int32
  if outElemType == GgmlTypeF16.int32 or
     outElemType == GgmlTypeQ5_0.int32 or
     outElemType == GgmlTypeQ2K.int32 or outElemType == GgmlTypeQ3K.int32 or
     outElemType == GgmlTypeQ4K.int32 or outElemType == GgmlTypeQ6K.int32 or
     outElemType == GgmlTypeQ8_0.int32 or
     outElemType == GgmlTypeQ5_1.int32 or
     outElemType == GgmlTypeQ5K.int32 or
     outElemType == GgmlTypeIQ4NL.int32:
    let qw = cachedQuantWeight("output.weight", m, outTensorName)
    modelPtrs.outputWeightQ = qw.devicePtr
    modelPtrs.outputWeight = nil
    modelPtrs.outputQType = outElemType
    modelPtrs.outputShape0 = hp.nEmb
    modelPtrs.outputShape1 = hp.nVocab
  else:
    let outW = m.getTensor(outTensorName)
    let a0 = outW.shape[0]
    let a1 = outW.shape[1]
    if a0 == hp.nEmb and a1 == hp.nVocab:
      modelPtrs.outputWeight = cachedWeight("output.weight", outW).devicePtr
      modelPtrs.outputShape0 = a0
      modelPtrs.outputShape1 = a1
    elif a0 == hp.nVocab and a1 == hp.nEmb:
      let reshaped = outW.reshape(@[a1, a0])
      modelPtrs.outputWeight = cachedWeight("output.weight", reshaped).devicePtr
      modelPtrs.outputShape0 = a1
      modelPtrs.outputShape1 = a0
    else:
      raise newException(ValueError, "output weight shape mismatch")
    modelPtrs.outputWeightQ = nil
    modelPtrs.outputQType = 0

  if hp.ropeDim > 0 and hp.nHead > 0:
    let ropeDim = hp.ropeDim
    let halfRope = ropeDim div 2
    var thetaBuf = newSeq[float32](halfRope)
    for i in 0 ..< halfRope:
      thetaBuf[i] = pow(1.0'f32 / hp.ropeFreqBase, (2.0'f32 * i.float32) / ropeDim.float32)
    ropeThetaAlloc = hippoMalloc(halfRope * sizeof(float32))
    let stream = gpuCtx.stream
    hippoMemcpyAsync(ropeThetaAlloc.p, addr thetaBuf[0], halfRope * sizeof(float32),
                      HippoMemcpyHostToDevice, stream)
    gpuStreamSync(stream)
    modelPtrs.ropeTheta = ropeThetaAlloc.p
  modelPtrs.initialized = true

# ---------------------------------------------------------------------------
# Profiling infrastructure (compile with -d:profileHippo)
# ---------------------------------------------------------------------------
when defined(profileHippo):
  import std/[times, strutils]

  type
    KernelCategory* = enum
      KcEmbedding, KcRmsNormAttn, KcLinearQkv, KcRope, KcKvStore,
      KcAttention, KcLinearO, KcResidualAttn, KcRmsNormFfn,
      KcLinearGateUp, KcSiluMul, KcLinearDown, KcResidualFfn,
      KcFinalNormOutput

    EventPair* = object
      start*: HippoEvent
      stop*: HippoEvent
      cat*: KernelCategory

  proc recordStart(pairs: var seq[EventPair], cat: KernelCategory, stream: HippoStream) =
    var ep: EventPair
    ep.cat = cat
    ep.start = hippoEventCreate()
    hippoEventRecord(ep.start, stream)
    pairs.add(ep)

  proc recordStop(pairs: var seq[EventPair], stream: HippoStream) =
    pairs[^1].stop = hippoEventCreate()
    hippoEventRecord(pairs[^1].stop, stream)

  proc printBreakdown(pairs: seq[EventPair]) =
    var catMs: array[KernelCategory, float32]
    var totalMs: float32 = 0
    for ep in pairs:
      let ms = hippoEventElapsedTime(ep.start, ep.stop)
      catMs[ep.cat] += ms
      totalMs += ms
      hippoEventDestroy(ep.start)
      hippoEventDestroy(ep.stop)
    echo "  GPU kernel breakdown (total=", totalMs.formatFloat(ffDecimal, 2), "ms):"
    for cat in KernelCategory:
      if catMs[cat] > 0:
        let pct = if totalMs > 0: catMs[cat] / totalMs * 100 else: 0.0
        echo "    ", ($cat).alignLeft(20), catMs[cat].formatFloat(ffDecimal, 3), " ms  ",
             pct.formatFloat(ffDecimal, 1), "%"

# ---------------------------------------------------------------------------
# Backend interface: initKvCache
# ---------------------------------------------------------------------------
proc initKvCache*(hp: HParams, maxLen: int): KvCache =
  if hp.nHead <= 0 and hp.ssmInnerSize <= 0:
    raise newException(ValueError, "model requires head_count or SSM parameters")
  result.maxLen = maxLen
  result.curLen = 0
  result.nHeadKv = hp.nHeadKv
  result.headDim = hp.headDim
  result.k = newSeq[Tensor](hp.nLayer)
  result.v = newSeq[Tensor](hp.nLayer)
  for i in 0 ..< hp.nLayer:
    let lkv = if hp.layerNHeadKv.len > i and hp.layerNHeadKv[i] > 0: hp.layerNHeadKv[i]
              else: hp.nHeadKv
    if lkv > 0:
      let kvDim = lkv * result.headDim
      result.k[i] = newTensor(@[kvDim, maxLen])
      result.v[i] = newTensor(@[kvDim, maxLen])
  result.gpuCache = initGpuKvCache(hp.nLayer, hp.nHeadKv, result.headDim, maxLen,
                                    hp.layerNHeadKv)
  if hp.ssmInnerSize > 0:
    result.ssmState = initGpuSsmState(hp, hp.nLayer)

# ---------------------------------------------------------------------------
# Backend interface: helpers
# ---------------------------------------------------------------------------
proc getTensorOr(m: var Model, a, b: string): Tensor =
  try:
    return m.getTensor(a)
  except KeyError:
    return m.getTensor(b)

proc outputWeightForLinear(w: Tensor, nEmb, nVocab: int): Tensor =
  if w.shape.len != 2:
    raise newException(ValueError, "output weight must be 2D")
  let a0 = w.shape[0]
  let a1 = w.shape[1]
  if a0 == nEmb and a1 == nVocab:
    return w
  if a0 == nVocab and a1 == nEmb:
    return w.reshape(@[a1, a0])
  raise newException(ValueError, "output weight shape mismatch")

# ---------------------------------------------------------------------------
# Backend interface: loadModelBackend
# ---------------------------------------------------------------------------
proc loadModelBackend*(m: var Model, hp: HParams) =
  ## Upload all model weights to GPU and prepare cached pointers.
  ensureModelGpuPtrs(m, hp)

proc unloadModelBackend*() =
  ## Cleanup GPU state (currently minimal).
  discard

# ---------------------------------------------------------------------------
# Backend interface: forwardPrefill
# ---------------------------------------------------------------------------
proc forwardPrefill*(m: var Model, tokens: seq[int32], cache: var KvCache): Tensor =
  let hp = m.hparams
  if hp.arch != "" and hp.arch notin ["llama", "qwen3", "nemotron_h", "nemotron_h_moe", "qwen35moe"]:
    raise newException(ValueError, "unsupported architecture: " & hp.arch)
  if hp.nHeadKv != 0 and hp.nHead > 0 and (hp.nHead mod hp.nHeadKv) != 0:
    raise newException(ValueError, "GQA requires head_count divisible by head_count_kv")
  if tokens.len == 0:
    raise newException(ValueError, "prefill requires at least one token")
  if tokens.len > cache.maxLen:
    raise newException(ValueError, "prefill exceeds KV cache capacity")

  let headDim = hp.headDim
  let ropeDim = if hp.ropeDim > 0: hp.ropeDim else: headDim
  let qDim = hp.nHead * headDim
  let kvDim = hp.nHeadKv * headDim
  let seqLen = tokens.len
  let ssmProjDim = if hp.ssmInnerSize > 0:
    hp.ssmInnerSize * 2 + 2 * hp.ssmGroupCount * hp.ssmStateSize + hp.ssmDtRank
  else: 0

  ensureGpuContext()
  var maxRows = max(max(max(hp.nEmb, hp.nFfn), hp.nVocab), qDim)
  maxRows = max(maxRows, ssmProjDim)
  for i in 0 ..< hp.layerNFfn.len:
    maxRows = max(maxRows, hp.layerNFfn[i])
  if hp.nExperts > 0:
    maxRows = max(maxRows, hp.nExperts)
    maxRows = max(maxRows, qDim + 2 * kvDim)
    maxRows = max(maxRows, hp.sharedExpertFfnDim)
  ensureActivationBuffers(maxRows * seqLen)
  ensureScratchBuffers(maxRows * seqLen)
  let stream = gpuCtx.stream

  ensureModelGpuPtrs(m, hp)

  let tokEmb = getTensorOr(m, "tok_embeddings.weight", "token_embd.weight")
  let dTokEmb = cachedWeight("token_embd_or_tok_embeddings", tokEmb)
  let tokenPtr = gpuUploadInt32Pooled(unsafeAddr tokens[0], seqLen, stream)

  var xPtr = gpuCtx.act0.devicePtr
  let xNormPtr = gpuCtx.act1.devicePtr
  let tmp0 = gpuCtx.scratch0.devicePtr
  let tmp1 = gpuCtx.scratch1.devicePtr
  let tmp2 = gpuCtx.scratch2.devicePtr
  let tmp3 = gpuCtx.scratch3.devicePtr

  gpuEmbedding(xPtr, dTokEmb.devicePtr, cast[ptr int32](tokenPtr),
               hp.nEmb, seqLen, hp.nVocab, stream)

  template loadF32Weight(name: string): GpuTensor =
    if m.infos[name].elemType == GgmlTypeF16:
      cachedF16WeightAsF32(name, m, name)
    else:
      cachedWeight(name, m.getTensor(name))

  for layer in 0 ..< hp.nLayer:
    let lp = "blk." & $layer & "."
    let lw = modelPtrs.layers[layer]
    let dAttnNorm = loadF32Weight(lp & "attn_norm.weight")

    when defined(traceNormWeights):
      if layer <= 3:
        block:
          var buf = newSeq[float32](hp.nEmb)
          gpuDownloadFromDevice(addr buf[0], dAttnNorm.devicePtr, hp.nEmb * sizeof(float32), stream)
          gpuStreamSync(stream)
          echo &"  L{layer} attn_norm first5={buf[0..4]}"

    when defined(traceGdn):
      block:
        let totalElems = hp.nEmb * seqLen
        var xBuf = newSeq[float32](totalElems)
        gpuDownloadFromDevice(addr xBuf[0], xPtr, totalElems * sizeof(float32), stream)
        gpuStreamSync(stream)
        let lastCol = seqLen - 1
        var ss = 0.0'f64
        var nNan, nInf = 0
        for ii in 0 ..< hp.nEmb:
          let v = xBuf[ii * seqLen + lastCol]
          if v != v: inc nNan
          elif v == Inf or v == -Inf: inc nInf
          else: ss += float64(v) * float64(v)
        echo "Layer ", layer, " (", lw.kind, ") x_rms=",
          formatFloat(math.sqrt(ss / float64(hp.nEmb)), ffDecimal, 6),
          " nan=", nNan, " inf=", nInf

    case lw.kind
    of lkAttentionFfn:
      let dFfnNorm = loadF32Weight(lp & "ffn_norm.weight")
      let dWq = loadF32Weight(lp & "attn_q.weight")
      let dWk = loadF32Weight(lp & "attn_k.weight")
      let dWv = loadF32Weight(lp & "attn_v.weight")
      let dWo = loadF32Weight(lp & "attn_output.weight")
      let dWGate = loadF32Weight(lp & "ffn_gate.weight")
      let dWUp = loadF32Weight(lp & "ffn_up.weight")
      let dWDown = loadF32Weight(lp & "ffn_down.weight")
      gpuRmsnormCols(xNormPtr, xPtr, dAttnNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
      gpuLinearCol(tmp0, xNormPtr, dWq.devicePtr, hp.nEmb, qDim, seqLen, stream)
      gpuLinearCol(tmp1, xNormPtr, dWk.devicePtr, hp.nEmb, kvDim, seqLen, stream)
      gpuLinearCol(tmp2, xNormPtr, dWv.devicePtr, hp.nEmb, kvDim, seqLen, stream)
      if m.infos.hasKey(lp & "attn_q_norm.weight"):
        let dQNorm = loadF32Weight(lp & "attn_q_norm.weight")
        gpuQkNormPrefill(tmp0, dQNorm.devicePtr, hp.nHead, headDim, seqLen, hp.rmsEps, stream)
      if m.infos.hasKey(lp & "attn_k_norm.weight"):
        let dKNorm = loadF32Weight(lp & "attn_k_norm.weight")
        gpuQkNormPrefill(tmp1, dKNorm.devicePtr, hp.nHeadKv, headDim, seqLen, hp.rmsEps, stream)
      gpuRopeAtPos(tmp0, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, 0, seqLen, stream)
      gpuRopeAtPos(tmp1, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, 0, seqLen, stream)
      gpuStoreKV(cache.gpuCache.k[layer].devicePtr, tmp1, kvDim, seqLen, cache.gpuCache.maxLen, 0, stream)
      gpuStoreKV(cache.gpuCache.v[layer].devicePtr, tmp2, kvDim, seqLen, cache.gpuCache.maxLen, 0, stream)
      gpuAttentionPrefill(xNormPtr, tmp0, tmp1, tmp2,
                          hp.nHead, hp.nHeadKv, headDim, seqLen, stream)
      gpuLinearCol(tmp0, xNormPtr, dWo.devicePtr, qDim, hp.nEmb, seqLen, stream)
      gpuAdd(xPtr, xPtr, tmp0, hp.nEmb * seqLen, stream)
      gpuRmsnormCols(xNormPtr, xPtr, dFfnNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
      gpuLinearCol(tmp0, xNormPtr, dWGate.devicePtr, hp.nEmb, hp.nFfn, seqLen, stream)
      gpuLinearCol(tmp1, xNormPtr, dWUp.devicePtr, hp.nEmb, hp.nFfn, seqLen, stream)
      gpuSiluMul(tmp2, tmp0, tmp1, hp.nFfn * seqLen, stream)
      gpuLinearCol(tmp0, tmp2, dWDown.devicePtr, hp.nFfn, hp.nEmb, seqLen, stream)
      gpuAdd(xPtr, xPtr, tmp0, hp.nEmb * seqLen, stream)

    of lkAttention:
      let lkvDim = lw.layerNHeadKv * headDim
      let lqDim = hp.nHead * headDim
      let dWq = loadF32Weight(lp & "attn_q.weight")
      let dWk = loadF32Weight(lp & "attn_k.weight")
      let dWv = loadF32Weight(lp & "attn_v.weight")
      let dWo = loadF32Weight(lp & "attn_output.weight")
      gpuRmsnormCols(xNormPtr, xPtr, dAttnNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
      gpuLinearCol(tmp0, xNormPtr, dWq.devicePtr, hp.nEmb, lqDim, seqLen, stream)
      gpuLinearCol(tmp1, xNormPtr, dWk.devicePtr, hp.nEmb, lkvDim, seqLen, stream)
      gpuLinearCol(tmp2, xNormPtr, dWv.devicePtr, hp.nEmb, lkvDim, seqLen, stream)
      gpuRopeAtPos(tmp0, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, 0, seqLen, stream)
      gpuRopeAtPos(tmp1, lw.layerNHeadKv, headDim, ropeDim, hp.ropeFreqBase, 0, seqLen, stream)
      gpuStoreKV(cache.gpuCache.k[layer].devicePtr, tmp1, lkvDim, seqLen, cache.gpuCache.maxLen, 0, stream)
      gpuStoreKV(cache.gpuCache.v[layer].devicePtr, tmp2, lkvDim, seqLen, cache.gpuCache.maxLen, 0, stream)
      gpuAttentionPrefill(xNormPtr, tmp0, tmp1, tmp2,
                          hp.nHead, lw.layerNHeadKv, headDim, seqLen, stream)
      gpuLinearCol(tmp0, xNormPtr, dWo.devicePtr, lqDim, hp.nEmb, seqLen, stream)
      gpuAdd(xPtr, xPtr, tmp0, hp.nEmb * seqLen, stream)

    of lkFfnOnly:
      let lnFfn = lw.layerNFfn
      let dWUp = loadF32Weight(lp & "ffn_up.weight")
      let dWDown = loadF32Weight(lp & "ffn_down.weight")
      gpuRmsnormCols(xNormPtr, xPtr, dAttnNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
      gpuLinearCol(tmp0, xNormPtr, dWUp.devicePtr, hp.nEmb, lnFfn, seqLen, stream)
      gpuReluSqr(tmp0, lnFfn * seqLen, stream)
      gpuLinearCol(tmp0, tmp0, dWDown.devicePtr, lnFfn, hp.nEmb, seqLen, stream)
      gpuAdd(xPtr, xPtr, tmp0, hp.nEmb * seqLen, stream)

    of lkMoeFfn:
      gpuRmsnormCols(xNormPtr, xPtr, dAttnNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
      let shFfn = hp.sharedExpertFfnDim
      for t in 0 ..< seqLen:
        gpuGatherCol(tmp2, xNormPtr, hp.nEmb, seqLen, t, stream)
        gpuLinearCol(tmp0, tmp2, lw.moeRouterW, hp.nEmb, hp.nExperts, 1, stream)
        if lw.moeExpertBias != nil:
          gpuAdd(tmp0, tmp0, lw.moeExpertBias, hp.nExperts, stream)
        let eiPtr = tmp3
        let ewPtr = cast[pointer](cast[uint](tmp3) + uint(hp.nExpertsUsed * sizeof(int32)))
        gpuMoeTopK(eiPtr, ewPtr, tmp0, hp.nExperts, hp.nExpertsUsed, stream)
        var expertIndices: array[32, int32]
        var expertWeights: array[32, float32]
        gpuDownloadFromDevice(addr expertIndices[0], eiPtr, hp.nExpertsUsed * sizeof(int32), stream)
        gpuDownloadFromDevice(addr expertWeights[0], ewPtr, hp.nExpertsUsed * sizeof(float32), stream)
        gpuStreamSync(stream)
        for e in 0 ..< hp.nExpertsUsed:
          expertWeights[e] *= hp.expertWeightsScale
        gpuZeroBuffer(tmp1, hp.nEmb, stream)
        for e in 0 ..< hp.nExpertsUsed:
          let eidx = int(expertIndices[e])
          let upOff = cast[pointer](cast[uint](lw.moeUpExpsQ) + uint(eidx * lw.moeUpExpsSliceBytes))
          gpuLinearColQuant(tmp0, tmp2, upOff, hp.nEmb, hp.expertFfnDim, lw.moeUpExpsQType, stream)
          gpuReluSqr(tmp0, hp.expertFfnDim, stream)
          let downOff = cast[pointer](cast[uint](lw.moeDownExpsQ) + uint(eidx * lw.moeDownExpsSliceBytes))
          gpuLinearColQuant(tmp3, tmp0, downOff, hp.expertFfnDim, hp.nEmb, lw.moeDownExpsQType, stream)
          gpuScaleAdd(tmp1, tmp3, expertWeights[e], hp.nEmb, stream)
        if lw.moeShUpQ != nil:
          gpuLinearColQuant(tmp0, tmp2, lw.moeShUpQ, hp.nEmb, shFfn, lw.moeShUpQType, stream)
        else:
          gpuLinearCol(tmp0, tmp2, lw.wUp, hp.nEmb, shFfn, 1, stream)
        gpuReluSqr(tmp0, shFfn, stream)
        if lw.moeShDownQ != nil:
          gpuLinearColQuant(tmp3, tmp0, lw.moeShDownQ, shFfn, hp.nEmb, lw.moeShDownQType, stream)
        else:
          gpuLinearCol(tmp3, tmp0, lw.wDown, shFfn, hp.nEmb, 1, stream)
        gpuAdd(tmp0, tmp1, tmp3, hp.nEmb, stream)
        gpuScatterAddCol(xPtr, tmp0, hp.nEmb, seqLen, t, stream)
      continue

    of lkSsm:
      # SSM prefill: process each token sequentially through the recurrence
      let ssmInner = hp.ssmInnerSize
      let nGroups = hp.ssmGroupCount
      let stateSize = hp.ssmStateSize
      let nHeads = hp.ssmDtRank
      let ssmHeadDim = ssmInner div nHeads
      let headsPerGroup = nHeads div nGroups
      let groupSize = headsPerGroup * ssmHeadDim
      let convDim = ssmInner + 2 * nGroups * stateSize
      let ssmLayerIdx = cache.ssmState.ssmLayerMap[layer]
      gpuRmsnormCols(xNormPtr, xPtr, dAttnNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
      # Process each token position sequentially
      for t in 0 ..< seqLen:
        let xColPtr = cast[pointer](cast[uint](xNormPtr) + uint(t * sizeof(float32)))
        # For column-major [nEmb, seqLen], element (row, col) = row * seqLen + col
        # We need to extract column t. Use tmp2 as single-token staging.
        # Copy column t from xNormPtr to a contiguous buffer
        # Actually for column-major: xNormPtr[row * seqLen + t] for row 0..nEmb-1
        # We need a gather kernel, or just use the decode path via ensureModelGpuPtrs
        # Simpler: use the already-uploaded weights in modelPtrs
        if lw.ssmInQ != nil:
          # For prefill we need to handle column-major layout. Use the decode GEMV which expects
          # contiguous input. We'll gather column t into tmp2, then project.
          # gather column: for row in 0..nEmb-1: tmp2[row] = xNormPtr[row*seqLen + t]
          gpuGatherCol(tmp2, xNormPtr, hp.nEmb, seqLen, t, stream)
          gpuLinearColQuant(tmp0, tmp2, lw.ssmInQ, hp.nEmb, ssmProjDim, lw.ssmInQType, stream)
        else:
          gpuGatherCol(tmp2, xNormPtr, hp.nEmb, seqLen, t, stream)
          gpuLinearCol(tmp0, tmp2, lw.wUp, hp.nEmb, ssmProjDim, 1, stream)
        let xBcDtPtr = cast[pointer](cast[uint](tmp0) + uint(ssmInner * sizeof(float32)))
        gpuConv1dDecode(cache.ssmState.convState[ssmLayerIdx].devicePtr,
                         xBcDtPtr, lw.ssmConv1dW, lw.ssmConv1dBias, tmp1,
                         convDim, hp.ssmConvKernel, stream)
        gpuSilu(tmp1, convDim, stream)
        let bPtr = cast[pointer](cast[uint](tmp1) + uint(ssmInner * sizeof(float32)))
        let cPtr = cast[pointer](cast[uint](bPtr) + uint(nGroups * stateSize * sizeof(float32)))
        let dtPtr = cast[pointer](cast[uint](xBcDtPtr) + uint(convDim * sizeof(float32)))
        gpuSsmScanDecode(cache.ssmState.recState[ssmLayerIdx].devicePtr,
                          tmp1, bPtr, cPtr, dtPtr, lw.ssmDtBias, lw.ssmA, lw.ssmD,
                          tmp2, nHeads, headsPerGroup, ssmHeadDim, stateSize, stream)
        gpuSilu(tmp0, ssmInner, stream)
        gpuElemMul(tmp2, tmp0, ssmInner, stream)
        gpuGroupRmsNorm(tmp2, lw.ssmNorm, nGroups, groupSize, hp.rmsEps, stream)
        if lw.ssmOutQ != nil:
          gpuLinearColQuant(tmp0, tmp2, lw.ssmOutQ, ssmInner, hp.nEmb, lw.ssmOutQType, stream)
        else:
          gpuLinearCol(tmp0, tmp2, lw.wDown, ssmInner, hp.nEmb, 1, stream)
        # Scatter result back: for row in 0..nEmb-1: xPtr[row*seqLen + t] += tmp0[row]
        gpuScatterAddCol(xPtr, tmp0, hp.nEmb, seqLen, t, stream)
      continue  # skip the gpuAdd below since we scattered per-token

    of lkAttnMoe:
      gpuRmsnormCols(xNormPtr, xPtr, dAttnNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
      let dWq = loadF32Weight(lp & "attn_q.weight")
      let dWk = loadF32Weight(lp & "attn_k.weight")
      let dWv = loadF32Weight(lp & "attn_v.weight")
      let dWo = loadF32Weight(lp & "attn_output.weight")
      # Q projection outputs Q+gate interleaved
      let qFullDim = 2 * qDim
      gpuLinearCol(tmp2, xNormPtr, dWq.devicePtr, hp.nEmb, qFullDim, seqLen, stream)
      gpuDeinterleaveQGatePrefill(tmp0, tmp3, tmp2, hp.nHead, headDim, seqLen, stream)
      gpuLinearCol(tmp1, xNormPtr, dWk.devicePtr, hp.nEmb, kvDim, seqLen, stream)
      gpuLinearCol(tmp2, xNormPtr, dWv.devicePtr, hp.nEmb, kvDim, seqLen, stream)
      if m.infos.hasKey(lp & "attn_q_norm.weight"):
        let dQNorm = loadF32Weight(lp & "attn_q_norm.weight")
        gpuQkNormPrefill(tmp0, dQNorm.devicePtr, hp.nHead, headDim, seqLen, hp.rmsEps, stream)
      if m.infos.hasKey(lp & "attn_k_norm.weight"):
        let dKNorm = loadF32Weight(lp & "attn_k_norm.weight")
        gpuQkNormPrefill(tmp1, dKNorm.devicePtr, hp.nHeadKv, headDim, seqLen, hp.rmsEps, stream)
      gpuRopeAtPosTheta(tmp0, hp.nHead, headDim, ropeDim, 0, seqLen, stream)
      gpuRopeAtPosTheta(tmp1, hp.nHeadKv, headDim, ropeDim, 0, seqLen, stream)
      gpuStoreKV(cache.gpuCache.k[layer].devicePtr, tmp1, kvDim, seqLen, cache.gpuCache.maxLen, 0, stream)
      gpuStoreKV(cache.gpuCache.v[layer].devicePtr, tmp2, kvDim, seqLen, cache.gpuCache.maxLen, 0, stream)
      gpuAttentionPrefill(xNormPtr, tmp0, tmp1, tmp2,
                          hp.nHead, hp.nHeadKv, headDim, seqLen, stream)
      when defined(traceAttn):
        if layer == 3:
          block:
            var buf = newSeq[float32](qDim * seqLen)
            gpuDownloadFromDevice(addr buf[0], xNormPtr, qDim * seqLen * sizeof(float32), stream)
            gpuStreamSync(stream)
            var ss = 0.0'f64
            for ii in 0 ..< qDim:
              let v = float64(buf[ii * seqLen + seqLen - 1])
              ss += v * v
            echo &"  L3 attn_out rms={math.sqrt(ss / float64(qDim)):.6f}"
            var gateBuf = newSeq[float32](qDim * seqLen)
            gpuDownloadFromDevice(addr gateBuf[0], tmp3, qDim * seqLen * sizeof(float32), stream)
            gpuStreamSync(stream)
            var gss = 0.0'f64
            var sigMin = 1.0'f64
            var sigMax = 0.0'f64
            for ii in 0 ..< qDim:
              let gv = float64(gateBuf[ii * seqLen + seqLen - 1])
              let sv = 1.0'f64 / (1.0'f64 + math.exp(-gv))
              gss += sv * sv
              if sv < sigMin: sigMin = sv
              if sv > sigMax: sigMax = sv
            echo &"  L3 gate_sigmoid rms={math.sqrt(gss / float64(qDim)):.6f} min={sigMin:.6f} max={sigMax:.6f}"
      # Apply attention gate
      gpuAttnGate(xNormPtr, tmp3, qDim * seqLen, stream)
      gpuLinearCol(tmp0, xNormPtr, dWo.devicePtr, qDim, hp.nEmb, seqLen, stream)
      when defined(traceAttn):
        if layer == 3:
          block:
            var buf = newSeq[float32](hp.nEmb * seqLen)
            gpuDownloadFromDevice(addr buf[0], tmp0, hp.nEmb * seqLen * sizeof(float32), stream)
            gpuStreamSync(stream)
            var ss = 0.0'f64
            for ii in 0 ..< hp.nEmb:
              let v = float64(buf[ii * seqLen + seqLen - 1])
              ss += v * v
            echo &"  L3 attn_residual rms={math.sqrt(ss / float64(hp.nEmb)):.6f}"
      gpuAdd(xPtr, xPtr, tmp0, hp.nEmb * seqLen, stream)
      # Post-attention norm
      let dPostNorm = loadF32Weight(lp & "post_attention_norm.weight")
      gpuRmsnormCols(xNormPtr, xPtr, dPostNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
      # MoE FFN per-token
      for t in 0 ..< seqLen:
        gpuGatherCol(tmp2, xNormPtr, hp.nEmb, seqLen, t, stream)
        gpuLinearCol(tmp0, tmp2, lw.moeRouterW, hp.nEmb, hp.nExperts, 1, stream)
        let eiPtr = tmp3
        let ewPtr = cast[pointer](cast[uint](tmp3) + uint(hp.nExpertsUsed * sizeof(int32)))
        gpuMoeTopK(eiPtr, ewPtr, tmp0, hp.nExperts, hp.nExpertsUsed, stream)
        var expertIndices: array[32, int32]
        var expertWeights: array[32, float32]
        gpuDownloadFromDevice(addr expertIndices[0], eiPtr, hp.nExpertsUsed * sizeof(int32), stream)
        gpuDownloadFromDevice(addr expertWeights[0], ewPtr, hp.nExpertsUsed * sizeof(float32), stream)
        gpuStreamSync(stream)
        gpuZeroBuffer(tmp1, hp.nEmb, stream)
        for e in 0 ..< hp.nExpertsUsed:
          let eidx = int(expertIndices[e])
          let gateOff = cast[pointer](cast[uint](lw.moeGateExpsQ) + uint(eidx * lw.moeGateExpsSliceBytes))
          gpuLinearColQuant(tmp0, tmp2, gateOff, hp.nEmb, hp.expertFfnDim, lw.moeGateExpsQType, stream)
          let upOff = cast[pointer](cast[uint](lw.moeUpExpsQ) + uint(eidx * lw.moeUpExpsSliceBytes))
          gpuLinearColQuant(tmp3, tmp2, upOff, hp.nEmb, hp.expertFfnDim, lw.moeUpExpsQType, stream)
          gpuSiluMul(tmp0, tmp0, tmp3, hp.expertFfnDim, stream)
          let downOff = cast[pointer](cast[uint](lw.moeDownExpsQ) + uint(eidx * lw.moeDownExpsSliceBytes))
          gpuLinearColQuant(tmp3, tmp0, downOff, hp.expertFfnDim, hp.nEmb, lw.moeDownExpsQType, stream)
          gpuScaleAdd(tmp1, tmp3, expertWeights[e], hp.nEmb, stream)
        if lw.moeShGateQ != nil:
          gpuLinearColQuant(tmp0, tmp2, lw.moeShGateQ, hp.nEmb, hp.expertFfnDim, lw.moeShGateQType, stream)
        else:
          gpuLinearCol(tmp0, tmp2, lw.wGate, hp.nEmb, hp.expertFfnDim, 1, stream)
        if lw.moeShUpQ != nil:
          gpuLinearColQuant(tmp3, tmp2, lw.moeShUpQ, hp.nEmb, hp.expertFfnDim, lw.moeShUpQType, stream)
        else:
          gpuLinearCol(tmp3, tmp2, lw.wUp, hp.nEmb, hp.expertFfnDim, 1, stream)
        gpuSiluMul(tmp0, tmp0, tmp3, hp.expertFfnDim, stream)
        if lw.moeShDownQ != nil:
          gpuLinearColQuant(tmp3, tmp0, lw.moeShDownQ, hp.expertFfnDim, hp.nEmb, lw.moeShDownQType, stream)
        else:
          gpuLinearCol(tmp3, tmp0, lw.wDown, hp.expertFfnDim, hp.nEmb, 1, stream)
        gpuSharedExpertGate(tmp3, tmp2, lw.moeShGateScalar, hp.nEmb, stream)
        gpuAdd(tmp0, tmp1, tmp3, hp.nEmb, stream)
        gpuScatterAddCol(xPtr, tmp0, hp.nEmb, seqLen, t, stream)
      continue

    of lkSsmAttnMoe:
      let nVHeads = hp.ssmDtRank
      let nKHeads = hp.ssmGroupCount
      let headKDim = hp.ssmStateSize
      let headVDim = hp.ssmInnerSize div nVHeads
      let convDim = 2 * nKHeads * headKDim + hp.ssmInnerSize
      let qkDim = nKHeads * headKDim
      let ssmLayerIdx = cache.ssmState.ssmLayerMap[layer]
      when defined(skipGdnAttn):
        block:
          let dPN = loadF32Weight(lp & "post_attention_norm.weight")
          gpuRmsnormCols(xNormPtr, xPtr, dPN.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
        for t in 0 ..< seqLen:
          gpuGatherCol(tmp2, xNormPtr, hp.nEmb, seqLen, t, stream)
          gpuLinearCol(tmp0, tmp2, lw.moeRouterW, hp.nEmb, hp.nExperts, 1, stream)
          let eiPtr = tmp3
          let ewPtr = cast[pointer](cast[uint](tmp3) + uint(hp.nExpertsUsed * sizeof(int32)))
          gpuMoeTopK(eiPtr, ewPtr, tmp0, hp.nExperts, hp.nExpertsUsed, stream)
          var expertIndices: array[32, int32]
          var expertWeights: array[32, float32]
          gpuDownloadFromDevice(addr expertIndices[0], eiPtr, hp.nExpertsUsed * sizeof(int32), stream)
          gpuDownloadFromDevice(addr expertWeights[0], ewPtr, hp.nExpertsUsed * sizeof(float32), stream)
          gpuStreamSync(stream)
          gpuZeroBuffer(tmp1, hp.nEmb, stream)
          for e in 0 ..< hp.nExpertsUsed:
            let eidx = int(expertIndices[e])
            let gateOff = cast[pointer](cast[uint](lw.moeGateExpsQ) + uint(eidx * lw.moeGateExpsSliceBytes))
            gpuLinearColQuant(tmp0, tmp2, gateOff, hp.nEmb, hp.expertFfnDim, lw.moeGateExpsQType, stream)
            let upOff = cast[pointer](cast[uint](lw.moeUpExpsQ) + uint(eidx * lw.moeUpExpsSliceBytes))
            gpuLinearColQuant(tmp3, tmp2, upOff, hp.nEmb, hp.expertFfnDim, lw.moeUpExpsQType, stream)
            gpuSiluMul(tmp0, tmp0, tmp3, hp.expertFfnDim, stream)
            let downOff = cast[pointer](cast[uint](lw.moeDownExpsQ) + uint(eidx * lw.moeDownExpsSliceBytes))
            gpuLinearColQuant(tmp3, tmp0, downOff, hp.expertFfnDim, hp.nEmb, lw.moeDownExpsQType, stream)
            gpuScaleAdd(tmp1, tmp3, expertWeights[e], hp.nEmb, stream)
          if lw.moeShGateQ != nil:
            gpuLinearColQuant(tmp0, tmp2, lw.moeShGateQ, hp.nEmb, hp.expertFfnDim, lw.moeShGateQType, stream)
          else:
            gpuLinearCol(tmp0, tmp2, lw.wGate, hp.nEmb, hp.expertFfnDim, 1, stream)
          if lw.moeShUpQ != nil:
            gpuLinearColQuant(tmp3, tmp2, lw.moeShUpQ, hp.nEmb, hp.expertFfnDim, lw.moeShUpQType, stream)
          else:
            gpuLinearCol(tmp3, tmp2, lw.wUp, hp.nEmb, hp.expertFfnDim, 1, stream)
          gpuSiluMul(tmp0, tmp0, tmp3, hp.expertFfnDim, stream)
          if lw.moeShDownQ != nil:
            gpuLinearColQuant(tmp3, tmp0, lw.moeShDownQ, hp.expertFfnDim, hp.nEmb, lw.moeShDownQType, stream)
          else:
            gpuLinearCol(tmp3, tmp0, lw.wDown, hp.expertFfnDim, hp.nEmb, 1, stream)
          gpuSharedExpertGate(tmp3, tmp2, lw.moeShGateScalar, hp.nEmb, stream)
          gpuAdd(tmp0, tmp1, tmp3, hp.nEmb, stream)
          gpuScatterAddCol(xPtr, tmp0, hp.nEmb, seqLen, t, stream)
        continue
      gpuRmsnormCols(xNormPtr, xPtr, dAttnNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
      # Delta Net per-token + post_attn_norm → xPtr, then MoE per-token
      for t in 0 ..< seqLen:
        gpuGatherCol(tmp2, xNormPtr, hp.nEmb, seqLen, t, stream)
        when defined(traceGdn):
          if layer == 0 and t == seqLen - 1:
            block:
              var buf = newSeq[float32](hp.nEmb)
              gpuDownloadFromDevice(addr buf[0], tmp2, hp.nEmb * sizeof(float32), stream)
              gpuStreamSync(stream)
              var ss = 0.0'f64
              for ii in 0 ..< hp.nEmb:
                ss += float64(buf[ii]) * float64(buf[ii])
              echo "  L0 xNorm rms=", formatFloat(math.sqrt(ss / float64(hp.nEmb)), ffDecimal, 6),
                " first5=", buf[0..4]
        # QKV projection
        if lw.wqkvQ != nil:
          gpuLinearColQuant(tmp0, tmp2, lw.wqkvQ, hp.nEmb, convDim, lw.wqkvQType, stream)
        else:
          gpuLinearCol(tmp0, tmp2, lw.wq, hp.nEmb, convDim, 1, stream)
        when defined(traceGdn):
          if layer == 0 and t == seqLen - 1:
            block:
              var buf = newSeq[float32](convDim)
              gpuDownloadFromDevice(addr buf[0], tmp0, convDim * sizeof(float32), stream)
              gpuStreamSync(stream)
              var ss = 0.0'f64
              for ii in 0 ..< convDim:
                ss += float64(buf[ii]) * float64(buf[ii])
              echo "  L0 qkv_proj rms=", formatFloat(math.sqrt(ss / float64(convDim)), ffDecimal, 6),
                " first5=", buf[0..4]
        # Gate (z) projection → tmp1
        if lw.ssmGateQ != nil:
          gpuLinearColQuant(tmp1, tmp2, lw.ssmGateQ, hp.nEmb, hp.ssmInnerSize, lw.ssmGateQType, stream)
        else:
          gpuLinearCol(tmp1, tmp2, lw.wo, hp.nEmb, hp.ssmInnerSize, 1, stream)
        # Alpha/beta
        let alphaPtr = tmp3
        let betaRawPtr = cast[pointer](cast[uint](tmp3) + uint(nVHeads * sizeof(float32)))
        if lw.ssmAlphaQ != nil:
          gpuLinearColQuant(alphaPtr, tmp2, lw.ssmAlphaQ, hp.nEmb, nVHeads, lw.ssmAlphaQType, stream)
        else:
          gpuLinearCol(alphaPtr, tmp2, lw.wk, hp.nEmb, nVHeads, 1, stream)
        if lw.ssmBetaQ != nil:
          gpuLinearColQuant(betaRawPtr, tmp2, lw.ssmBetaQ, hp.nEmb, nVHeads, lw.ssmBetaQType, stream)
        else:
          gpuLinearCol(betaRawPtr, tmp2, lw.wv, hp.nEmb, nVHeads, 1, stream)
        let gateDecayPtr = cast[pointer](cast[uint](tmp3) + uint(2 * nVHeads * sizeof(float32)))
        let betaSigPtr = cast[pointer](cast[uint](tmp3) + uint(3 * nVHeads * sizeof(float32)))
        gpuDeltaNetGate(gateDecayPtr, betaSigPtr, alphaPtr, lw.ssmDtBias, lw.ssmA, betaRawPtr,
                         nVHeads, stream)
        when defined(traceGdn):
          if layer == 0 and t == seqLen - 1:
            block:
              var abuf = newSeq[float32](nVHeads)
              var bbuf = newSeq[float32](nVHeads)
              var gbuf = newSeq[float32](nVHeads)
              var bsbuf = newSeq[float32](nVHeads)
              gpuDownloadFromDevice(addr abuf[0], alphaPtr, nVHeads * sizeof(float32), stream)
              gpuDownloadFromDevice(addr bbuf[0], betaRawPtr, nVHeads * sizeof(float32), stream)
              gpuDownloadFromDevice(addr gbuf[0], gateDecayPtr, nVHeads * sizeof(float32), stream)
              gpuDownloadFromDevice(addr bsbuf[0], betaSigPtr, nVHeads * sizeof(float32), stream)
              gpuStreamSync(stream)
              echo "  L0 alpha[0..3]=", abuf[0..3], " beta_raw[0..3]=", bbuf[0..3]
              echo "  L0 gate[0..3]=", gbuf[0..3], " beta_sig[0..3]=", bsbuf[0..3]
              echo "  L0 exp(gate)[0..3]=", @[math.exp(float64(gbuf[0])), math.exp(float64(gbuf[1])), math.exp(float64(gbuf[2])), math.exp(float64(gbuf[3]))]
        # Conv1d
        gpuConv1dDecode(cache.ssmState.convState[ssmLayerIdx].devicePtr,
                         tmp0, lw.ssmConv1dW, lw.ssmConv1dBias, tmp2,
                         convDim, hp.ssmConvKernel, stream)
        gpuSilu(tmp2, convDim, stream)
        when defined(traceGdn):
          if layer == 0 and t == seqLen - 1:
            block:
              var buf = newSeq[float32](convDim)
              gpuDownloadFromDevice(addr buf[0], tmp2, convDim * sizeof(float32), stream)
              gpuStreamSync(stream)
              var ss = 0.0'f64
              for ii in 0 ..< convDim:
                ss += float64(buf[ii]) * float64(buf[ii])
              echo "  L0 conv+silu rms=", formatFloat(math.sqrt(ss / float64(convDim)), ffDecimal, 6),
                " Q_first5=", buf[0..4], " K_first5=", buf[qkDim..qkDim+4], " V_first5=", buf[2*qkDim..2*qkDim+4]
        # Split Q, K, V from tmp2
        let qConvPtr = tmp2
        let kConvPtr = cast[pointer](cast[uint](tmp2) + uint(qkDim * sizeof(float32)))
        let vConvPtr = cast[pointer](cast[uint](kConvPtr) + uint(qkDim * sizeof(float32)))
        gpuL2NormPerHead(qConvPtr, nKHeads, headKDim, stream)
        gpuL2NormPerHead(kConvPtr, nKHeads, headKDim, stream)
        # Expand Q, K to nVHeads
        let qExpPtr = tmp0
        let kExpPtr = cast[pointer](cast[uint](tmp0) + uint(nVHeads * headKDim * sizeof(float32)))
        gpuExpandHeads(qExpPtr, qConvPtr, nKHeads, nVHeads, headKDim, stream)
        gpuExpandHeads(kExpPtr, kConvPtr, nKHeads, nVHeads, headKDim, stream)
        when defined(traceGdn):
          if layer == 0 and t == seqLen - 1:
            block:
              var qbuf = newSeq[float32](nVHeads * headKDim)
              var kbuf = newSeq[float32](nVHeads * headKDim)
              var vbuf = newSeq[float32](nVHeads * headVDim)
              gpuDownloadFromDevice(addr qbuf[0], qExpPtr, qbuf.len * sizeof(float32), stream)
              gpuDownloadFromDevice(addr kbuf[0], kExpPtr, kbuf.len * sizeof(float32), stream)
              gpuDownloadFromDevice(addr vbuf[0], vConvPtr, vbuf.len * sizeof(float32), stream)
              gpuStreamSync(stream)
              var qss, kss, vss = 0.0'f64
              for ii in 0 ..< qbuf.len: qss += float64(qbuf[ii]) * float64(qbuf[ii])
              for ii in 0 ..< kbuf.len: kss += float64(kbuf[ii]) * float64(kbuf[ii])
              for ii in 0 ..< vbuf.len: vss += float64(vbuf[ii]) * float64(vbuf[ii])
              echo "  L0 Q_exp rms=", formatFloat(math.sqrt(qss / float64(qbuf.len)), ffDecimal, 6),
                " K_exp rms=", formatFloat(math.sqrt(kss / float64(kbuf.len)), ffDecimal, 6),
                " V rms=", formatFloat(math.sqrt(vss / float64(vbuf.len)), ffDecimal, 6)
        # Delta Net recurrence
        let dnOutPtr = tmp2
        gpuDeltaNetDecode(cache.ssmState.recState[ssmLayerIdx].devicePtr,
                           qExpPtr, kExpPtr, vConvPtr, gateDecayPtr, betaSigPtr, dnOutPtr,
                           nVHeads, headKDim, headVDim, stream)
        when defined(traceGdn):
          if layer == 0 and t == seqLen - 1:
            block:
              var buf = newSeq[float32](nVHeads * headVDim)
              gpuDownloadFromDevice(addr buf[0], dnOutPtr, buf.len * sizeof(float32), stream)
              gpuStreamSync(stream)
              var ss = 0.0'f64
              for ii in 0 ..< buf.len:
                ss += float64(buf[ii]) * float64(buf[ii])
              echo "  L0 dn_out rms=", formatFloat(math.sqrt(ss / float64(buf.len)), ffDecimal, 6),
                " first5=", buf[0..4]
        # Gated RMS norm
        gpuGatedRmsNorm(tmp0, dnOutPtr, tmp1, lw.ssmNorm, nVHeads, headVDim, hp.rmsEps, stream)
        when defined(traceGdn):
          if layer == 0 and t == seqLen - 1:
            block:
              var buf = newSeq[float32](hp.ssmInnerSize)
              gpuDownloadFromDevice(addr buf[0], tmp0, buf.len * sizeof(float32), stream)
              gpuStreamSync(stream)
              var ss = 0.0'f64
              for ii in 0 ..< buf.len:
                ss += float64(buf[ii]) * float64(buf[ii])
              echo "  L0 gated_rmsnorm rms=", formatFloat(math.sqrt(ss / float64(buf.len)), ffDecimal, 6),
                " first5=", buf[0..4]
        # Output projection
        if lw.ssmOutQ != nil:
          gpuLinearColQuant(tmp2, tmp0, lw.ssmOutQ, hp.ssmInnerSize, hp.nEmb, lw.ssmOutQType, stream)
        else:
          gpuLinearCol(tmp2, tmp0, lw.wGate, hp.ssmInnerSize, hp.nEmb, 1, stream)
        when defined(traceGdn):
          if layer == 0 and t == seqLen - 1:
            block:
              var buf = newSeq[float32](hp.nEmb)
              gpuDownloadFromDevice(addr buf[0], tmp2, hp.nEmb * sizeof(float32), stream)
              gpuStreamSync(stream)
              var ss = 0.0'f64
              for ii in 0 ..< hp.nEmb:
                ss += float64(buf[ii]) * float64(buf[ii])
              echo "  L0 out_proj rms=", formatFloat(math.sqrt(ss / float64(hp.nEmb)), ffDecimal, 6),
                " first5=", buf[0..4]
        gpuScatterAddCol(xPtr, tmp2, hp.nEmb, seqLen, t, stream)
      # Post-attention norm on full sequence
      let dPostNorm = loadF32Weight(lp & "post_attention_norm.weight")
      gpuRmsnormCols(xNormPtr, xPtr, dPostNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
      # MoE FFN per-token
      for t in 0 ..< seqLen:
        gpuGatherCol(tmp2, xNormPtr, hp.nEmb, seqLen, t, stream)
        gpuLinearCol(tmp0, tmp2, lw.moeRouterW, hp.nEmb, hp.nExperts, 1, stream)
        let eiPtr = tmp3
        let ewPtr = cast[pointer](cast[uint](tmp3) + uint(hp.nExpertsUsed * sizeof(int32)))
        gpuMoeTopK(eiPtr, ewPtr, tmp0, hp.nExperts, hp.nExpertsUsed, stream)
        var expertIndices: array[32, int32]
        var expertWeights: array[32, float32]
        gpuDownloadFromDevice(addr expertIndices[0], eiPtr, hp.nExpertsUsed * sizeof(int32), stream)
        gpuDownloadFromDevice(addr expertWeights[0], ewPtr, hp.nExpertsUsed * sizeof(float32), stream)
        gpuStreamSync(stream)
        gpuZeroBuffer(tmp1, hp.nEmb, stream)
        for e in 0 ..< hp.nExpertsUsed:
          let eidx = int(expertIndices[e])
          let gateOff = cast[pointer](cast[uint](lw.moeGateExpsQ) + uint(eidx * lw.moeGateExpsSliceBytes))
          gpuLinearColQuant(tmp0, tmp2, gateOff, hp.nEmb, hp.expertFfnDim, lw.moeGateExpsQType, stream)
          let upOff = cast[pointer](cast[uint](lw.moeUpExpsQ) + uint(eidx * lw.moeUpExpsSliceBytes))
          gpuLinearColQuant(tmp3, tmp2, upOff, hp.nEmb, hp.expertFfnDim, lw.moeUpExpsQType, stream)
          gpuSiluMul(tmp0, tmp0, tmp3, hp.expertFfnDim, stream)
          let downOff = cast[pointer](cast[uint](lw.moeDownExpsQ) + uint(eidx * lw.moeDownExpsSliceBytes))
          gpuLinearColQuant(tmp3, tmp0, downOff, hp.expertFfnDim, hp.nEmb, lw.moeDownExpsQType, stream)
          gpuScaleAdd(tmp1, tmp3, expertWeights[e], hp.nEmb, stream)
        if lw.moeShGateQ != nil:
          gpuLinearColQuant(tmp0, tmp2, lw.moeShGateQ, hp.nEmb, hp.expertFfnDim, lw.moeShGateQType, stream)
        else:
          gpuLinearCol(tmp0, tmp2, lw.wGate, hp.nEmb, hp.expertFfnDim, 1, stream)
        if lw.moeShUpQ != nil:
          gpuLinearColQuant(tmp3, tmp2, lw.moeShUpQ, hp.nEmb, hp.expertFfnDim, lw.moeShUpQType, stream)
        else:
          gpuLinearCol(tmp3, tmp2, lw.wUp, hp.nEmb, hp.expertFfnDim, 1, stream)
        gpuSiluMul(tmp0, tmp0, tmp3, hp.expertFfnDim, stream)
        if lw.moeShDownQ != nil:
          gpuLinearColQuant(tmp3, tmp0, lw.moeShDownQ, hp.expertFfnDim, hp.nEmb, lw.moeShDownQType, stream)
        else:
          gpuLinearCol(tmp3, tmp0, lw.wDown, hp.expertFfnDim, hp.nEmb, 1, stream)
        gpuSharedExpertGate(tmp3, tmp2, lw.moeShGateScalar, hp.nEmb, stream)
        gpuAdd(tmp0, tmp1, tmp3, hp.nEmb, stream)
        gpuScatterAddCol(xPtr, tmp0, hp.nEmb, seqLen, t, stream)
      continue

  let outTName = if m.infos.hasKey("output.weight"): "output.weight"
                 else: "token_embd.weight"
  let dNorm = loadF32Weight(if m.infos.hasKey("norm.weight"): "norm.weight"
                            else: "output_norm.weight")
  let outW = if m.infos[outTName].elemType == GgmlTypeF16:
    cachedF16WeightAsF32("output.weight", m, outTName)
  else:
    let w = outputWeightForLinear(m.getTensor(outTName), hp.nEmb, hp.nVocab)
    cachedWeight("output.weight", w)

  gpuRmsnormCols(xNormPtr, xPtr, dNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
  gpuLinearCol(xPtr, xNormPtr, outW.devicePtr, outW.shape[0], outW.shape[1], seqLen, stream)

  result = newTensor(@[outW.shape[1], seqLen])
  let bytes = result.data.len * sizeof(float32)
  gpuDownloadFromDevice(addr result.data[0], xPtr, bytes, stream)
  gpuStreamSync(stream)

  cache.curLen = seqLen
  cache.gpuCache.curLen = seqLen

# ---------------------------------------------------------------------------
# Backend interface: forwardDecode
# ---------------------------------------------------------------------------
proc forwardDecode*(m: var Model, token: int32, cache: var KvCache): Tensor =
  let hp = m.hparams
  if hp.arch != "" and hp.arch notin ["llama", "qwen3", "nemotron_h", "nemotron_h_moe", "qwen35moe"]:
    raise newException(ValueError, "unsupported architecture: " & hp.arch)
  if hp.nHeadKv != 0 and hp.nHead > 0 and (hp.nHead mod hp.nHeadKv) != 0:
    raise newException(ValueError, "GQA requires head_count divisible by head_count_kv")
  if cache.curLen >= cache.maxLen:
    raise newException(ValueError, "KV cache full")

  let headDim = hp.headDim
  let ropeDim = if hp.ropeDim > 0: hp.ropeDim else: headDim
  let qDim = hp.nHead * headDim
  let kvDim = hp.nHeadKv * headDim
  let pos = cache.curLen
  let ssmProjDim = if hp.ssmInnerSize > 0:
    hp.ssmInnerSize * 2 + 2 * hp.ssmGroupCount * hp.ssmStateSize + hp.ssmDtRank
  else: 0

  ensureGpuContext()
  var maxRows = max(max(max(hp.nEmb, hp.nFfn), hp.nVocab), qDim)
  maxRows = max(maxRows, ssmProjDim)
  for i in 0 ..< hp.layerNFfn.len:
    maxRows = max(maxRows, hp.layerNFfn[i])
  if hp.nExperts > 0:
    maxRows = max(maxRows, hp.nExperts)
    maxRows = max(maxRows, qDim + 2 * kvDim)
    maxRows = max(maxRows, hp.sharedExpertFfnDim)
  ensureActivationBuffers(maxRows)
  ensureScratchBuffers(maxRows)
  let stream = gpuCtx.stream

  ensureModelGpuPtrs(m, hp)

  when defined(profileHippo):
    let wallStart = epochTime()
    let gpuStartEvt = hippoEventCreate()
    let gpuEndEvt = hippoEventCreate()
    hippoEventRecord(gpuStartEvt, stream)
    var eventPairs: seq[EventPair]

  var tok = token
  let tokenPtr = gpuUploadInt32Pooled(unsafeAddr tok, 1, stream)

  var xPtr = gpuCtx.act0.devicePtr
  let xNormPtr = gpuCtx.act1.devicePtr
  let tmp0 = gpuCtx.scratch0.devicePtr
  let tmp1 = gpuCtx.scratch1.devicePtr
  let tmp2 = gpuCtx.scratch2.devicePtr
  let tmp3 = gpuCtx.scratch3.devicePtr

  when defined(profileHippo):
    recordStart(eventPairs, KcEmbedding, stream)
  gpuEmbedding(xPtr, modelPtrs.tokEmb, cast[ptr int32](tokenPtr),
               hp.nEmb, 1, hp.nVocab, stream)
  when defined(profileHippo):
    recordStop(eventPairs, stream)

  when defined(profileHippo):
    var kernelLaunchMs = 0.0

  when defined(profileHippo):
    var kernelLaunchMs0 = 0.0
    let klStart0 = epochTime()
    recordStart(eventPairs, KcRmsNormAttn, stream)
  gpuRmsnormCols(xNormPtr, xPtr, modelPtrs.layers[0].attnNorm, hp.nEmb, 1, hp.rmsEps, stream)
  when defined(profileHippo):
    recordStop(eventPairs, stream)
    kernelLaunchMs0 += (epochTime() - klStart0) * 1000

  for layer in 0 ..< hp.nLayer:
    let lw = modelPtrs.layers[layer]

    when defined(profileHippo):
      let klStart = epochTime()
      recordStart(eventPairs, KcLinearQkv, stream)

    case lw.kind
    of lkAttentionFfn:
      if lw.wqQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.wqQ, hp.nEmb, qDim, lw.wqQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wq, hp.nEmb, qDim, 1, stream)
      when HippoWarpSize == 32:
        if lw.wkQ != nil and lw.wvQ != nil and lw.wkQType == GgmlTypeQ4K and lw.wvQType == GgmlTypeQ4K:
          gpuFusedKVLinearQ4K(tmp1, tmp2, xNormPtr, lw.wkQ, lw.wvQ, hp.nEmb, kvDim, stream)
        elif lw.wkQ != nil and lw.wvQ != nil and lw.wkQType == GgmlTypeQ2K and lw.wvQType == GgmlTypeQ3K:
          gpuFusedKVLinearQ2KQ3K(tmp1, tmp2, xNormPtr, lw.wkQ, lw.wvQ, hp.nEmb, kvDim, stream)
        else:
          if lw.wkQ != nil:
            gpuLinearColQuant(tmp1, xNormPtr, lw.wkQ, hp.nEmb, kvDim, lw.wkQType, stream)
          else:
            gpuLinearCol(tmp1, xNormPtr, lw.wk, hp.nEmb, kvDim, 1, stream)
          if lw.wvQ != nil:
            gpuLinearColQuant(tmp2, xNormPtr, lw.wvQ, hp.nEmb, kvDim, lw.wvQType, stream)
          else:
            gpuLinearCol(tmp2, xNormPtr, lw.wv, hp.nEmb, kvDim, 1, stream)
      else:
        if lw.wkQ != nil:
          gpuLinearColQuant(tmp1, xNormPtr, lw.wkQ, hp.nEmb, kvDim, lw.wkQType, stream)
        else:
          gpuLinearCol(tmp1, xNormPtr, lw.wk, hp.nEmb, kvDim, 1, stream)
        if lw.wvQ != nil:
          gpuLinearColQuant(tmp2, xNormPtr, lw.wvQ, hp.nEmb, kvDim, lw.wvQType, stream)
        else:
          gpuLinearCol(tmp2, xNormPtr, lw.wv, hp.nEmb, kvDim, 1, stream)
      if lw.attnQNorm != nil:
        gpuQkNorm(tmp0, lw.attnQNorm, hp.nHead, headDim, hp.rmsEps, stream)
      if lw.attnKNorm != nil:
        gpuQkNorm(tmp1, lw.attnKNorm, hp.nHeadKv, headDim, hp.rmsEps, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcRope, stream)
      gpuRopeQKDecode(tmp0, tmp1, hp.nHead, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, pos, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcKvStore, stream)
      gpuStoreKVPair(cache.gpuCache.k[layer].devicePtr, tmp1,
                      cache.gpuCache.v[layer].devicePtr, tmp2,
                      kvDim, 1, cache.gpuCache.maxLen, pos, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcAttention, stream)
      gpuAttentionDecode(xNormPtr, tmp0, cache.gpuCache.k[layer].devicePtr,
                         cache.gpuCache.v[layer].devicePtr,
                         hp.nHead, hp.nHeadKv, headDim, pos + 1,
                         cache.gpuCache.maxLen, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcLinearO, stream)
      if lw.woQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.woQ, qDim, hp.nEmb, lw.woQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wo, qDim, hp.nEmb, 1, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcResidualAttn, stream)
      gpuResidualRmsnorm(xNormPtr, xPtr, tmp0, lw.ffnNorm, hp.nEmb, hp.rmsEps, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcLinearGateUp, stream)
      when HippoWarpSize == 32:
        if lw.wGateQType == GgmlTypeQ4K and lw.wUpQType == GgmlTypeQ4K:
          gpuFusedGateUpSiluQ4K(tmp2, xNormPtr, lw.wGateQ, lw.wUpQ,
                                hp.nEmb, hp.nFfn, stream)
        elif lw.wGateQType == GgmlTypeQ3K and lw.wUpQType == GgmlTypeQ3K:
          gpuFusedGateUpSiluQ3K(tmp2, xNormPtr, lw.wGateQ, lw.wUpQ,
                                hp.nEmb, hp.nFfn, stream)
        else:
          if lw.wGateQ != nil:
            gpuLinearColQuant(tmp0, xNormPtr, lw.wGateQ, hp.nEmb, hp.nFfn, lw.wGateQType, stream)
          else:
            gpuLinearCol(tmp0, xNormPtr, lw.wGate, hp.nEmb, hp.nFfn, 1, stream)
          if lw.wUpQ != nil:
            gpuLinearColQuant(tmp1, xNormPtr, lw.wUpQ, hp.nEmb, hp.nFfn, lw.wUpQType, stream)
          else:
            gpuLinearCol(tmp1, xNormPtr, lw.wUp, hp.nEmb, hp.nFfn, 1, stream)
          when defined(profileHippo):
            recordStop(eventPairs, stream)
            recordStart(eventPairs, KcSiluMul, stream)
          gpuSiluMul(tmp2, tmp0, tmp1, hp.nFfn, stream)
      else:
        if lw.wGateQ != nil:
          gpuLinearColQuant(tmp0, xNormPtr, lw.wGateQ, hp.nEmb, hp.nFfn, lw.wGateQType, stream)
        else:
          gpuLinearCol(tmp0, xNormPtr, lw.wGate, hp.nEmb, hp.nFfn, 1, stream)
        if lw.wUpQ != nil:
          gpuLinearColQuant(tmp1, xNormPtr, lw.wUpQ, hp.nEmb, hp.nFfn, lw.wUpQType, stream)
        else:
          gpuLinearCol(tmp1, xNormPtr, lw.wUp, hp.nEmb, hp.nFfn, 1, stream)
        when defined(profileHippo):
          recordStop(eventPairs, stream)
          recordStart(eventPairs, KcSiluMul, stream)
        gpuSiluMul(tmp2, tmp0, tmp1, hp.nFfn, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcLinearDown, stream)
      if lw.wDownQ != nil:
        gpuLinearColQuant(tmp0, tmp2, lw.wDownQ, hp.nFfn, hp.nEmb, lw.wDownQType, stream)
      else:
        gpuLinearCol(tmp0, tmp2, lw.wDown, hp.nFfn, hp.nEmb, 1, stream)

    of lkAttention:
      let lkvDim = lw.layerNHeadKv * headDim
      let lqDim = hp.nHead * headDim
      if lw.wqQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.wqQ, hp.nEmb, lqDim, lw.wqQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wq, hp.nEmb, lqDim, 1, stream)
      if lw.wkQ != nil:
        gpuLinearColQuant(tmp1, xNormPtr, lw.wkQ, hp.nEmb, lkvDim, lw.wkQType, stream)
      else:
        gpuLinearCol(tmp1, xNormPtr, lw.wk, hp.nEmb, lkvDim, 1, stream)
      if lw.wvQ != nil:
        gpuLinearColQuant(tmp2, xNormPtr, lw.wvQ, hp.nEmb, lkvDim, lw.wvQType, stream)
      else:
        gpuLinearCol(tmp2, xNormPtr, lw.wv, hp.nEmb, lkvDim, 1, stream)
      gpuRopeQKDecode(tmp0, tmp1, hp.nHead, lw.layerNHeadKv, headDim, ropeDim, hp.ropeFreqBase, pos, stream)
      gpuStoreKVPair(cache.gpuCache.k[layer].devicePtr, tmp1,
                      cache.gpuCache.v[layer].devicePtr, tmp2,
                      lkvDim, 1, cache.gpuCache.maxLen, pos, stream)
      gpuAttentionDecode(xNormPtr, tmp0, cache.gpuCache.k[layer].devicePtr,
                         cache.gpuCache.v[layer].devicePtr,
                         hp.nHead, lw.layerNHeadKv, headDim, pos + 1,
                         cache.gpuCache.maxLen, stream)
      if lw.woQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.woQ, lqDim, hp.nEmb, lw.woQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wo, lqDim, hp.nEmb, 1, stream)

    of lkFfnOnly:
      let lnFfn = lw.layerNFfn
      if lw.wUpQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.wUpQ, hp.nEmb, lnFfn, lw.wUpQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wUp, hp.nEmb, lnFfn, 1, stream)
      gpuReluSqr(tmp0, lnFfn, stream)
      if lw.wDownQ != nil:
        gpuLinearColQuant(tmp0, tmp0, lw.wDownQ, lnFfn, hp.nEmb, lw.wDownQType, stream)
      else:
        gpuLinearCol(tmp0, tmp0, lw.wDown, lnFfn, hp.nEmb, 1, stream)

    of lkMoeFfn:
      gpuLinearCol(tmp0, xNormPtr, lw.moeRouterW, hp.nEmb, hp.nExperts, 1, stream)
      if lw.moeExpertBias != nil:
        gpuAdd(tmp0, tmp0, lw.moeExpertBias, hp.nExperts, stream)
      block:
        let eiPtr = tmp3
        let ewPtr = cast[pointer](cast[uint](tmp3) + uint(hp.nExpertsUsed * sizeof(int32)))
        gpuMoeTopK(eiPtr, ewPtr, tmp0, hp.nExperts, hp.nExpertsUsed, stream)
        var expertIndices: array[32, int32]
        var expertWeights: array[32, float32]
        gpuDownloadFromDevice(addr expertIndices[0], eiPtr, hp.nExpertsUsed * sizeof(int32), stream)
        gpuDownloadFromDevice(addr expertWeights[0], ewPtr, hp.nExpertsUsed * sizeof(float32), stream)
        gpuStreamSync(stream)
        for e in 0 ..< hp.nExpertsUsed:
          expertWeights[e] *= hp.expertWeightsScale
        gpuZeroBuffer(tmp2, hp.nEmb, stream)
        for e in 0 ..< hp.nExpertsUsed:
          let eidx = int(expertIndices[e])
          let upOff = cast[pointer](cast[uint](lw.moeUpExpsQ) + uint(eidx * lw.moeUpExpsSliceBytes))
          gpuLinearColQuant(tmp0, xNormPtr, upOff, hp.nEmb, hp.expertFfnDim, lw.moeUpExpsQType, stream)
          gpuReluSqr(tmp0, hp.expertFfnDim, stream)
          let downOff = cast[pointer](cast[uint](lw.moeDownExpsQ) + uint(eidx * lw.moeDownExpsSliceBytes))
          gpuLinearColQuant(tmp1, tmp0, downOff, hp.expertFfnDim, hp.nEmb, lw.moeDownExpsQType, stream)
          gpuScaleAdd(tmp2, tmp1, expertWeights[e], hp.nEmb, stream)
        let shFfn = hp.sharedExpertFfnDim
        if lw.moeShUpQ != nil:
          gpuLinearColQuant(tmp0, xNormPtr, lw.moeShUpQ, hp.nEmb, shFfn, lw.moeShUpQType, stream)
        else:
          gpuLinearCol(tmp0, xNormPtr, lw.wUp, hp.nEmb, shFfn, 1, stream)
        gpuReluSqr(tmp0, shFfn, stream)
        if lw.moeShDownQ != nil:
          gpuLinearColQuant(tmp1, tmp0, lw.moeShDownQ, shFfn, hp.nEmb, lw.moeShDownQType, stream)
        else:
          gpuLinearCol(tmp1, tmp0, lw.wDown, shFfn, hp.nEmb, 1, stream)
        gpuAdd(tmp0, tmp2, tmp1, hp.nEmb, stream)

    of lkSsm:
      let ssmInner = hp.ssmInnerSize
      let nGroups = hp.ssmGroupCount
      let stateSize = hp.ssmStateSize
      let nHeads = hp.ssmDtRank
      let ssmHeadDim = ssmInner div nHeads
      let headsPerGroup = nHeads div nGroups
      let groupSize = headsPerGroup * ssmHeadDim
      let convDim = ssmInner + 2 * nGroups * stateSize
      # ssm_in projection: xNormPtr[nEmb] → tmp0[ssmProjDim]
      if lw.ssmInQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.ssmInQ, hp.nEmb, ssmProjDim, lw.ssmInQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wUp, hp.nEmb, ssmProjDim, 1, stream)
      # Split: z = tmp0[0..ssmInner-1], x_bc_dt = tmp0[ssmInner..]
      # Conv1d on x_bc portion (first convDim of x_bc_dt), dt passes through
      let xBcDtPtr = cast[pointer](cast[uint](tmp0) + uint(ssmInner * sizeof(float32)))
      let ssmLayerIdx = cache.ssmState.ssmLayerMap[layer]
      gpuConv1dDecode(cache.ssmState.convState[ssmLayerIdx].devicePtr,
                       xBcDtPtr, lw.ssmConv1dW, lw.ssmConv1dBias, tmp1,
                       convDim, hp.ssmConvKernel, stream)
      # SiLU on entire conv output (x + B + C)
      gpuSilu(tmp1, convDim, stream)
      let bPtr = cast[pointer](cast[uint](tmp1) + uint(ssmInner * sizeof(float32)))
      let cPtr = cast[pointer](cast[uint](bPtr) + uint(nGroups * stateSize * sizeof(float32)))
      let dtPtr = cast[pointer](cast[uint](xBcDtPtr) + uint(convDim * sizeof(float32)))
      gpuSsmScanDecode(cache.ssmState.recState[ssmLayerIdx].devicePtr,
                        tmp1, bPtr, cPtr, dtPtr, lw.ssmDtBias, lw.ssmA, lw.ssmD,
                        tmp2, nHeads, headsPerGroup, ssmHeadDim, stateSize, stream)
      # Gate: tmp2 = (tmp2 + D*x) * silu(z), then group RMSNorm
      gpuSilu(tmp0, ssmInner, stream)
      gpuElemMul(tmp2, tmp0, ssmInner, stream)
      gpuGroupRmsNorm(tmp2, lw.ssmNorm, nGroups, groupSize, hp.rmsEps, stream)
      # ssm_out projection: tmp2[ssmInner] → tmp0[nEmb]
      if lw.ssmOutQ != nil:
        gpuLinearColQuant(tmp0, tmp2, lw.ssmOutQ, ssmInner, hp.nEmb, lw.ssmOutQType, stream)
      else:
        gpuLinearCol(tmp0, tmp2, lw.wDown, ssmInner, hp.nEmb, 1, stream)

    of lkAttnMoe:
      let qFullDim = 2 * qDim
      if lw.wqQ != nil:
        gpuLinearColQuant(tmp2, xNormPtr, lw.wqQ, hp.nEmb, qFullDim, lw.wqQType, stream)
      else:
        gpuLinearCol(tmp2, xNormPtr, lw.wq, hp.nEmb, qFullDim, 1, stream)
      gpuDeinterleaveQGate(tmp0, tmp3, tmp2, hp.nHead, headDim, stream)
      if lw.wkQ != nil:
        gpuLinearColQuant(tmp1, xNormPtr, lw.wkQ, hp.nEmb, kvDim, lw.wkQType, stream)
      else:
        gpuLinearCol(tmp1, xNormPtr, lw.wk, hp.nEmb, kvDim, 1, stream)
      if lw.wvQ != nil:
        gpuLinearColQuant(tmp2, xNormPtr, lw.wvQ, hp.nEmb, kvDim, lw.wvQType, stream)
      else:
        gpuLinearCol(tmp2, xNormPtr, lw.wv, hp.nEmb, kvDim, 1, stream)
      if lw.attnQNorm != nil:
        gpuQkNorm(tmp0, lw.attnQNorm, hp.nHead, headDim, hp.rmsEps, stream)
      if lw.attnKNorm != nil:
        gpuQkNorm(tmp1, lw.attnKNorm, hp.nHeadKv, headDim, hp.rmsEps, stream)
      gpuFusedRopeStoreKV(tmp0, tmp1, tmp2,
                          cache.gpuCache.k[layer].devicePtr,
                          cache.gpuCache.v[layer].devicePtr,
                          hp.nHead, hp.nHeadKv, headDim, ropeDim,
                          kvDim, cache.gpuCache.maxLen, pos, stream)
      gpuAttentionDecode(xNormPtr, tmp0, cache.gpuCache.k[layer].devicePtr,
                         cache.gpuCache.v[layer].devicePtr,
                         hp.nHead, hp.nHeadKv, headDim, pos + 1,
                         cache.gpuCache.maxLen, stream)
      gpuAttnGate(xNormPtr, tmp3, qDim, stream)
      if lw.woQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.woQ, qDim, hp.nEmb, lw.woQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wo, qDim, hp.nEmb, 1, stream)
      gpuResidualRmsnorm(xNormPtr, xPtr, tmp0, lw.postAttnNorm, hp.nEmb, hp.rmsEps, stream)
      gpuLinearCol(tmp0, xNormPtr, lw.moeRouterW, hp.nEmb, hp.nExperts, 1, stream)
      let eiPtr = tmp3
      let ewPtr = cast[pointer](cast[uint](tmp3) + uint(hp.nExpertsUsed * sizeof(int32)))
      gpuMoeTopK(eiPtr, ewPtr, tmp0, hp.nExperts, hp.nExpertsUsed, stream)
      var expertIndices: array[32, int32]
      var expertWeights: array[32, float32]
      gpuDownloadFromDevice(addr expertIndices[0], eiPtr, hp.nExpertsUsed * sizeof(int32), stream)
      gpuDownloadFromDevice(addr expertWeights[0], ewPtr, hp.nExpertsUsed * sizeof(float32), stream)
      gpuStreamSync(stream)
      gpuZeroBuffer(tmp2, hp.nEmb, stream)
      for e in 0 ..< hp.nExpertsUsed:
        let eidx = int(expertIndices[e])
        let gateOff = cast[pointer](cast[uint](lw.moeGateExpsQ) + uint(eidx * lw.moeGateExpsSliceBytes))
        gpuLinearColQuant(tmp0, xNormPtr, gateOff, hp.nEmb, hp.expertFfnDim, lw.moeGateExpsQType, stream)
        let upOff = cast[pointer](cast[uint](lw.moeUpExpsQ) + uint(eidx * lw.moeUpExpsSliceBytes))
        gpuLinearColQuant(tmp1, xNormPtr, upOff, hp.nEmb, hp.expertFfnDim, lw.moeUpExpsQType, stream)
        gpuSiluMul(tmp0, tmp0, tmp1, hp.expertFfnDim, stream)
        let downOff = cast[pointer](cast[uint](lw.moeDownExpsQ) + uint(eidx * lw.moeDownExpsSliceBytes))
        gpuLinearColQuant(tmp1, tmp0, downOff, hp.expertFfnDim, hp.nEmb, lw.moeDownExpsQType, stream)
        gpuScaleAdd(tmp2, tmp1, expertWeights[e], hp.nEmb, stream)
      if lw.moeShGateQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.moeShGateQ, hp.nEmb, hp.expertFfnDim, lw.moeShGateQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wGate, hp.nEmb, hp.expertFfnDim, 1, stream)
      if lw.moeShUpQ != nil:
        gpuLinearColQuant(tmp1, xNormPtr, lw.moeShUpQ, hp.nEmb, hp.expertFfnDim, lw.moeShUpQType, stream)
      else:
        gpuLinearCol(tmp1, xNormPtr, lw.wUp, hp.nEmb, hp.expertFfnDim, 1, stream)
      gpuSiluMul(tmp0, tmp0, tmp1, hp.expertFfnDim, stream)
      if lw.moeShDownQ != nil:
        gpuLinearColQuant(tmp1, tmp0, lw.moeShDownQ, hp.expertFfnDim, hp.nEmb, lw.moeShDownQType, stream)
      else:
        gpuLinearCol(tmp1, tmp0, lw.wDown, hp.expertFfnDim, hp.nEmb, 1, stream)
      gpuSharedExpertGate(tmp1, xNormPtr, lw.moeShGateScalar, hp.nEmb, stream)
      gpuAdd(tmp0, tmp2, tmp1, hp.nEmb, stream)

    of lkSsmAttnMoe:
      let nVHeads = hp.ssmDtRank
      let nKHeads = hp.ssmGroupCount
      let headKDim = hp.ssmStateSize
      let headVDim = hp.ssmInnerSize div nVHeads
      let convDim = 2 * nKHeads * headKDim + hp.ssmInnerSize
      let qkDim = nKHeads * headKDim
      if lw.wqkvQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.wqkvQ, hp.nEmb, convDim, lw.wqkvQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wq, hp.nEmb, convDim, 1, stream)
      if lw.ssmGateQ != nil:
        gpuLinearColQuant(tmp2, xNormPtr, lw.ssmGateQ, hp.nEmb, hp.ssmInnerSize, lw.ssmGateQType, stream)
      else:
        gpuLinearCol(tmp2, xNormPtr, lw.wo, hp.nEmb, hp.ssmInnerSize, 1, stream)
      let alphaPtr = tmp3
      let betaRawPtr = cast[pointer](cast[uint](tmp3) + uint(nVHeads * sizeof(float32)))
      if lw.ssmAlphaQ != nil:
        gpuLinearColQuant(alphaPtr, xNormPtr, lw.ssmAlphaQ, hp.nEmb, nVHeads, lw.ssmAlphaQType, stream)
      else:
        gpuLinearCol(alphaPtr, xNormPtr, lw.wk, hp.nEmb, nVHeads, 1, stream)
      if lw.ssmBetaQ != nil:
        gpuLinearColQuant(betaRawPtr, xNormPtr, lw.ssmBetaQ, hp.nEmb, nVHeads, lw.ssmBetaQType, stream)
      else:
        gpuLinearCol(betaRawPtr, xNormPtr, lw.wv, hp.nEmb, nVHeads, 1, stream)
      let gateDecayPtr = cast[pointer](cast[uint](tmp3) + uint(2 * nVHeads * sizeof(float32)))
      let betaSigPtr = cast[pointer](cast[uint](tmp3) + uint(3 * nVHeads * sizeof(float32)))
      gpuDeltaNetGate(gateDecayPtr, betaSigPtr, alphaPtr, lw.ssmDtBias, lw.ssmA, betaRawPtr,
                       nVHeads, stream)
      let ssmLayerIdx = cache.ssmState.ssmLayerMap[layer]
      gpuConv1dDecode(cache.ssmState.convState[ssmLayerIdx].devicePtr,
                       tmp0, lw.ssmConv1dW, lw.ssmConv1dBias, tmp1,
                       convDim, hp.ssmConvKernel, stream)
      gpuSilu(tmp1, convDim, stream)
      let qConvPtr = tmp1
      let kConvPtr = cast[pointer](cast[uint](tmp1) + uint(qkDim * sizeof(float32)))
      let vConvPtr = cast[pointer](cast[uint](kConvPtr) + uint(qkDim * sizeof(float32)))
      gpuL2NormPerHead(qConvPtr, nKHeads, headKDim, stream)
      gpuL2NormPerHead(kConvPtr, nKHeads, headKDim, stream)
      let qExpPtr = tmp0
      let kExpPtr = cast[pointer](cast[uint](tmp0) + uint(nVHeads * headKDim * sizeof(float32)))
      gpuExpandHeads(qExpPtr, qConvPtr, nKHeads, nVHeads, headKDim, stream)
      gpuExpandHeads(kExpPtr, kConvPtr, nKHeads, nVHeads, headKDim, stream)
      let dnOutPtr = tmp1
      gpuDeltaNetDecode(cache.ssmState.recState[ssmLayerIdx].devicePtr,
                         qExpPtr, kExpPtr, vConvPtr, gateDecayPtr, betaSigPtr, dnOutPtr,
                         nVHeads, headKDim, headVDim, stream)
      gpuGatedRmsNorm(tmp0, dnOutPtr, tmp2, lw.ssmNorm, nVHeads, headVDim, hp.rmsEps, stream)
      if lw.ssmOutQ != nil:
        gpuLinearColQuant(tmp1, tmp0, lw.ssmOutQ, hp.ssmInnerSize, hp.nEmb, lw.ssmOutQType, stream)
      else:
        gpuLinearCol(tmp1, tmp0, lw.wGate, hp.ssmInnerSize, hp.nEmb, 1, stream)
      gpuResidualRmsnorm(xNormPtr, xPtr, tmp1, lw.postAttnNorm, hp.nEmb, hp.rmsEps, stream)
      gpuLinearCol(tmp0, xNormPtr, lw.moeRouterW, hp.nEmb, hp.nExperts, 1, stream)
      let eiPtr = tmp3
      let ewPtr = cast[pointer](cast[uint](tmp3) + uint(hp.nExpertsUsed * sizeof(int32)))
      gpuMoeTopK(eiPtr, ewPtr, tmp0, hp.nExperts, hp.nExpertsUsed, stream)
      var expertIndices: array[32, int32]
      var expertWeights: array[32, float32]
      gpuDownloadFromDevice(addr expertIndices[0], eiPtr, hp.nExpertsUsed * sizeof(int32), stream)
      gpuDownloadFromDevice(addr expertWeights[0], ewPtr, hp.nExpertsUsed * sizeof(float32), stream)
      gpuStreamSync(stream)
      gpuZeroBuffer(tmp2, hp.nEmb, stream)
      for e in 0 ..< hp.nExpertsUsed:
        let eidx = int(expertIndices[e])
        let gateOff = cast[pointer](cast[uint](lw.moeGateExpsQ) + uint(eidx * lw.moeGateExpsSliceBytes))
        gpuLinearColQuant(tmp0, xNormPtr, gateOff, hp.nEmb, hp.expertFfnDim, lw.moeGateExpsQType, stream)
        let upOff = cast[pointer](cast[uint](lw.moeUpExpsQ) + uint(eidx * lw.moeUpExpsSliceBytes))
        gpuLinearColQuant(tmp1, xNormPtr, upOff, hp.nEmb, hp.expertFfnDim, lw.moeUpExpsQType, stream)
        gpuSiluMul(tmp0, tmp0, tmp1, hp.expertFfnDim, stream)
        let downOff = cast[pointer](cast[uint](lw.moeDownExpsQ) + uint(eidx * lw.moeDownExpsSliceBytes))
        gpuLinearColQuant(tmp1, tmp0, downOff, hp.expertFfnDim, hp.nEmb, lw.moeDownExpsQType, stream)
        gpuScaleAdd(tmp2, tmp1, expertWeights[e], hp.nEmb, stream)
      if lw.moeShGateQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.moeShGateQ, hp.nEmb, hp.expertFfnDim, lw.moeShGateQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wGate, hp.nEmb, hp.expertFfnDim, 1, stream)
      if lw.moeShUpQ != nil:
        gpuLinearColQuant(tmp1, xNormPtr, lw.moeShUpQ, hp.nEmb, hp.expertFfnDim, lw.moeShUpQType, stream)
      else:
        gpuLinearCol(tmp1, xNormPtr, lw.wUp, hp.nEmb, hp.expertFfnDim, 1, stream)
      gpuSiluMul(tmp0, tmp0, tmp1, hp.expertFfnDim, stream)
      if lw.moeShDownQ != nil:
        gpuLinearColQuant(tmp1, tmp0, lw.moeShDownQ, hp.expertFfnDim, hp.nEmb, lw.moeShDownQType, stream)
      else:
        gpuLinearCol(tmp1, tmp0, lw.wDown, hp.expertFfnDim, hp.nEmb, 1, stream)
      gpuSharedExpertGate(tmp1, xNormPtr, lw.moeShGateScalar, hp.nEmb, stream)
      gpuAdd(tmp0, tmp2, tmp1, hp.nEmb, stream)

    when defined(profileHippo):
      recordStop(eventPairs, stream)
      recordStart(eventPairs, KcResidualFfn, stream)

    if layer < hp.nLayer - 1:
      gpuResidualRmsnorm(xNormPtr, xPtr, tmp0,
                          modelPtrs.layers[layer + 1].attnNorm,
                          hp.nEmb, hp.rmsEps, stream)
    else:
      gpuAdd(xPtr, xPtr, tmp0, hp.nEmb, stream)

    when defined(profileHippo):
      recordStop(eventPairs, stream)
      kernelLaunchMs += (epochTime() - klStart) * 1000

  when defined(profileHippo):
    recordStart(eventPairs, KcFinalNormOutput, stream)
  gpuRmsnormCols(xNormPtr, xPtr, modelPtrs.normWeight, hp.nEmb, 1, hp.rmsEps, stream)
  if modelPtrs.outputWeightQ != nil:
    gpuLinearColQuant(xPtr, xNormPtr, modelPtrs.outputWeightQ,
                      modelPtrs.outputShape0, modelPtrs.outputShape1,
                      modelPtrs.outputQType, stream)
  else:
    gpuLinearCol(xPtr, xNormPtr, modelPtrs.outputWeight, modelPtrs.outputShape0, modelPtrs.outputShape1, 1, stream)
  when defined(profileHippo):
    recordStop(eventPairs, stream)

  when defined(profileHippo):
    hippoEventRecord(gpuEndEvt, stream)
    let preSyncWall = epochTime()

  result = newTensor(@[modelPtrs.outputShape1, 1])
  let bytes = result.data.len * sizeof(float32)
  gpuDownloadFromDevice(addr result.data[0], xPtr, bytes, stream)
  gpuStreamSync(stream)

  when defined(profileHippo):
    let wallEnd = epochTime()
    hippoEventSynchronize(gpuEndEvt)
    printBreakdown(eventPairs)
    let gpuMs = hippoEventElapsedTime(gpuStartEvt, gpuEndEvt)
    let totalMs = (wallEnd - wallStart) * 1000
    let cpuMs = (preSyncWall - wallStart) * 1000
    let syncMs = (wallEnd - preSyncWall) * 1000
    echo "decode[pos=", pos, "]: total=", totalMs.formatFloat(ffDecimal, 1), "ms",
         " cpu=", cpuMs.formatFloat(ffDecimal, 1), "ms",
         " sync=", syncMs.formatFloat(ffDecimal, 1), "ms",
         " gpu=", gpuMs.formatFloat(ffDecimal, 1), "ms",
         " kernelLaunch=", kernelLaunchMs.formatFloat(ffDecimal, 1), "ms"
    hippoEventDestroy(gpuStartEvt)
    hippoEventDestroy(gpuEndEvt)

  cache.curLen = pos + 1
  cache.gpuCache.curLen = cache.curLen

proc forwardDecodeToken*(m: var Model, token: int32, cache: var KvCache): int32 =
  let hp = m.hparams
  if hp.arch != "" and hp.arch notin ["llama", "qwen3", "nemotron_h", "nemotron_h_moe", "qwen35moe"]:
    raise newException(ValueError, "unsupported architecture: " & hp.arch)
  if hp.nHeadKv != 0 and hp.nHead > 0 and (hp.nHead mod hp.nHeadKv) != 0:
    raise newException(ValueError, "GQA requires head_count divisible by head_count_kv")
  if cache.curLen >= cache.maxLen:
    raise newException(ValueError, "KV cache full")

  let headDim = hp.headDim
  let ropeDim = if hp.ropeDim > 0: hp.ropeDim else: headDim
  let qDim = hp.nHead * headDim
  let kvDim = hp.nHeadKv * headDim
  let pos = cache.curLen
  let ssmProjDim = if hp.ssmInnerSize > 0:
    hp.ssmInnerSize * 2 + 2 * hp.ssmGroupCount * hp.ssmStateSize + hp.ssmDtRank
  else: 0

  ensureGpuContext()
  var maxRows = max(max(max(hp.nEmb, hp.nFfn), hp.nVocab), qDim)
  maxRows = max(maxRows, ssmProjDim)
  for i in 0 ..< hp.layerNFfn.len:
    maxRows = max(maxRows, hp.layerNFfn[i])
  if hp.nExperts > 0:
    maxRows = max(maxRows, hp.nExperts)
    maxRows = max(maxRows, qDim + 2 * kvDim)
    maxRows = max(maxRows, hp.sharedExpertFfnDim)
  ensureActivationBuffers(maxRows)
  ensureScratchBuffers(maxRows)
  let stream = gpuCtx.stream

  ensureModelGpuPtrs(m, hp)

  var tok = token
  let tokenPtr = gpuUploadInt32Pooled(unsafeAddr tok, 1, stream)

  var xPtr = gpuCtx.act0.devicePtr
  let xNormPtr = gpuCtx.act1.devicePtr
  let tmp0 = gpuCtx.scratch0.devicePtr
  let tmp1 = gpuCtx.scratch1.devicePtr
  let tmp2 = gpuCtx.scratch2.devicePtr
  let tmp3 = gpuCtx.scratch3.devicePtr

  gpuEmbedding(xPtr, modelPtrs.tokEmb, cast[ptr int32](tokenPtr),
               hp.nEmb, 1, hp.nVocab, stream)

  gpuRmsnormCols(xNormPtr, xPtr, modelPtrs.layers[0].attnNorm, hp.nEmb, 1, hp.rmsEps, stream)

  for layer in 0 ..< hp.nLayer:
    let lw = modelPtrs.layers[layer]

    case lw.kind
    of lkAttentionFfn:
      if lw.wqQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.wqQ, hp.nEmb, qDim, lw.wqQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wq, hp.nEmb, qDim, 1, stream)
      when HippoWarpSize == 32:
        if lw.wkQ != nil and lw.wvQ != nil and lw.wkQType == GgmlTypeQ4K and lw.wvQType == GgmlTypeQ4K:
          gpuFusedKVLinearQ4K(tmp1, tmp2, xNormPtr, lw.wkQ, lw.wvQ, hp.nEmb, kvDim, stream)
        elif lw.wkQ != nil and lw.wvQ != nil and lw.wkQType == GgmlTypeQ2K and lw.wvQType == GgmlTypeQ3K:
          gpuFusedKVLinearQ2KQ3K(tmp1, tmp2, xNormPtr, lw.wkQ, lw.wvQ, hp.nEmb, kvDim, stream)
        else:
          if lw.wkQ != nil:
            gpuLinearColQuant(tmp1, xNormPtr, lw.wkQ, hp.nEmb, kvDim, lw.wkQType, stream)
          else:
            gpuLinearCol(tmp1, xNormPtr, lw.wk, hp.nEmb, kvDim, 1, stream)
          if lw.wvQ != nil:
            gpuLinearColQuant(tmp2, xNormPtr, lw.wvQ, hp.nEmb, kvDim, lw.wvQType, stream)
          else:
            gpuLinearCol(tmp2, xNormPtr, lw.wv, hp.nEmb, kvDim, 1, stream)
      else:
        if lw.wkQ != nil:
          gpuLinearColQuant(tmp1, xNormPtr, lw.wkQ, hp.nEmb, kvDim, lw.wkQType, stream)
        else:
          gpuLinearCol(tmp1, xNormPtr, lw.wk, hp.nEmb, kvDim, 1, stream)
        if lw.wvQ != nil:
          gpuLinearColQuant(tmp2, xNormPtr, lw.wvQ, hp.nEmb, kvDim, lw.wvQType, stream)
        else:
          gpuLinearCol(tmp2, xNormPtr, lw.wv, hp.nEmb, kvDim, 1, stream)
      if lw.attnQNorm != nil:
        gpuQkNorm(tmp0, lw.attnQNorm, hp.nHead, headDim, hp.rmsEps, stream)
      if lw.attnKNorm != nil:
        gpuQkNorm(tmp1, lw.attnKNorm, hp.nHeadKv, headDim, hp.rmsEps, stream)
      gpuFusedRopeStoreKV(tmp0, tmp1, tmp2,
                          cache.gpuCache.k[layer].devicePtr,
                          cache.gpuCache.v[layer].devicePtr,
                          hp.nHead, hp.nHeadKv, headDim, ropeDim,
                          kvDim, cache.gpuCache.maxLen, pos, stream)
      gpuAttentionDecode(xNormPtr, tmp0, cache.gpuCache.k[layer].devicePtr,
                         cache.gpuCache.v[layer].devicePtr,
                         hp.nHead, hp.nHeadKv, headDim, pos + 1,
                         cache.gpuCache.maxLen, stream)
      if lw.woQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.woQ, qDim, hp.nEmb, lw.woQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wo, qDim, hp.nEmb, 1, stream)
      gpuResidualRmsnorm(xNormPtr, xPtr, tmp0, lw.ffnNorm, hp.nEmb, hp.rmsEps, stream)
      when HippoWarpSize == 32:
        if lw.wGateQType == GgmlTypeQ4K and lw.wUpQType == GgmlTypeQ4K:
          gpuFusedGateUpSiluQ4K(tmp2, xNormPtr, lw.wGateQ, lw.wUpQ,
                                hp.nEmb, hp.nFfn, stream)
        elif lw.wGateQType == GgmlTypeQ3K and lw.wUpQType == GgmlTypeQ3K:
          gpuFusedGateUpSiluQ3K(tmp2, xNormPtr, lw.wGateQ, lw.wUpQ,
                                hp.nEmb, hp.nFfn, stream)
        else:
          if lw.wGateQ != nil:
            gpuLinearColQuant(tmp0, xNormPtr, lw.wGateQ, hp.nEmb, hp.nFfn, lw.wGateQType, stream)
          else:
            gpuLinearCol(tmp0, xNormPtr, lw.wGate, hp.nEmb, hp.nFfn, 1, stream)
          if lw.wUpQ != nil:
            gpuLinearColQuant(tmp1, xNormPtr, lw.wUpQ, hp.nEmb, hp.nFfn, lw.wUpQType, stream)
          else:
            gpuLinearCol(tmp1, xNormPtr, lw.wUp, hp.nEmb, hp.nFfn, 1, stream)
          gpuSiluMul(tmp2, tmp0, tmp1, hp.nFfn, stream)
      else:
        if lw.wGateQ != nil:
          gpuLinearColQuant(tmp0, xNormPtr, lw.wGateQ, hp.nEmb, hp.nFfn, lw.wGateQType, stream)
        else:
          gpuLinearCol(tmp0, xNormPtr, lw.wGate, hp.nEmb, hp.nFfn, 1, stream)
        if lw.wUpQ != nil:
          gpuLinearColQuant(tmp1, xNormPtr, lw.wUpQ, hp.nEmb, hp.nFfn, lw.wUpQType, stream)
        else:
          gpuLinearCol(tmp1, xNormPtr, lw.wUp, hp.nEmb, hp.nFfn, 1, stream)
        gpuSiluMul(tmp2, tmp0, tmp1, hp.nFfn, stream)
      if lw.wDownQ != nil:
        gpuLinearColQuant(tmp0, tmp2, lw.wDownQ, hp.nFfn, hp.nEmb, lw.wDownQType, stream)
      else:
        gpuLinearCol(tmp0, tmp2, lw.wDown, hp.nFfn, hp.nEmb, 1, stream)

    of lkAttention:
      let lkvDim = lw.layerNHeadKv * headDim
      let lqDim = hp.nHead * headDim
      if lw.wqQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.wqQ, hp.nEmb, lqDim, lw.wqQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wq, hp.nEmb, lqDim, 1, stream)
      if lw.wkQ != nil:
        gpuLinearColQuant(tmp1, xNormPtr, lw.wkQ, hp.nEmb, lkvDim, lw.wkQType, stream)
      else:
        gpuLinearCol(tmp1, xNormPtr, lw.wk, hp.nEmb, lkvDim, 1, stream)
      if lw.wvQ != nil:
        gpuLinearColQuant(tmp2, xNormPtr, lw.wvQ, hp.nEmb, lkvDim, lw.wvQType, stream)
      else:
        gpuLinearCol(tmp2, xNormPtr, lw.wv, hp.nEmb, lkvDim, 1, stream)
      gpuFusedRopeStoreKV(tmp0, tmp1, tmp2,
                          cache.gpuCache.k[layer].devicePtr,
                          cache.gpuCache.v[layer].devicePtr,
                          hp.nHead, lw.layerNHeadKv, headDim, ropeDim,
                          lkvDim, cache.gpuCache.maxLen, pos, stream)
      gpuAttentionDecode(xNormPtr, tmp0, cache.gpuCache.k[layer].devicePtr,
                         cache.gpuCache.v[layer].devicePtr,
                         hp.nHead, lw.layerNHeadKv, headDim, pos + 1,
                         cache.gpuCache.maxLen, stream)
      if lw.woQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.woQ, lqDim, hp.nEmb, lw.woQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wo, lqDim, hp.nEmb, 1, stream)

    of lkFfnOnly:
      let lnFfn = lw.layerNFfn
      if lw.wUpQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.wUpQ, hp.nEmb, lnFfn, lw.wUpQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wUp, hp.nEmb, lnFfn, 1, stream)
      gpuReluSqr(tmp0, lnFfn, stream)
      if lw.wDownQ != nil:
        gpuLinearColQuant(tmp0, tmp0, lw.wDownQ, lnFfn, hp.nEmb, lw.wDownQType, stream)
      else:
        gpuLinearCol(tmp0, tmp0, lw.wDown, lnFfn, hp.nEmb, 1, stream)

    of lkMoeFfn:
      gpuLinearCol(tmp0, xNormPtr, lw.moeRouterW, hp.nEmb, hp.nExperts, 1, stream)
      if lw.moeExpertBias != nil:
        gpuAdd(tmp0, tmp0, lw.moeExpertBias, hp.nExperts, stream)
      block:
        let eiPtr = tmp3
        let ewPtr = cast[pointer](cast[uint](tmp3) + uint(hp.nExpertsUsed * sizeof(int32)))
        gpuMoeTopK(eiPtr, ewPtr, tmp0, hp.nExperts, hp.nExpertsUsed, stream)
        var expertIndices: array[32, int32]
        var expertWeights: array[32, float32]
        gpuDownloadFromDevice(addr expertIndices[0], eiPtr, hp.nExpertsUsed * sizeof(int32), stream)
        gpuDownloadFromDevice(addr expertWeights[0], ewPtr, hp.nExpertsUsed * sizeof(float32), stream)
        gpuStreamSync(stream)
        for e in 0 ..< hp.nExpertsUsed:
          expertWeights[e] *= hp.expertWeightsScale
        gpuZeroBuffer(tmp2, hp.nEmb, stream)
        for e in 0 ..< hp.nExpertsUsed:
          let eidx = int(expertIndices[e])
          let upOff = cast[pointer](cast[uint](lw.moeUpExpsQ) + uint(eidx * lw.moeUpExpsSliceBytes))
          gpuLinearColQuant(tmp0, xNormPtr, upOff, hp.nEmb, hp.expertFfnDim, lw.moeUpExpsQType, stream)
          gpuReluSqr(tmp0, hp.expertFfnDim, stream)
          let downOff = cast[pointer](cast[uint](lw.moeDownExpsQ) + uint(eidx * lw.moeDownExpsSliceBytes))
          gpuLinearColQuant(tmp1, tmp0, downOff, hp.expertFfnDim, hp.nEmb, lw.moeDownExpsQType, stream)
          gpuScaleAdd(tmp2, tmp1, expertWeights[e], hp.nEmb, stream)
        let shFfn = hp.sharedExpertFfnDim
        if lw.moeShUpQ != nil:
          gpuLinearColQuant(tmp0, xNormPtr, lw.moeShUpQ, hp.nEmb, shFfn, lw.moeShUpQType, stream)
        else:
          gpuLinearCol(tmp0, xNormPtr, lw.wUp, hp.nEmb, shFfn, 1, stream)
        gpuReluSqr(tmp0, shFfn, stream)
        if lw.moeShDownQ != nil:
          gpuLinearColQuant(tmp1, tmp0, lw.moeShDownQ, shFfn, hp.nEmb, lw.moeShDownQType, stream)
        else:
          gpuLinearCol(tmp1, tmp0, lw.wDown, shFfn, hp.nEmb, 1, stream)
        gpuAdd(tmp0, tmp2, tmp1, hp.nEmb, stream)

    of lkSsm:
      let ssmInner = hp.ssmInnerSize
      let nGroups = hp.ssmGroupCount
      let stateSize = hp.ssmStateSize
      let nHeads = hp.ssmDtRank
      let ssmHeadDim = ssmInner div nHeads
      let headsPerGroup = nHeads div nGroups
      let groupSize = headsPerGroup * ssmHeadDim
      let convDim = ssmInner + 2 * nGroups * stateSize
      if lw.ssmInQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.ssmInQ, hp.nEmb, ssmProjDim, lw.ssmInQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wUp, hp.nEmb, ssmProjDim, 1, stream)
      let xBcDtPtr = cast[pointer](cast[uint](tmp0) + uint(ssmInner * sizeof(float32)))
      let ssmLayerIdx = cache.ssmState.ssmLayerMap[layer]
      gpuConv1dDecode(cache.ssmState.convState[ssmLayerIdx].devicePtr,
                       xBcDtPtr, lw.ssmConv1dW, lw.ssmConv1dBias, tmp1,
                       convDim, hp.ssmConvKernel, stream)
      gpuSilu(tmp1, convDim, stream)
      let bPtr = cast[pointer](cast[uint](tmp1) + uint(ssmInner * sizeof(float32)))
      let cPtr = cast[pointer](cast[uint](bPtr) + uint(nGroups * stateSize * sizeof(float32)))
      let dtPtr = cast[pointer](cast[uint](xBcDtPtr) + uint(convDim * sizeof(float32)))
      gpuSsmScanDecode(cache.ssmState.recState[ssmLayerIdx].devicePtr,
                        tmp1, bPtr, cPtr, dtPtr, lw.ssmDtBias, lw.ssmA, lw.ssmD,
                        tmp2, nHeads, headsPerGroup, ssmHeadDim, stateSize, stream)
      gpuSilu(tmp0, ssmInner, stream)
      gpuElemMul(tmp2, tmp0, ssmInner, stream)
      gpuGroupRmsNorm(tmp2, lw.ssmNorm, nGroups, groupSize, hp.rmsEps, stream)
      if lw.ssmOutQ != nil:
        gpuLinearColQuant(tmp0, tmp2, lw.ssmOutQ, ssmInner, hp.nEmb, lw.ssmOutQType, stream)
      else:
        gpuLinearCol(tmp0, tmp2, lw.wDown, ssmInner, hp.nEmb, 1, stream)

    of lkAttnMoe:
      # Full attention + gated Q + MoE FFN (every 4th layer)
      # Q projection outputs Q+gate interleaved: [Q0(hd), G0(hd), Q1(hd), G1(hd), ...]
      let qFullDim = 2 * qDim
      if lw.wqQ != nil:
        gpuLinearColQuant(tmp2, xNormPtr, lw.wqQ, hp.nEmb, qFullDim, lw.wqQType, stream)
      else:
        gpuLinearCol(tmp2, xNormPtr, lw.wq, hp.nEmb, qFullDim, 1, stream)
      # De-interleave: tmp0=Q[qDim], tmp3=gate[qDim]
      gpuDeinterleaveQGate(tmp0, tmp3, tmp2, hp.nHead, headDim, stream)
      # K, V projections
      if lw.wkQ != nil:
        gpuLinearColQuant(tmp1, xNormPtr, lw.wkQ, hp.nEmb, kvDim, lw.wkQType, stream)
      else:
        gpuLinearCol(tmp1, xNormPtr, lw.wk, hp.nEmb, kvDim, 1, stream)
      if lw.wvQ != nil:
        gpuLinearColQuant(tmp2, xNormPtr, lw.wvQ, hp.nEmb, kvDim, lw.wvQType, stream)
      else:
        gpuLinearCol(tmp2, xNormPtr, lw.wv, hp.nEmb, kvDim, 1, stream)
      # tmp0=Q, tmp1=K, tmp2=V, tmp3=gate
      if lw.attnQNorm != nil:
        gpuQkNorm(tmp0, lw.attnQNorm, hp.nHead, headDim, hp.rmsEps, stream)
      if lw.attnKNorm != nil:
        gpuQkNorm(tmp1, lw.attnKNorm, hp.nHeadKv, headDim, hp.rmsEps, stream)
      gpuFusedRopeStoreKV(tmp0, tmp1, tmp2,
                          cache.gpuCache.k[layer].devicePtr,
                          cache.gpuCache.v[layer].devicePtr,
                          hp.nHead, hp.nHeadKv, headDim, ropeDim,
                          kvDim, cache.gpuCache.maxLen, pos, stream)
      gpuAttentionDecode(xNormPtr, tmp0, cache.gpuCache.k[layer].devicePtr,
                         cache.gpuCache.v[layer].devicePtr,
                         hp.nHead, hp.nHeadKv, headDim, pos + 1,
                         cache.gpuCache.maxLen, stream)
      # Apply attention gate: xNormPtr = sigmoid(gate) * xNormPtr
      gpuAttnGate(xNormPtr, tmp3, qDim, stream)
      if lw.woQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.woQ, qDim, hp.nEmb, lw.woQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wo, qDim, hp.nEmb, 1, stream)
      # First residual + post_attention_norm
      gpuResidualRmsnorm(xNormPtr, xPtr, tmp0, lw.postAttnNorm, hp.nEmb, hp.rmsEps, stream)
      # MoE FFN: router → top-K → sequential expert eval → shared expert → accumulate
      gpuLinearCol(tmp0, xNormPtr, lw.moeRouterW, hp.nEmb, hp.nExperts, 1, stream)
      let eiPtr = tmp3
      let ewPtr = cast[pointer](cast[uint](tmp3) + uint(hp.nExpertsUsed * sizeof(int32)))
      gpuMoeTopK(eiPtr, ewPtr, tmp0, hp.nExperts, hp.nExpertsUsed, stream)
      var expertIndices: array[32, int32]
      var expertWeights: array[32, float32]
      gpuDownloadFromDevice(addr expertIndices[0], eiPtr, hp.nExpertsUsed * sizeof(int32), stream)
      gpuDownloadFromDevice(addr expertWeights[0], ewPtr, hp.nExpertsUsed * sizeof(float32), stream)
      gpuStreamSync(stream)
      gpuZeroBuffer(tmp2, hp.nEmb, stream)
      for e in 0 ..< hp.nExpertsUsed:
        let eidx = int(expertIndices[e])
        let gateOff = cast[pointer](cast[uint](lw.moeGateExpsQ) + uint(eidx * lw.moeGateExpsSliceBytes))
        gpuLinearColQuant(tmp0, xNormPtr, gateOff, hp.nEmb, hp.expertFfnDim, lw.moeGateExpsQType, stream)
        let upOff = cast[pointer](cast[uint](lw.moeUpExpsQ) + uint(eidx * lw.moeUpExpsSliceBytes))
        gpuLinearColQuant(tmp1, xNormPtr, upOff, hp.nEmb, hp.expertFfnDim, lw.moeUpExpsQType, stream)
        gpuSiluMul(tmp0, tmp0, tmp1, hp.expertFfnDim, stream)
        let downOff = cast[pointer](cast[uint](lw.moeDownExpsQ) + uint(eidx * lw.moeDownExpsSliceBytes))
        gpuLinearColQuant(tmp1, tmp0, downOff, hp.expertFfnDim, hp.nEmb, lw.moeDownExpsQType, stream)
        gpuScaleAdd(tmp2, tmp1, expertWeights[e], hp.nEmb, stream)
      # Shared expert
      if lw.moeShGateQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.moeShGateQ, hp.nEmb, hp.expertFfnDim, lw.moeShGateQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wGate, hp.nEmb, hp.expertFfnDim, 1, stream)
      if lw.moeShUpQ != nil:
        gpuLinearColQuant(tmp1, xNormPtr, lw.moeShUpQ, hp.nEmb, hp.expertFfnDim, lw.moeShUpQType, stream)
      else:
        gpuLinearCol(tmp1, xNormPtr, lw.wUp, hp.nEmb, hp.expertFfnDim, 1, stream)
      gpuSiluMul(tmp0, tmp0, tmp1, hp.expertFfnDim, stream)
      if lw.moeShDownQ != nil:
        gpuLinearColQuant(tmp1, tmp0, lw.moeShDownQ, hp.expertFfnDim, hp.nEmb, lw.moeShDownQType, stream)
      else:
        gpuLinearCol(tmp1, tmp0, lw.wDown, hp.expertFfnDim, hp.nEmb, 1, stream)
      gpuSharedExpertGate(tmp1, xNormPtr, lw.moeShGateScalar, hp.nEmb, stream)
      gpuAdd(tmp0, tmp2, tmp1, hp.nEmb, stream)

    of lkSsmAttnMoe:
      # Gated Delta Net + MoE FFN
      let nVHeads = hp.ssmDtRank         # 32
      let nKHeads = hp.ssmGroupCount     # 16
      let headKDim = hp.ssmStateSize     # 128
      let headVDim = hp.ssmInnerSize div nVHeads  # 128
      let convDim = 2 * nKHeads * headKDim + hp.ssmInnerSize  # 8192
      let qkDim = nKHeads * headKDim     # 2048
      # 1. QKV projection: xNormPtr[nEmb] → tmp0[convDim=8192]
      if lw.wqkvQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.wqkvQ, hp.nEmb, convDim, lw.wqkvQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wq, hp.nEmb, convDim, 1, stream)
      # 2. Gate (z) projection: xNormPtr[nEmb] → tmp2[ssmInnerSize=4096]
      if lw.ssmGateQ != nil:
        gpuLinearColQuant(tmp2, xNormPtr, lw.ssmGateQ, hp.nEmb, hp.ssmInnerSize, lw.ssmGateQType, stream)
      else:
        gpuLinearCol(tmp2, xNormPtr, lw.wo, hp.nEmb, hp.ssmInnerSize, 1, stream)
      # 3. Alpha/beta projections: xNormPtr[nEmb] → [nVHeads=32] each
      #    We place alpha at tmp3[0..nVHeads-1] and beta_raw at tmp3[nVHeads..2*nVHeads-1]
      let alphaPtr = tmp3
      let betaRawPtr = cast[pointer](cast[uint](tmp3) + uint(nVHeads * sizeof(float32)))
      if lw.ssmAlphaQ != nil:
        gpuLinearColQuant(alphaPtr, xNormPtr, lw.ssmAlphaQ, hp.nEmb, nVHeads, lw.ssmAlphaQType, stream)
      else:
        gpuLinearCol(alphaPtr, xNormPtr, lw.wk, hp.nEmb, nVHeads, 1, stream)
      if lw.ssmBetaQ != nil:
        gpuLinearColQuant(betaRawPtr, xNormPtr, lw.ssmBetaQ, hp.nEmb, nVHeads, lw.ssmBetaQType, stream)
      else:
        gpuLinearCol(betaRawPtr, xNormPtr, lw.wv, hp.nEmb, nVHeads, 1, stream)
      # 4. Compute gate = softplus(alpha + dt_bias) * A, beta = sigmoid(beta_raw)
      let gateDecayPtr = cast[pointer](cast[uint](tmp3) + uint(2 * nVHeads * sizeof(float32)))
      let betaSigPtr = cast[pointer](cast[uint](tmp3) + uint(3 * nVHeads * sizeof(float32)))
      gpuDeltaNetGate(gateDecayPtr, betaSigPtr, alphaPtr, lw.ssmDtBias, lw.ssmA, betaRawPtr,
                       nVHeads, stream)
      # 5. Conv1d on QKV mixed: tmp0[convDim] → tmp1[convDim]
      let ssmLayerIdx = cache.ssmState.ssmLayerMap[layer]
      gpuConv1dDecode(cache.ssmState.convState[ssmLayerIdx].devicePtr,
                       tmp0, lw.ssmConv1dW, lw.ssmConv1dBias, tmp1,
                       convDim, hp.ssmConvKernel, stream)
      # 6. SiLU activation on conv output
      gpuSilu(tmp1, convDim, stream)
      # 7. Split Q[nKHeads*headKDim], K[nKHeads*headKDim], V[nVHeads*headVDim] from tmp1
      let qConvPtr = tmp1                # Q at start
      let kConvPtr = cast[pointer](cast[uint](tmp1) + uint(qkDim * sizeof(float32)))
      let vConvPtr = cast[pointer](cast[uint](kConvPtr) + uint(qkDim * sizeof(float32)))
      # 8. L2 normalize Q and K per head
      gpuL2NormPerHead(qConvPtr, nKHeads, headKDim, stream)
      gpuL2NormPerHead(kConvPtr, nKHeads, headKDim, stream)
      # 9. Expand Q, K from nKHeads to nVHeads (repeat interleave)
      #    We need separate buffers: use tmp0 for expanded Q, scratch for expanded K
      let qExpPtr = tmp0
      let kExpPtr = cast[pointer](cast[uint](tmp0) + uint(nVHeads * headKDim * sizeof(float32)))
      gpuExpandHeads(qExpPtr, qConvPtr, nKHeads, nVHeads, headKDim, stream)
      gpuExpandHeads(kExpPtr, kConvPtr, nKHeads, nVHeads, headKDim, stream)
      # 10. Delta Net recurrence: updates state, produces output[nVHeads*headVDim]
      let dnOutPtr = tmp1  # reuse tmp1 for output
      gpuDeltaNetDecode(cache.ssmState.recState[ssmLayerIdx].devicePtr,
                         qExpPtr, kExpPtr, vConvPtr, gateDecayPtr, betaSigPtr, dnOutPtr,
                         nVHeads, headKDim, headVDim, stream)
      # 11. Gated RMS norm: rms_norm(output) * silu(z) using ssm_norm.weight
      #     dnOutPtr=tmp1[ssmInnerSize], z=tmp2[ssmInnerSize] → tmp0[ssmInnerSize]
      gpuGatedRmsNorm(tmp0, dnOutPtr, tmp2, lw.ssmNorm, nVHeads, headVDim, hp.rmsEps, stream)
      # 12. Output projection: tmp0[ssmInnerSize] → tmp1[nEmb]
      if lw.ssmOutQ != nil:
        gpuLinearColQuant(tmp1, tmp0, lw.ssmOutQ, hp.ssmInnerSize, hp.nEmb, lw.ssmOutQType, stream)
      else:
        gpuLinearCol(tmp1, tmp0, lw.wGate, hp.ssmInnerSize, hp.nEmb, 1, stream)
      # First residual + post_attention_norm
      gpuResidualRmsnorm(xNormPtr, xPtr, tmp1, lw.postAttnNorm, hp.nEmb, hp.rmsEps, stream)
      # MoE FFN (same as lkAttnMoe)
      gpuLinearCol(tmp0, xNormPtr, lw.moeRouterW, hp.nEmb, hp.nExperts, 1, stream)
      let eiPtr = tmp3
      let ewPtr = cast[pointer](cast[uint](tmp3) + uint(hp.nExpertsUsed * sizeof(int32)))
      gpuMoeTopK(eiPtr, ewPtr, tmp0, hp.nExperts, hp.nExpertsUsed, stream)
      var expertIndices: array[32, int32]
      var expertWeights: array[32, float32]
      gpuDownloadFromDevice(addr expertIndices[0], eiPtr, hp.nExpertsUsed * sizeof(int32), stream)
      gpuDownloadFromDevice(addr expertWeights[0], ewPtr, hp.nExpertsUsed * sizeof(float32), stream)
      gpuStreamSync(stream)
      gpuZeroBuffer(tmp2, hp.nEmb, stream)
      for e in 0 ..< hp.nExpertsUsed:
        let eidx = int(expertIndices[e])
        let gateOff = cast[pointer](cast[uint](lw.moeGateExpsQ) + uint(eidx * lw.moeGateExpsSliceBytes))
        gpuLinearColQuant(tmp0, xNormPtr, gateOff, hp.nEmb, hp.expertFfnDim, lw.moeGateExpsQType, stream)
        let upOff = cast[pointer](cast[uint](lw.moeUpExpsQ) + uint(eidx * lw.moeUpExpsSliceBytes))
        gpuLinearColQuant(tmp1, xNormPtr, upOff, hp.nEmb, hp.expertFfnDim, lw.moeUpExpsQType, stream)
        gpuSiluMul(tmp0, tmp0, tmp1, hp.expertFfnDim, stream)
        let downOff = cast[pointer](cast[uint](lw.moeDownExpsQ) + uint(eidx * lw.moeDownExpsSliceBytes))
        gpuLinearColQuant(tmp1, tmp0, downOff, hp.expertFfnDim, hp.nEmb, lw.moeDownExpsQType, stream)
        gpuScaleAdd(tmp2, tmp1, expertWeights[e], hp.nEmb, stream)
      # Shared expert
      if lw.moeShGateQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.moeShGateQ, hp.nEmb, hp.expertFfnDim, lw.moeShGateQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wGate, hp.nEmb, hp.expertFfnDim, 1, stream)
      if lw.moeShUpQ != nil:
        gpuLinearColQuant(tmp1, xNormPtr, lw.moeShUpQ, hp.nEmb, hp.expertFfnDim, lw.moeShUpQType, stream)
      else:
        gpuLinearCol(tmp1, xNormPtr, lw.wUp, hp.nEmb, hp.expertFfnDim, 1, stream)
      gpuSiluMul(tmp0, tmp0, tmp1, hp.expertFfnDim, stream)
      if lw.moeShDownQ != nil:
        gpuLinearColQuant(tmp1, tmp0, lw.moeShDownQ, hp.expertFfnDim, hp.nEmb, lw.moeShDownQType, stream)
      else:
        gpuLinearCol(tmp1, tmp0, lw.wDown, hp.expertFfnDim, hp.nEmb, 1, stream)
      gpuSharedExpertGate(tmp1, xNormPtr, lw.moeShGateScalar, hp.nEmb, stream)
      gpuAdd(tmp0, tmp2, tmp1, hp.nEmb, stream)

    if layer < hp.nLayer - 1:
      gpuResidualRmsnorm(xNormPtr, xPtr, tmp0,
                          modelPtrs.layers[layer + 1].attnNorm,
                          hp.nEmb, hp.rmsEps, stream)
    else:
      gpuAdd(xPtr, xPtr, tmp0, hp.nEmb, stream)

  gpuRmsnormCols(xNormPtr, xPtr, modelPtrs.normWeight, hp.nEmb, 1, hp.rmsEps, stream)
  if modelPtrs.outputWeightQ != nil:
    gpuLinearColQuant(xPtr, xNormPtr, modelPtrs.outputWeightQ,
                      modelPtrs.outputShape0, modelPtrs.outputShape1,
                      modelPtrs.outputQType, stream)
  else:
    gpuLinearCol(xPtr, xNormPtr, modelPtrs.outputWeight, modelPtrs.outputShape0, modelPtrs.outputShape1, 1, stream)

  result = gpuArgmax(xPtr, modelPtrs.outputShape1, stream)
  cache.curLen = pos + 1
  cache.gpuCache.curLen = cache.curLen
