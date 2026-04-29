## Naive GPU backend: verbatim port of tinylama's forward_hippo.nim.
##
## All model weights, activations, and KV caches live on GPU.
## Only the initial token IDs are uploaded and final logits downloaded.

import
  std/[tables, math],
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
  HippoMaxDecodeCols = 5632

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
    of GgmlTypeQ2K: rowSizeQ2K(nCols)
    of GgmlTypeQ3K: rowSizeQ3K(nCols)
    of GgmlTypeQ4K: rowSizeQ4K(nCols)
    of GgmlTypeQ6K: rowSizeQ6K(nCols)
    of GgmlTypeQ8_0: rowSizeQ8_0(nCols)
    else: raise newException(ValueError, "unsupported quant type for GPU upload: " & $info.elemType)
  let totalBytes = rowSize * nRows
  let alloc = hippoMalloc(totalBytes)
  hippoMemcpyAsync(alloc.p, dataPtr, totalBytes, HippoMemcpyHostToDevice, gpuCtx.stream)
  result = GpuQuantWeight(devicePtr: alloc.p, alloc: alloc, sizeBytes: totalBytes,
                          nRows: nRows, nCols: nCols)
  quantWeightCache[name] = result

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
  of GgmlTypeQ2K: gpuLinearColQ2K(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ3K: gpuLinearColQ3K(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ4K: gpuLinearColQ4K(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ6K: gpuLinearColQ6K(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ8_0: gpuLinearColQ8_0(dst, x, wQuant, wCols, wRows, stream)
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
    let idx0 = hOffset + 2 * i
    let idx1 = hOffset + 2 * i + 1
    let v0 = x[idx0]
    let v1 = x[idx1]
    x[idx0] = v0 * c - v1 * s
    x[idx1] = v0 * s + v1 * c
  else:
    var p = 0
    while p < int(seqLen):
      let idx0 = (hOffset + 2 * i) * int(seqLen) + p
      let idx1 = (hOffset + 2 * i + 1) * int(seqLen) + p
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
  let idx0 = hOffset + 2'i32 * i
  let idx1 = hOffset + 2'i32 * i + 1'i32
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
proc attentionDecodeKernel(
  qData, kCacheData, vCacheData, outData: ptr float32,
  nHead, nHeadKv, headDim, curLen, cacheCols: cint,
  invSqrtHeadDim: float32
) {.hippoGlobal.} =
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
  var scores {.hippoShared.}: array[2048, float32]
  var sMax {.hippoShared.}: array[HippoBlockSize, float32]
  var sSum {.hippoShared.}: array[HippoBlockSize, float32]
  var localMax = -1e30'f32
  var j = tid
  while j < int(curLen):
    var dot = 0.0'f32
    for d in 0 ..< int(headDim):
      let qIdx = hOff + d
      let kIdx = (kvh * int(headDim) + d) * int(cacheCols) + j
      dot = dot + q[qIdx] * kc[kIdx]
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
  while j < int(curLen):
    let e = expf(scores[j] - globalMax)
    scores[j] = e
    localSum = localSum + e
    j = j + int(blockDim.x)
  sSum[tid] = localSum
  hippoSyncthreads()
  reduceSum256(sSum, tid)
  let invSum = 1.0'f32 / sSum[0]
  var d = tid
  while d < int(headDim):
    var acc = 0.0'f32
    for jj in 0 ..< int(curLen):
      let vIdx = (kvh * int(headDim) + d) * int(cacheCols) + jj
      acc = acc + scores[jj] * vc[vIdx]
    let outIdx = hOff + d
    o[outIdx] = acc * invSum
    d = d + int(blockDim.x)

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
proc initGpuKvCache*(nLayer, nHeadKv, headDim, maxLen: int): GpuKvCache =
  ## Allocate GPU-resident KV cache tensors.
  ensureGpuContext()
  let kvDim = nHeadKv * headDim
  result.maxLen = maxLen
  result.curLen = 0
  result.nHeadKv = nHeadKv
  result.headDim = headDim
  result.k = newSeq[GpuTensor](nLayer)
  result.v = newSeq[GpuTensor](nLayer)
  for i in 0 ..< nLayer:
    result.k[i] = newGpuTensor(@[kvDim, maxLen])
    result.v[i] = newGpuTensor(@[kvDim, maxLen])

# ---------------------------------------------------------------------------
# Weight upload and model GPU pointer setup
# ---------------------------------------------------------------------------
proc ensureModelGpuPtrs*(m: var Model, hp: HParams) =
  ## Populate modelPtrs once by caching all weight device pointers.
  if modelPtrs.initialized:
    return
  ensureGpuContext()

  var tokEmb: Tensor
  try:
    tokEmb = m.getTensor("token_embd.weight")
  except KeyError:
    tokEmb = m.getTensor("tok_embeddings.weight")
  modelPtrs.tokEmb = cachedWeight("token_embd_or_tok_embeddings", tokEmb).devicePtr

  modelPtrs.layers = newSeq[LayerGpuPtrs](hp.nLayer)
  for layer in 0 ..< hp.nLayer:
    let lp = "blk." & $layer & "."
    var lw: LayerGpuPtrs
    lw.attnNorm = cachedWeight(lp & "attn_norm.weight", m.getTensor(lp & "attn_norm.weight")).devicePtr
    lw.ffnNorm = cachedWeight(lp & "ffn_norm.weight", m.getTensor(lp & "ffn_norm.weight")).devicePtr

    template uploadWeight(fp32Field, quantField, qtypeField: untyped, tensorSuffix: string) =
      let tn = lp & tensorSuffix
      let et = m.infos[tn].elemType.int32
      if et == GgmlTypeQ2K.int32 or et == GgmlTypeQ3K.int32 or
         et == GgmlTypeQ4K.int32 or et == GgmlTypeQ6K.int32 or
         et == GgmlTypeQ8_0.int32:
        let qw = cachedQuantWeight(tn, m, tn)
        quantField = qw.devicePtr
        fp32Field = nil
        qtypeField = et
      else:
        fp32Field = cachedWeight(tn, m.getTensor(tn)).devicePtr
        quantField = nil
        qtypeField = 0

    uploadWeight(lw.wq, lw.wqQ, lw.wqQType, "attn_q.weight")
    uploadWeight(lw.wk, lw.wkQ, lw.wkQType, "attn_k.weight")
    uploadWeight(lw.wv, lw.wvQ, lw.wvQType, "attn_v.weight")
    uploadWeight(lw.wo, lw.woQ, lw.woQType, "attn_output.weight")
    uploadWeight(lw.wGate, lw.wGateQ, lw.wGateQType, "ffn_gate.weight")
    uploadWeight(lw.wUp, lw.wUpQ, lw.wUpQType, "ffn_up.weight")
    uploadWeight(lw.wDown, lw.wDownQ, lw.wDownQType, "ffn_down.weight")
    lw.wColsQ = hp.nEmb
    lw.wColsDown = hp.nFfn
    modelPtrs.layers[layer] = lw

  var norm: Tensor
  try:
    norm = m.getTensor("output_norm.weight")
  except KeyError:
    norm = m.getTensor("norm.weight")
  modelPtrs.normWeight = cachedWeight("norm_or_output_norm.weight", norm).devicePtr

  let outTensorName = if m.infos.hasKey("output.weight"): "output.weight"
                      else: "token_embd.weight"
  let outElemType = m.infos[outTensorName].elemType.int32
  if outElemType == GgmlTypeQ2K.int32 or outElemType == GgmlTypeQ3K.int32 or
     outElemType == GgmlTypeQ4K.int32 or outElemType == GgmlTypeQ6K.int32 or
     outElemType == GgmlTypeQ8_0.int32:
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
  ## Initialize CPU + GPU KV cache.
  if hp.nHead <= 0:
    raise newException(ValueError, "KV cache requires llama-style head_count")
  result.maxLen = maxLen
  result.curLen = 0
  result.nHeadKv = hp.nHeadKv
  result.headDim = hp.nEmb div hp.nHead
  let kvDim = hp.nHeadKv * result.headDim
  result.k = newSeq[Tensor](hp.nLayer)
  result.v = newSeq[Tensor](hp.nLayer)
  for i in 0 ..< hp.nLayer:
    result.k[i] = newTensor(@[kvDim, maxLen])
    result.v[i] = newTensor(@[kvDim, maxLen])
  result.gpuCache = initGpuKvCache(hp.nLayer, hp.nHeadKv, result.headDim, maxLen)

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
  ## Run prefill forward pass on GPU for the given token sequence.
  let hp = m.hparams
  if hp.arch != "" and hp.arch != "llama":
    raise newException(ValueError, "unsupported architecture: " & hp.arch)
  if hp.nHeadKv != 0 and (hp.nHead mod hp.nHeadKv) != 0:
    raise newException(ValueError, "GQA requires head_count divisible by head_count_kv")
  if tokens.len == 0:
    raise newException(ValueError, "prefill requires at least one token")
  if tokens.len > cache.maxLen:
    raise newException(ValueError, "prefill exceeds KV cache capacity")

  let headDim = hp.nEmb div hp.nHead
  let ropeDim = if hp.ropeDim > 0: hp.ropeDim else: headDim
  let kvDim = hp.nHeadKv * headDim
  let seqLen = tokens.len

  ensureGpuContext()
  let maxRows = max(max(hp.nEmb, hp.nFfn), hp.nVocab)
  ensureActivationBuffers(maxRows * seqLen)
  ensureScratchBuffers(maxRows * seqLen)
  let stream = gpuCtx.stream

  let tokEmb = getTensorOr(m, "tok_embeddings.weight", "token_embd.weight")
  let dTokEmb = cachedWeight("token_embd_or_tok_embeddings", tokEmb)
  let tokenPtr = gpuUploadInt32Pooled(unsafeAddr tokens[0], seqLen, stream)

  var xPtr = gpuCtx.act0.devicePtr
  let xNormPtr = gpuCtx.act1.devicePtr
  let tmp0 = gpuCtx.scratch0.devicePtr
  let tmp1 = gpuCtx.scratch1.devicePtr
  let tmp2 = gpuCtx.scratch2.devicePtr

  gpuEmbedding(xPtr, dTokEmb.devicePtr, cast[ptr int32](tokenPtr),
               hp.nEmb, seqLen, hp.nVocab, stream)

  for layer in 0 ..< hp.nLayer:
    let lp = "blk." & $layer & "."
    let dAttnNorm = cachedWeight(lp & "attn_norm.weight", m.getTensor(lp & "attn_norm.weight"))
    let dFfnNorm = cachedWeight(lp & "ffn_norm.weight", m.getTensor(lp & "ffn_norm.weight"))
    let dWq = cachedWeight(lp & "attn_q.weight", m.getTensor(lp & "attn_q.weight"))
    let dWk = cachedWeight(lp & "attn_k.weight", m.getTensor(lp & "attn_k.weight"))
    let dWv = cachedWeight(lp & "attn_v.weight", m.getTensor(lp & "attn_v.weight"))
    let dWo = cachedWeight(lp & "attn_output.weight", m.getTensor(lp & "attn_output.weight"))
    let dWGate = cachedWeight(lp & "ffn_gate.weight", m.getTensor(lp & "ffn_gate.weight"))
    let dWUp = cachedWeight(lp & "ffn_up.weight", m.getTensor(lp & "ffn_up.weight"))
    let dWDown = cachedWeight(lp & "ffn_down.weight", m.getTensor(lp & "ffn_down.weight"))

    gpuRmsnormCols(xNormPtr, xPtr, dAttnNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
    gpuLinearCol(tmp0, xNormPtr, dWq.devicePtr, hp.nEmb, hp.nEmb, seqLen, stream)
    gpuLinearCol(tmp1, xNormPtr, dWk.devicePtr, hp.nEmb, kvDim, seqLen, stream)
    gpuLinearCol(tmp2, xNormPtr, dWv.devicePtr, hp.nEmb, kvDim, seqLen, stream)
    gpuRopeAtPos(tmp0, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, 0, seqLen, stream)
    gpuRopeAtPos(tmp1, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, 0, seqLen, stream)

    gpuStoreKV(cache.gpuCache.k[layer].devicePtr, tmp1, kvDim, seqLen, cache.gpuCache.maxLen, 0, stream)
    gpuStoreKV(cache.gpuCache.v[layer].devicePtr, tmp2, kvDim, seqLen, cache.gpuCache.maxLen, 0, stream)

    gpuAttentionPrefill(xNormPtr, tmp0, tmp1, tmp2,
                        hp.nHead, hp.nHeadKv, headDim, seqLen, stream)
    gpuLinearCol(tmp0, xNormPtr, dWo.devicePtr, hp.nEmb, hp.nEmb, seqLen, stream)
    gpuAdd(xPtr, xPtr, tmp0, hp.nEmb * seqLen, stream)

    gpuRmsnormCols(xNormPtr, xPtr, dFfnNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
    gpuLinearCol(tmp0, xNormPtr, dWGate.devicePtr, hp.nEmb, hp.nFfn, seqLen, stream)
    gpuLinearCol(tmp1, xNormPtr, dWUp.devicePtr, hp.nEmb, hp.nFfn, seqLen, stream)
    gpuSiluMul(tmp2, tmp0, tmp1, hp.nFfn * seqLen, stream)
    gpuLinearCol(tmp0, tmp2, dWDown.devicePtr, hp.nFfn, hp.nEmb, seqLen, stream)
    gpuAdd(xPtr, xPtr, tmp0, hp.nEmb * seqLen, stream)

  let norm = getTensorOr(m, "norm.weight", "output_norm.weight")
  let outTName = if m.infos.hasKey("output.weight"): "output.weight"
                 else: "token_embd.weight"
  let outW = outputWeightForLinear(m.getTensor(outTName), hp.nEmb, hp.nVocab)
  let dNorm = cachedWeight("norm_or_output_norm.weight", norm)
  let dOutW = cachedWeight("output.weight", outW)

  gpuRmsnormCols(xNormPtr, xPtr, dNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
  gpuLinearCol(xPtr, xNormPtr, dOutW.devicePtr, outW.shape[0], outW.shape[1], seqLen, stream)

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
  ## Run single-token decode forward pass on GPU.
  let hp = m.hparams
  if hp.arch != "" and hp.arch != "llama":
    raise newException(ValueError, "unsupported architecture: " & hp.arch)
  if hp.nHeadKv != 0 and (hp.nHead mod hp.nHeadKv) != 0:
    raise newException(ValueError, "GQA requires head_count divisible by head_count_kv")
  if cache.curLen >= cache.maxLen:
    raise newException(ValueError, "KV cache full")

  let headDim = hp.nEmb div hp.nHead
  let ropeDim = if hp.ropeDim > 0: hp.ropeDim else: headDim
  let kvDim = hp.nHeadKv * headDim
  let pos = cache.curLen

  ensureGpuContext()
  let maxRows = max(max(hp.nEmb, hp.nFfn), hp.nVocab)
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
    if lw.wqQ != nil:
      gpuLinearColQuant(tmp0, xNormPtr, lw.wqQ, hp.nEmb, hp.nEmb, lw.wqQType, stream)
    else:
      gpuLinearCol(tmp0, xNormPtr, lw.wq, hp.nEmb, hp.nEmb, 1, stream)
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
      gpuLinearColQuant(tmp0, xNormPtr, lw.woQ, hp.nEmb, hp.nEmb, lw.woQType, stream)
    else:
      gpuLinearCol(tmp0, xNormPtr, lw.wo, hp.nEmb, hp.nEmb, 1, stream)
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
  ## Run single-token decode and return argmax token ID without downloading logits.
  let hp = m.hparams
  if hp.arch != "" and hp.arch != "llama":
    raise newException(ValueError, "unsupported architecture: " & hp.arch)
  if hp.nHeadKv != 0 and (hp.nHead mod hp.nHeadKv) != 0:
    raise newException(ValueError, "GQA requires head_count divisible by head_count_kv")
  if cache.curLen >= cache.maxLen:
    raise newException(ValueError, "KV cache full")

  let headDim = hp.nEmb div hp.nHead
  let ropeDim = if hp.ropeDim > 0: hp.ropeDim else: headDim
  let kvDim = hp.nHeadKv * headDim
  let pos = cache.curLen

  ensureGpuContext()
  let maxRows = max(max(hp.nEmb, hp.nFfn), hp.nVocab)
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

  gpuEmbedding(xPtr, modelPtrs.tokEmb, cast[ptr int32](tokenPtr),
               hp.nEmb, 1, hp.nVocab, stream)

  gpuRmsnormCols(xNormPtr, xPtr, modelPtrs.layers[0].attnNorm, hp.nEmb, 1, hp.rmsEps, stream)

  for layer in 0 ..< hp.nLayer:
    let lw = modelPtrs.layers[layer]

    if lw.wqQ != nil:
      gpuLinearColQuant(tmp0, xNormPtr, lw.wqQ, hp.nEmb, hp.nEmb, lw.wqQType, stream)
    else:
      gpuLinearCol(tmp0, xNormPtr, lw.wq, hp.nEmb, hp.nEmb, 1, stream)
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
    gpuRopeQKDecode(tmp0, tmp1, hp.nHead, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, pos, stream)
    gpuStoreKVPair(cache.gpuCache.k[layer].devicePtr, tmp1,
                    cache.gpuCache.v[layer].devicePtr, tmp2,
                    kvDim, 1, cache.gpuCache.maxLen, pos, stream)
    gpuAttentionDecode(xNormPtr, tmp0, cache.gpuCache.k[layer].devicePtr,
                       cache.gpuCache.v[layer].devicePtr,
                       hp.nHead, hp.nHeadKv, headDim, pos + 1,
                       cache.gpuCache.maxLen, stream)
    if lw.woQ != nil:
      gpuLinearColQuant(tmp0, xNormPtr, lw.woQ, hp.nEmb, hp.nEmb, lw.woQType, stream)
    else:
      gpuLinearCol(tmp0, xNormPtr, lw.wo, hp.nEmb, hp.nEmb, 1, stream)
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
