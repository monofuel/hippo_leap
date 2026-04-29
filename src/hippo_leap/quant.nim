## Dequantization helpers for GGUF tensors.

import std/[math]
when cpuEndian != littleEndian:
  import std/endians

const
  QK_K* = 256
  QK8_0* = 32
  BlockQ2KSize* = 2 + 2 + (QK_K div 16) + (QK_K div 4)
  BlockQ3KSize* = 2 + (QK_K div 4) + (QK_K div 8) + 12
  BlockQ4KSize* = 2 + 2 + 12 + (QK_K div 2)   # d + dmin + scales + 4-bit quants
  BlockQ6KSize* = 2 + (QK_K div 16) + 3 * (QK_K div 4)
  BlockQ8_0Size* = 2 + QK8_0                    # fp16 scale + int8 quants

proc rowSizeQ2K*(rowLen: int): int =
  ## Compute byte size of a Q2_K quantized row.
  if (rowLen mod QK_K) != 0:
    raise newException(ValueError, "q2_K row size must be multiple of 256")
  (rowLen div QK_K) * BlockQ2KSize

proc rowSizeQ3K*(rowLen: int): int =
  ## Compute byte size of a Q3_K quantized row.
  if (rowLen mod QK_K) != 0:
    raise newException(ValueError, "q3_K row size must be multiple of 256")
  (rowLen div QK_K) * BlockQ3KSize

proc rowSizeQ4K*(rowLen: int): int =
  if (rowLen mod QK_K) != 0:
    raise newException(ValueError, "q4_K row size must be multiple of 256")
  (rowLen div QK_K) * BlockQ4KSize

proc rowSizeQ6K*(rowLen: int): int =
  ## Compute byte size of a Q6_K quantized row.
  if (rowLen mod QK_K) != 0:
    raise newException(ValueError, "q6_K row size must be multiple of 256")
  (rowLen div QK_K) * BlockQ6KSize

proc rowSizeQ8_0*(rowLen: int): int =
  if (rowLen mod QK8_0) != 0:
    raise newException(ValueError, "q8_0 row size must be multiple of 32")
  (rowLen div QK8_0) * BlockQ8_0Size

proc halfToFloat*(h: uint16): float32 =
  ## Convert IEEE 754 half-precision to float32.
  let s = (h shr 15) and 0x1
  let e = (h shr 10) and 0x1F
  let f = h and 0x3FF
  if e == 0:
    if f == 0:
      return (if s == 1: -0.0'f32 else: 0.0'f32)
    return (if s == 1: -1.0'f32 else: 1.0'f32) *
      pow(2.0'f32, -14.0'f32) * (float32(f) / 1024.0'f32)
  elif e == 31:
    if f == 0:
      return (if s == 1: -Inf else: Inf)
    return NaN
  else:
    let exp = int(e) - 15
    let mant = 1.0'f32 + float32(f) / 1024.0'f32
    let sign = (if s == 1: -1.0'f32 else: 1.0'f32)
    return sign * mant * pow(2.0'f32, float32(exp))

proc dequantRowQ2K*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  ## Dequantize a row of k elements in Q2_K format.
  if (k mod QK_K) != 0:
    raise newException(ValueError, "q2_K row size must be multiple of 256")
  let nBlocks = k div QK_K
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    let scalesPtr = cast[ptr UncheckedArray[uint8]](addr src[offset])
    let qsPtr = cast[ptr UncheckedArray[uint8]](addr src[offset + (QK_K div 16)])
    var dRaw = cast[ptr UncheckedArray[uint16]](addr src[offset + (QK_K div 16) + (QK_K div 4)])[0]
    var dminRaw = cast[ptr UncheckedArray[uint16]](addr src[offset + (QK_K div 16) + (QK_K div 4)])[1]
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
      dminRaw = swapEndian(dminRaw)
    let d = halfToFloat(dRaw)
    let dmin = halfToFloat(dminRaw)
    var scaleIdx = 0
    var qOffset = 0
    for n in 0 ..< 2:
      var shift = 0
      for _ in 0 ..< 4:
        var sc = scalesPtr[scaleIdx]
        inc scaleIdx
        var dl = d * float32(sc and 0x0F)
        var ml = dmin * float32(sc shr 4)
        for l in 0 ..< 16:
          dst[outIdx] = dl * float32((qsPtr[qOffset + l] shr shift) and 3) - ml
          inc outIdx
        sc = scalesPtr[scaleIdx]
        inc scaleIdx
        dl = d * float32(sc and 0x0F)
        ml = dmin * float32(sc shr 4)
        for l in 0 ..< 16:
          dst[outIdx] = dl * float32((qsPtr[qOffset + 16 + l] shr shift) and 3) - ml
          inc outIdx
        shift += 2
      qOffset += 32
    offset += BlockQ2KSize

proc dequantRowQ3K*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  ## Dequantize a row of k elements in Q3_K format.
  if (k mod QK_K) != 0:
    raise newException(ValueError, "q3_K row size must be multiple of 256")
  let nBlocks = k div QK_K
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    let hmask = cast[ptr UncheckedArray[uint8]](addr src[offset])
    let qs = cast[ptr UncheckedArray[uint8]](addr src[offset + (QK_K div 8)])
    var scalesBytes: array[12, byte]
    copyMem(addr scalesBytes[0], addr src[offset + (QK_K div 8) + (QK_K div 4)], 12)
    var dRaw = cast[ptr UncheckedArray[uint16]](addr src[offset + (QK_K div 8) + (QK_K div 4) + 12])[0]
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
    let dAll = halfToFloat(dRaw)

    var aux: array[4, uint32]
    for i in 0 ..< 3:
      let b = i * 4
      aux[i] = uint32(scalesBytes[b]) or (uint32(scalesBytes[b + 1]) shl 8) or
               (uint32(scalesBytes[b + 2]) shl 16) or (uint32(scalesBytes[b + 3]) shl 24)
    let kmask1 = 0x03030303'u32
    let kmask2 = 0x0f0f0f0f'u32
    let tmp = aux[2]
    aux[2] = ((aux[0] shr 4) and kmask2) or (((tmp shr 4) and kmask1) shl 4)
    aux[3] = ((aux[1] shr 4) and kmask2) or (((tmp shr 6) and kmask1) shl 4)
    aux[0] = (aux[0] and kmask2) or (((tmp shr 0) and kmask1) shl 4)
    aux[1] = (aux[1] and kmask2) or (((tmp shr 2) and kmask1) shl 4)

    var scales: array[16, int8]
    for i in 0 ..< 4:
      let v = aux[i]
      scales[i * 4 + 0] = int8((v shr 0) and 0xFF)
      scales[i * 4 + 1] = int8((v shr 8) and 0xFF)
      scales[i * 4 + 2] = int8((v shr 16) and 0xFF)
      scales[i * 4 + 3] = int8((v shr 24) and 0xFF)

    var scaleIdx = 0
    var m: uint8 = 1
    var qOffset = 0
    for _ in 0 ..< 2:
      var shift = 0
      for _ in 0 ..< 4:
        let dl = dAll * float32(scales[scaleIdx] - 32)
        inc scaleIdx
        for l in 0 ..< 16:
          let qv = int8((qs[qOffset + l] shr shift) and 3)
          let hm = (if (hmask[l] and m) != 0: 0 else: 4)
          dst[outIdx] = dl * float32(qv - hm)
          inc outIdx
        let dl2 = dAll * float32(scales[scaleIdx] - 32)
        inc scaleIdx
        for l in 0 ..< 16:
          let qv = int8((qs[qOffset + 16 + l] shr shift) and 3)
          let hm = (if (hmask[16 + l] and m) != 0: 0 else: 4)
          dst[outIdx] = dl2 * float32(qv - hm)
          inc outIdx
        shift += 2
        m = m shl 1
      qOffset += 32
    offset += BlockQ3KSize

proc getScaleMinK4(j: int, scales: ptr UncheckedArray[uint8], sc, mn: var uint8) =
  ## Unpack 6-bit scale and min from Q4_K packed scales array (12 bytes → 8 pairs).
  if j < 4:
    sc = scales[j] and 63
    mn = scales[j + 4] and 63
  else:
    sc = (scales[j + 4] and 0x0F) or ((scales[j - 4] shr 6) shl 4)
    mn = (scales[j + 4] shr 4) or ((scales[j] shr 6) shl 4)

proc dequantRowQ4K*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  if (k mod QK_K) != 0:
    raise newException(ValueError, "q4_K row size must be multiple of 256")
  let nBlocks = k div QK_K
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    var dRaw = cast[ptr UncheckedArray[uint16]](addr src[offset])[0]
    var dminRaw = cast[ptr UncheckedArray[uint16]](addr src[offset + 2])[0]
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
      dminRaw = swapEndian(dminRaw)
    let d = halfToFloat(dRaw)
    let dmin = halfToFloat(dminRaw)
    let scales = cast[ptr UncheckedArray[uint8]](addr src[offset + 4])
    let q = cast[ptr UncheckedArray[uint8]](addr src[offset + 16])
    var isIdx = 0
    var qOff = 0
    for j in countup(0, QK_K - 1, 64):
      var sc, mn: uint8
      getScaleMinK4(isIdx, scales, sc, mn)
      let d1 = d * float32(sc)
      let m1 = dmin * float32(mn)
      getScaleMinK4(isIdx + 1, scales, sc, mn)
      let d2 = d * float32(sc)
      let m2 = dmin * float32(mn)
      for l in 0 ..< 32:
        dst[outIdx + l] = d1 * float32(q[qOff + l] and 0x0F) - m1
      for l in 0 ..< 32:
        dst[outIdx + 32 + l] = d2 * float32(q[qOff + l] shr 4) - m2
      outIdx += 64
      qOff += 32
      isIdx += 2
    offset += BlockQ4KSize

proc dequantRowQ8_0*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  if (k mod QK8_0) != 0:
    raise newException(ValueError, "q8_0 row size must be multiple of 32")
  let nBlocks = k div QK8_0
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    var dRaw = cast[ptr UncheckedArray[uint16]](addr src[offset])[0]
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
    let d = halfToFloat(dRaw)
    let qs = cast[ptr UncheckedArray[int8]](addr src[offset + 2])
    for j in 0 ..< QK8_0:
      dst[outIdx + j] = float32(qs[j]) * d
    outIdx += QK8_0
    offset += BlockQ8_0Size

proc dequantRowQ6K*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  ## Dequantize a row of k elements in Q6_K format.
  if (k mod QK_K) != 0:
    raise newException(ValueError, "q6_K row size must be multiple of 256")
  let nBlocks = k div QK_K
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    let ql = cast[ptr UncheckedArray[uint8]](addr src[offset])
    let qh = cast[ptr UncheckedArray[uint8]](addr src[offset + (QK_K div 2)])
    let sc = cast[ptr UncheckedArray[int8]](addr src[offset + (QK_K div 2) + (QK_K div 4)])
    var dRaw = cast[ptr UncheckedArray[uint16]](addr src[offset + (QK_K div 2) + (QK_K div 4) + (QK_K div 16)])[0]
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
    let d = halfToFloat(dRaw)

    var qlOff = 0
    var qhOff = 0
    var scOff = 0
    for _ in 0 ..< 2:
      for l in 0 ..< 32:
        let scaleBlock = l div 16
        let q1 = int8((ql[qlOff + l + 0] and 0xF) or (((qh[qhOff + l] shr 0) and 3) shl 4)) - 32
        let q2 = int8((ql[qlOff + l + 32] and 0xF) or (((qh[qhOff + l] shr 2) and 3) shl 4)) - 32
        let q3 = int8((ql[qlOff + l + 0] shr 4) or (((qh[qhOff + l] shr 4) and 3) shl 4)) - 32
        let q4 = int8((ql[qlOff + l + 32] shr 4) or (((qh[qhOff + l] shr 6) and 3) shl 4)) - 32
        dst[outIdx + l + 0]   = d * float32(sc[scOff + scaleBlock + 0]) * float32(q1)
        dst[outIdx + l + 32]  = d * float32(sc[scOff + scaleBlock + 2]) * float32(q2)
        dst[outIdx + l + 64]  = d * float32(sc[scOff + scaleBlock + 4]) * float32(q3)
        dst[outIdx + l + 96]  = d * float32(sc[scOff + scaleBlock + 6]) * float32(q4)
      outIdx += 128
      qlOff += 64
      qhOff += 32
      scOff += 8
    offset += BlockQ6KSize
