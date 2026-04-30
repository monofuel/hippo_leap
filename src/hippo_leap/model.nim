## Minimal model loader that reads GGUF tensors into float32.

import
  std/[sequtils, tables],
  ./[gguf_loader, tensor, quant]

when cpuEndian != littleEndian:
  import std/endians

const
  GgmlTypeF32* = 0
  GgmlTypeF16* = 1
  GgmlTypeQ5_0* = 6
  GgmlTypeQ5_1* = 7
  GgmlTypeQ8_0* = 8
  GgmlTypeQ2K* = 10
  GgmlTypeQ3K* = 11
  GgmlTypeQ4K* = 12
  GgmlTypeQ5K* = 13
  GgmlTypeQ6K* = 14
  GgmlTypeIQ4NL* = 20

type
  HParams* = object
    arch*: string
    nVocab*: int
    nCtx*: int
    nEmb*: int
    nLayer*: int
    nFfn*: int
    nHead*: int
    nHeadKv*: int
    headDim*: int
    ropeDim*: int
    ropeFreqBase*: float32
    rmsEps*: float32
    ssmConvKernel*: int
    ssmStateSize*: int
    ssmGroupCount*: int
    ssmInnerSize*: int
    ssmDtRank*: int
    layerNFfn*: seq[int]
    layerNHeadKv*: seq[int]
    nExperts*: int
    nExpertsUsed*: int
    expertFfnDim*: int
    sharedExpertFfnDim*: int
    expertWeightsScale*: float32
    fullAttnInterval*: int
    ropeDimSections*: seq[int32]

  Model* = object
    hparams*: HParams
    gguf*: GgufFile
    infos*: Table[string, GgufTensorInfo]
    cache*: Table[string, Tensor]

proc tensorElemCount*(info: GgufTensorInfo): int =
  ## Compute the total number of elements in a tensor.
  var n = 1'u64
  for i in 0 ..< int(info.nDims):
    n *= info.ne[i]
  if n > uint64(high(int)):
    raise newException(ValueError, "tensor too large")
  int(n)

proc tensorShape(info: GgufTensorInfo): seq[int] =
  result = newSeq[int](int(info.nDims))
  for i in 0 ..< int(info.nDims):
    result[i] = int(info.ne[i])

proc loadTensorF32(g: GgufFile, info: GgufTensorInfo): Tensor =
  let count = tensorElemCount(info)
  result = newTensor(tensorShape(info))
  let dataPtr = tensorDataPtr(g, info)
  let rowLen = int(info.ne[0])
  let rows = if rowLen > 0: count div rowLen else: 0
  case info.elemType
  of GgmlTypeF32:
    copyMem(addr result.data[0], addr dataPtr[0], count * 4)
  of GgmlTypeF16:
    for i in 0 ..< count:
      var u = cast[ptr UncheckedArray[uint16]](addr dataPtr[i * 2])[0]
      when cpuEndian != littleEndian:
        u = swapEndian(u)
      result.data[i] = halfToFloat(u)
  of GgmlTypeQ8_0:
    let rowSize = rowSizeQ8_0(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ8_0(src, dst, rowLen)
  of GgmlTypeQ5_0:
    let rowSize = rowSizeQ5_0(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ5_0(src, dst, rowLen)
  of GgmlTypeQ2K:
    let rowSize = rowSizeQ2K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ2K(src, dst, rowLen)
  of GgmlTypeQ3K:
    let rowSize = rowSizeQ3K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ3K(src, dst, rowLen)
  of GgmlTypeQ4K:
    let rowSize = rowSizeQ4K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ4K(src, dst, rowLen)
  of GgmlTypeQ5_1:
    let rowSize = rowSizeQ5_1(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ5_1(src, dst, rowLen)
  of GgmlTypeQ5K:
    let rowSize = rowSizeQ5K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ5K(src, dst, rowLen)
  of GgmlTypeQ6K:
    let rowSize = rowSizeQ6K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ6K(src, dst, rowLen)
  of GgmlTypeIQ4NL:
    let rowSize = rowSizeIQ4NL(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowIQ4NL(src, dst, rowLen)
  else:
    raise newException(ValueError, "unsupported ggml type: " & $info.elemType)

proc loadHParams(g: GgufFile): HParams =
  discard g.getKvStr("general.architecture", result.arch)
  let prefix = if result.arch.len > 0: result.arch & "." else: "llama."
  var v: uint32
  if g.getKvU32(prefix & "vocab_size", v): result.nVocab = int(v)
  if g.getKvU32(prefix & "context_length", v): result.nCtx = int(v)
  if g.getKvU32(prefix & "embedding_length", v): result.nEmb = int(v)
  if g.getKvU32(prefix & "block_count", v): result.nLayer = int(v)
  if g.getKvU32(prefix & "feed_forward_length", v): result.nFfn = int(v)
  if g.getKvU32(prefix & "attention.head_count", v): result.nHead = int(v)
  if g.getKvU32(prefix & "attention.head_count_kv", v): result.nHeadKv = int(v)
  if g.getKvU32(prefix & "attention.key_length", v): result.headDim = int(v)
  if g.getKvU32(prefix & "rope.dimension_count", v): result.ropeDim = int(v)
  var f: float32
  if g.getKvF32(prefix & "rope.freq_base", f): result.ropeFreqBase = f
  if g.getKvF32(prefix & "attention.layer_norm_rms_epsilon", f): result.rmsEps = f
  if g.getKvU32(prefix & "ssm.conv_kernel", v): result.ssmConvKernel = int(v)
  if g.getKvU32(prefix & "ssm.state_size", v): result.ssmStateSize = int(v)
  if g.getKvU32(prefix & "ssm.group_count", v): result.ssmGroupCount = int(v)
  if g.getKvU32(prefix & "ssm.inner_size", v): result.ssmInnerSize = int(v)
  if g.getKvU32(prefix & "ssm.time_step_rank", v): result.ssmDtRank = int(v)
  if g.getKvU32(prefix & "expert_count", v): result.nExperts = int(v)
  if g.getKvU32(prefix & "expert_used_count", v): result.nExpertsUsed = int(v)
  if g.getKvU32(prefix & "expert_feed_forward_length", v): result.expertFfnDim = int(v)
  if g.getKvU32(prefix & "expert_shared_feed_forward_length", v): result.sharedExpertFfnDim = int(v)
  if g.getKvF32(prefix & "expert_weights_scale", f): result.expertWeightsScale = f
  if g.getKvU32(prefix & "full_attention_interval", v): result.fullAttnInterval = int(v)
  var arrI32: seq[int32]
  if g.getKvArrI32(prefix & "feed_forward_length", arrI32):
    result.layerNFfn = arrI32.mapIt(int(it))
  if g.getKvArrI32(prefix & "attention.head_count_kv", arrI32):
    result.layerNHeadKv = arrI32.mapIt(int(it))
  if g.getKvArrI32(prefix & "rope.dimension_sections", arrI32):
    result.ropeDimSections = arrI32
  if result.headDim == 0 and result.nHead > 0 and result.nEmb > 0:
    result.headDim = result.nEmb div result.nHead
  if result.arch == "qwen35moe" and result.ssmGroupCount == 0 and result.ssmStateSize > 0:
    result.ssmGroupCount = result.ssmDtRank  # preliminary, may be corrected after tensor load
  if result.sharedExpertFfnDim == 0 and result.expertFfnDim > 0:
    result.sharedExpertFfnDim = result.expertFfnDim
  if result.expertWeightsScale == 0.0'f32:
    result.expertWeightsScale = 1.0'f32
  if result.ropeFreqBase == 0.0'f32:
    result.ropeFreqBase = 10000.0'f32
  if result.ropeDim == 0:
    result.ropeDim = result.headDim
  if result.nVocab == 0:
    var tokens: seq[string]
    if g.getKvArrStr("tokenizer.ggml.tokens", tokens):
      result.nVocab = tokens.len

proc loadModel*(path: string): Model =
  ## Load a GGUF model file and parse its hyperparameters and tensor index.
  result.gguf = openGguf(path)
  result.hparams = loadHParams(result.gguf)
  result.infos = initTable[string, GgufTensorInfo](result.gguf.tensors.len * 2)
  result.cache = initTable[string, Tensor]()
  for info in result.gguf.tensors:
    result.infos[info.name] = info
  # Derive ssmGroupCount from tensor shapes for qwen35moe (Delta Net nKHeads)
  if result.hparams.arch == "qwen35moe" and result.hparams.ssmInnerSize > 0:
    if result.infos.hasKey("blk.0.ssm_conv1d.weight"):
      let convInfo = result.infos["blk.0.ssm_conv1d.weight"]
      let convDim = int(convInfo.ne[1])  # [kernelSize, convDim]
      let nKHeads = (convDim - result.hparams.ssmInnerSize) div (2 * result.hparams.ssmStateSize)
      result.hparams.ssmGroupCount = nKHeads

proc close*(m: var Model) =
  ## Close the underlying GGUF memory-mapped file.
  m.gguf.close()

proc getTensor*(m: var Model, name: string): lent Tensor =
  ## Get a tensor by name, lazily dequantizing from the GGUF file on first access.
  if not m.cache.hasKey(name):
    if not m.infos.hasKey(name):
      raise newException(KeyError, "missing tensor: " & name)
    m.cache[name] = loadTensorF32(m.gguf, m.infos[name])
  m.cache[name]
