## GPU integration test for Nemotron-H 4B (Mamba2 hybrid SSM+attention).
##
## Requires: nim cpp --cc:hipcc -d:backendNaive -d:useMalloc -d:zippyNoSimd
## Requires: Nemotron-H 4B model file on disk.

import
  std/[os, strformat, strutils, times, tables],
  hippo_leap/[inference, tokenizer, model, sampling]

const
  ModelPathFineTune = "/mnt/steel-chest/LLM/lmstudio/models/monofuel/racha-mini-2026-04-28-nemotron-4b/racha-mini-2026-04-28-nemotron-4b_Q4_K_M.gguf"
  ModelPathStock = "/mnt/steel-chest/LLM/lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"
  MaxContextLen = 256

let ModelPath = if fileExists(ModelPathFineTune): ModelPathFineTune
                else: ModelPathStock

proc testHParams() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found: " & ModelPath
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  let hp = m.hparams
  doAssert hp.arch == "nemotron_h", "expected arch nemotron_h, got " & hp.arch
  doAssert hp.nLayer == 42, &"expected 42 layers, got {hp.nLayer}"
  doAssert hp.nEmb == 3136, &"expected 3136 emb, got {hp.nEmb}"
  doAssert hp.nHead == 40, &"expected 40 heads, got {hp.nHead}"
  doAssert hp.headDim == 128, &"expected 128 headDim, got {hp.headDim}"
  doAssert hp.nVocab == 131072, &"expected 131072 vocab, got {hp.nVocab}"
  doAssert hp.ropeDim == 78, &"expected 78 ropeDim, got {hp.ropeDim}"

  doAssert hp.ssmConvKernel == 4, &"expected ssmConvKernel=4, got {hp.ssmConvKernel}"
  doAssert hp.ssmStateSize == 128, &"expected ssmStateSize=128, got {hp.ssmStateSize}"
  doAssert hp.ssmGroupCount == 8, &"expected ssmGroupCount=8, got {hp.ssmGroupCount}"
  doAssert hp.ssmInnerSize == 7680, &"expected ssmInnerSize=7680, got {hp.ssmInnerSize}"
  doAssert hp.ssmDtRank == 96, &"expected ssmDtRank=96, got {hp.ssmDtRank}"

  doAssert hp.layerNFfn.len == 42, &"expected 42 layerNFfn, got {hp.layerNFfn.len}"
  doAssert hp.layerNHeadKv.len == 42, &"expected 42 layerNHeadKv, got {hp.layerNHeadKv.len}"

  # Layer 0 should be SSM (no attention, no FFN)
  doAssert hp.layerNHeadKv[0] == 0, &"layer 0 should have 0 kv heads, got {hp.layerNHeadKv[0]}"
  doAssert hp.layerNFfn[0] == 0, &"layer 0 should have 0 ffn, got {hp.layerNFfn[0]}"

  # Layer 12 should be attention
  doAssert hp.layerNHeadKv[12] == 8, &"layer 12 should have 8 kv heads, got {hp.layerNHeadKv[12]}"
  doAssert hp.layerNFfn[12] == 0, &"layer 12 should have 0 ffn, got {hp.layerNFfn[12]}"

  # Layer 1 should be FFN
  doAssert hp.layerNFfn[1] == 12544, &"layer 1 should have ffn=12544, got {hp.layerNFfn[1]}"
  doAssert hp.layerNHeadKv[1] == 0, &"layer 1 should have 0 kv heads, got {hp.layerNHeadKv[1]}"

  echo &"[OK] Nemotron-H hparams: {hp.nLayer} layers, {hp.nEmb} emb, headDim={hp.headDim}, {hp.nVocab} vocab"
  echo &"     SSM: convK={hp.ssmConvKernel} state={hp.ssmStateSize} groups={hp.ssmGroupCount} inner={hp.ssmInnerSize} dtRank={hp.ssmDtRank}"

proc testLayerTypes() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  var nSsm, nAttn, nFfn = 0
  for layer in 0 ..< m.hparams.nLayer:
    let lp = "blk." & $layer & "."
    if m.infos.hasKey(lp & "ssm_in.weight"):
      inc nSsm
    elif m.hparams.layerNHeadKv[layer] > 0:
      inc nAttn
    elif m.hparams.layerNFfn[layer] > 0:
      inc nFfn

  doAssert nSsm == 21, &"expected 21 SSM layers, got {nSsm}"
  doAssert nAttn == 4, &"expected 4 attention layers, got {nAttn}"
  doAssert nFfn == 17, &"expected 17 FFN layers, got {nFfn}"
  echo &"[OK] Layer types: {nSsm} SSM, {nAttn} attention, {nFfn} FFN"

proc testTokenizer() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  var m = loadModel(ModelPath)
  defer: m.close()
  let vocab = loadVocab(m.gguf)

  doAssert vocab.tokenToId.hasKey("<|im_start|>"), "missing <|im_start|> token"
  doAssert vocab.tokenToId.hasKey("<|im_end|>"), "missing <|im_end|> token"

  let formatted = vocab.formatChatPrompt("Hello")
  doAssert formatted.contains("<|im_start|>"), "missing <|im_start|> in chat template"
  doAssert formatted.contains("<|im_end|>"), "missing <|im_end|> in chat template"
  echo "[OK] Nemotron-H tokenizer + ChatML template"

proc testWeightShapes() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  # SSM layer 0
  let ssmIn = m.infos["blk.0.ssm_in.weight"]
  doAssert ssmIn.ne[0] == 3136, &"ssm_in cols should be 3136, got {ssmIn.ne[0]}"
  doAssert ssmIn.ne[1] == 17504, &"ssm_in rows should be 17504, got {ssmIn.ne[1]}"

  let ssmConv = m.infos["blk.0.ssm_conv1d.weight"]
  doAssert ssmConv.ne[0] == 4, &"conv1d kernel should be 4, got {ssmConv.ne[0]}"
  doAssert ssmConv.ne[1] == 9728, &"conv1d channels should be 9728, got {ssmConv.ne[1]}"

  # Attention layer 12
  let wq = m.infos["blk.12.attn_q.weight"]
  doAssert wq.ne[0] == 3136, &"attn_q cols should be 3136, got {wq.ne[0]}"
  doAssert wq.ne[1] == 5120, &"attn_q rows should be 5120, got {wq.ne[1]}"

  # FFN layer 1
  let ffnUp = m.infos["blk.1.ffn_up.weight"]
  doAssert ffnUp.ne[0] == 3136, &"ffn_up cols should be 3136, got {ffnUp.ne[0]}"
  doAssert ffnUp.ne[1] == 12544, &"ffn_up rows should be 12544, got {ffnUp.ne[1]}"

  echo "[OK] Nemotron-H weight shapes verified"

proc testModelLoad() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  echo "Loading Nemotron-H 4B..."
  let t0 = epochTime()
  var ctx = loadInferenceContext(ModelPath, MaxContextLen)
  let t1 = epochTime()
  defer: unloadInferenceContext(ctx)
  echo &"[OK] Nemotron-H model loaded in {(t1-t0):.1f}s"

proc testInference() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  echo "Loading Nemotron-H 4B for inference..."
  var ctx = loadInferenceContext(ModelPath, MaxContextLen)
  defer: unloadInferenceContext(ctx)

  let promptTokens = encodePromptTokens(ctx.vocab, "Say hello.")
  doAssert promptTokens.len > 0, "empty prompt tokens"
  echo &"Prompt tokens: {promptTokens.len}"

  ctx.cache.curLen = 0
  var logits = forwardPrefill(ctx.model, promptTokens, ctx.cache)
  let firstToken = argmaxLast(logits, ctx.model.hparams.nVocab)
  doAssert firstToken >= 0 and firstToken < int32(ctx.model.hparams.nVocab),
    &"first token out of range: {firstToken}"
  echo &"First token: {firstToken} = \"{ctx.vocab.tokenToPiece(firstToken)}\""

  var generated: seq[int32] = @[firstToken]
  var nextToken = firstToken
  for i in 1 ..< 16:
    if nextToken in ctx.vocab.stopTokenIds:
      break
    logits = forwardDecode(ctx.model, nextToken, ctx.cache)
    nextToken = argmaxLast(logits, ctx.model.hparams.nVocab)
    generated.add(nextToken)

  let text = ctx.vocab.detokenize(generated)
  doAssert text.len > 0, "generated empty text"
  echo &"Generated: \"{text}\" ({generated.len} tokens)"
  echo "[OK] Nemotron-H inference produces output"

proc testGenerateLoop() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  var ctx = loadInferenceContext(ModelPath, MaxContextLen)
  defer: unloadInferenceContext(ctx)

  let result = generate(ctx, "Hello", 16)
  doAssert result.text.len > 0, "generate returned empty text"
  doAssert result.promptTokens > 0, "zero prompt tokens"
  doAssert result.completionTokens > 0, "zero completion tokens"
  doAssert result.elapsedMs > 0.0, "zero elapsed time"
  let tokPerSec = float64(result.completionTokens) / (result.elapsedMs / 1000.0)
  echo &"Generated {result.completionTokens} tokens in {result.elapsedMs:.1f}ms ({tokPerSec:.1f} tok/s)"
  echo &"Text: \"{result.text}\""
  echo "[OK] Nemotron-H generate() loop works"

when isMainModule:
  testHParams()
  testLayerTypes()
  testTokenizer()
  testWeightShapes()
  testModelLoad()
  testInference()
  testGenerateLoop()
