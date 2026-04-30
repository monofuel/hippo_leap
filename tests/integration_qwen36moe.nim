## GPU integration test for Qwen3.6 35B A3B (Triple hybrid: GDN + MoE + Attention).
##
## Requires: nim cpp --cc:hipcc -d:backendNaive -d:useMalloc -d:zippyNoSimd
## Requires: Qwen3.6 35B A3B model file on disk.

import
  std/[os, strformat, strutils, times, tables],
  hippo_leap/[inference, tokenizer, model, sampling]

const
  ModelPath = "/mnt/steel-chest/LLM/lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"
  MaxContextLen = 256

proc testHParams() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found: " & ModelPath
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  let hp = m.hparams
  doAssert hp.arch == "qwen35moe", "expected arch qwen35moe, got " & hp.arch
  doAssert hp.nLayer == 40, &"expected 40 layers, got {hp.nLayer}"
  doAssert hp.nEmb == 2048, &"expected 2048 emb, got {hp.nEmb}"
  doAssert hp.nHead == 16, &"expected 16 heads, got {hp.nHead}"
  doAssert hp.nHeadKv == 2, &"expected 2 kv heads, got {hp.nHeadKv}"
  doAssert hp.headDim == 256, &"expected 256 headDim, got {hp.headDim}"
  doAssert hp.nVocab == 248320, &"expected 248320 vocab, got {hp.nVocab}"

  doAssert hp.nExperts == 256, &"expected 256 experts, got {hp.nExperts}"
  doAssert hp.nExpertsUsed == 8, &"expected 8 experts used, got {hp.nExpertsUsed}"
  doAssert hp.expertFfnDim == 512, &"expected 512 expertFfnDim, got {hp.expertFfnDim}"
  doAssert hp.fullAttnInterval == 4, &"expected fullAttnInterval=4, got {hp.fullAttnInterval}"

  doAssert hp.ssmConvKernel == 4, &"expected ssmConvKernel=4, got {hp.ssmConvKernel}"
  doAssert hp.ssmStateSize == 128, &"expected ssmStateSize=128, got {hp.ssmStateSize}"
  doAssert hp.ssmInnerSize == 4096, &"expected ssmInnerSize=4096, got {hp.ssmInnerSize}"
  doAssert hp.ssmDtRank == 32, &"expected ssmDtRank=32, got {hp.ssmDtRank}"
  doAssert hp.ssmGroupCount == 16, &"expected ssmGroupCount=16 (nKHeads), got {hp.ssmGroupCount}"

  doAssert hp.ropeDimSections.len == 4, &"expected 4 rope sections, got {hp.ropeDimSections.len}"
  doAssert hp.ropeDimSections == @[11'i32, 11, 10, 0], &"unexpected ropeDimSections: {hp.ropeDimSections}"

  echo &"[OK] Qwen3.6 hparams: {hp.nLayer} layers, {hp.nEmb} emb, headDim={hp.headDim}, {hp.nVocab} vocab"
  echo &"     MoE: {hp.nExperts} experts, {hp.nExpertsUsed} active, ffnDim={hp.expertFfnDim}"
  echo &"     SSM: convK={hp.ssmConvKernel} state={hp.ssmStateSize} inner={hp.ssmInnerSize} dtRank={hp.ssmDtRank} nKHeads={hp.ssmGroupCount}"

proc testLayerTypes() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  var nSsmAttnMoe, nAttnMoe = 0
  for layer in 0 ..< m.hparams.nLayer:
    if m.hparams.fullAttnInterval > 0 and
       (layer mod m.hparams.fullAttnInterval) == (m.hparams.fullAttnInterval - 1):
      inc nAttnMoe
    else:
      inc nSsmAttnMoe

  doAssert nSsmAttnMoe == 30, &"expected 30 SSM+MoE layers, got {nSsmAttnMoe}"
  doAssert nAttnMoe == 10, &"expected 10 Attn+MoE layers, got {nAttnMoe}"
  echo &"[OK] Layer types: {nSsmAttnMoe} GDN+MoE, {nAttnMoe} Attention+MoE"

proc testWeightShapes() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  # Layer 0: GDN + MoE (lkSsmAttnMoe)
  let qkv = m.infos["blk.0.attn_qkv.weight"]
  doAssert qkv.ne[0] == 2048, &"attn_qkv cols should be 2048, got {qkv.ne[0]}"
  doAssert qkv.ne[1] == 8192, &"attn_qkv rows should be 8192, got {qkv.ne[1]}"

  let gate = m.infos["blk.0.attn_gate.weight"]
  doAssert gate.ne[0] == 2048, &"attn_gate cols should be 2048, got {gate.ne[0]}"
  doAssert gate.ne[1] == 4096, &"attn_gate rows should be 4096, got {gate.ne[1]}"

  let conv = m.infos["blk.0.ssm_conv1d.weight"]
  doAssert conv.ne[0] == 4, &"conv1d kernel should be 4, got {conv.ne[0]}"
  doAssert conv.ne[1] == 8192, &"conv1d channels should be 8192, got {conv.ne[1]}"

  let ssmOut = m.infos["blk.0.ssm_out.weight"]
  doAssert ssmOut.ne[0] == 4096, &"ssm_out cols should be 4096, got {ssmOut.ne[0]}"
  doAssert ssmOut.ne[1] == 2048, &"ssm_out rows should be 2048, got {ssmOut.ne[1]}"

  # Layer 3: Full attention + MoE (lkAttnMoe)
  let wq = m.infos["blk.3.attn_q.weight"]
  doAssert wq.ne[0] == 2048, &"attn_q cols should be 2048, got {wq.ne[0]}"
  doAssert wq.ne[1] == 8192, &"attn_q rows should be 8192 (Q+gate), got {wq.ne[1]}"

  let wk = m.infos["blk.3.attn_k.weight"]
  doAssert wk.ne[0] == 2048, &"attn_k cols should be 2048, got {wk.ne[0]}"
  doAssert wk.ne[1] == 512, &"attn_k rows should be 512, got {wk.ne[1]}"

  # MoE expert tensors (3D)
  let gateExps = m.infos["blk.0.ffn_gate_exps.weight"]
  doAssert gateExps.nDims == 3, &"ffn_gate_exps should be 3D, got {gateExps.nDims}D"
  doAssert gateExps.ne[2] == 256, &"ffn_gate_exps ne[2] should be 256, got {gateExps.ne[2]}"

  echo "[OK] Qwen3.6 weight shapes verified"

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
  echo "[OK] Qwen3.6 tokenizer + ChatML template"

proc testModelLoad() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  echo "Loading Qwen3.6 35B A3B..."
  let t0 = epochTime()
  var ctx = loadInferenceContext(ModelPath, MaxContextLen)
  let t1 = epochTime()
  defer: unloadInferenceContext(ctx)
  echo &"[OK] Qwen3.6 model loaded in {(t1-t0):.1f}s"

proc testInference() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  echo "Loading Qwen3.6 35B A3B for inference..."
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
  echo "[OK] Qwen3.6 inference produces output"

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
  echo "[OK] Qwen3.6 generate() loop works"

when isMainModule:
  testHParams()
  testLayerTypes()
  testWeightShapes()
  testTokenizer()
  testModelLoad()
  testInference()
  testGenerateLoop()
