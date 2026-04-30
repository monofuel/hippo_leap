## GPU integration test for Nemotron 3 30B A3B (Mamba2 + MoE + Attention hybrid).
##
## Requires: nim cpp --cc:hipcc -d:backendNaive -d:useMalloc -d:zippyNoSimd
## Requires: Nemotron 3 30B A3B model file on disk.

import
  std/[os, strformat, strutils, times, tables],
  hippo_leap/[inference, tokenizer, model, sampling]

const
  ModelPath = "/mnt/steel-chest/LLM/lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF/NVIDIA-Nemotron-3-Nano-30B-A3B-Q3_K_L.gguf"
  MaxContextLen = 256

proc testHParams() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found: " & ModelPath
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  let hp = m.hparams
  doAssert hp.arch == "nemotron_h_moe", "expected arch nemotron_h_moe, got " & hp.arch
  doAssert hp.nLayer == 52, &"expected 52 layers, got {hp.nLayer}"
  doAssert hp.nEmb == 2688, &"expected 2688 emb, got {hp.nEmb}"
  doAssert hp.nHead == 32, &"expected 32 heads, got {hp.nHead}"
  doAssert hp.headDim == 128, &"expected 128 headDim, got {hp.headDim}"
  doAssert hp.nVocab == 131072, &"expected 131072 vocab, got {hp.nVocab}"
  doAssert hp.ropeDim == 84, &"expected 84 ropeDim, got {hp.ropeDim}"

  doAssert hp.ssmConvKernel == 4, &"expected ssmConvKernel=4, got {hp.ssmConvKernel}"
  doAssert hp.ssmStateSize == 128, &"expected ssmStateSize=128, got {hp.ssmStateSize}"
  doAssert hp.ssmGroupCount == 8, &"expected ssmGroupCount=8, got {hp.ssmGroupCount}"
  doAssert hp.ssmInnerSize == 4096, &"expected ssmInnerSize=4096, got {hp.ssmInnerSize}"
  doAssert hp.ssmDtRank == 64, &"expected ssmDtRank=64, got {hp.ssmDtRank}"

  doAssert hp.nExperts == 128, &"expected 128 experts, got {hp.nExperts}"
  doAssert hp.nExpertsUsed == 6, &"expected 6 experts used, got {hp.nExpertsUsed}"
  doAssert hp.expertFfnDim == 1856, &"expected expertFfnDim=1856, got {hp.expertFfnDim}"
  doAssert hp.sharedExpertFfnDim == 3712, &"expected sharedExpertFfnDim=3712, got {hp.sharedExpertFfnDim}"
  doAssert hp.expertWeightsScale == 2.5'f32, &"expected expertWeightsScale=2.5, got {hp.expertWeightsScale}"

  doAssert hp.layerNFfn.len == 52, &"expected 52 layerNFfn, got {hp.layerNFfn.len}"
  doAssert hp.layerNHeadKv.len == 52, &"expected 52 layerNHeadKv, got {hp.layerNHeadKv.len}"

  echo &"[OK] Nemotron-3 30B hparams: {hp.nLayer} layers, {hp.nEmb} emb, headDim={hp.headDim}, {hp.nVocab} vocab"
  echo &"     SSM: convK={hp.ssmConvKernel} state={hp.ssmStateSize} groups={hp.ssmGroupCount} inner={hp.ssmInnerSize} dtRank={hp.ssmDtRank}"
  echo &"     MoE: {hp.nExperts} experts, top-{hp.nExpertsUsed}, expertFfn={hp.expertFfnDim}, sharedFfn={hp.sharedExpertFfnDim}, scale={hp.expertWeightsScale}"

proc testLayerTypes() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  var nSsm, nAttn, nMoeFfn = 0
  for layer in 0 ..< m.hparams.nLayer:
    let lp = "blk." & $layer & "."
    if m.infos.hasKey(lp & "ssm_in.weight"):
      inc nSsm
    elif m.infos.hasKey(lp & "ffn_gate_inp.weight") and
         not m.infos.hasKey(lp & "attn_q.weight"):
      inc nMoeFfn
    elif m.hparams.layerNHeadKv[layer] > 0:
      inc nAttn

  doAssert nSsm == 23, &"expected 23 SSM layers, got {nSsm}"
  doAssert nAttn == 6, &"expected 6 attention layers, got {nAttn}"
  doAssert nMoeFfn == 23, &"expected 23 MoE FFN layers, got {nMoeFfn}"
  echo &"[OK] Layer types: {nSsm} SSM, {nAttn} attention, {nMoeFfn} MoE FFN"

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
  echo "[OK] Nemotron-3 30B tokenizer + ChatML template"

proc testWeightShapes() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  # SSM layer 0
  let ssmIn = m.infos["blk.0.ssm_in.weight"]
  doAssert ssmIn.ne[0] == 2688, &"ssm_in cols should be 2688, got {ssmIn.ne[0]}"

  let ssmConv = m.infos["blk.0.ssm_conv1d.weight"]
  doAssert ssmConv.ne[0] == 4, &"conv1d kernel should be 4, got {ssmConv.ne[0]}"

  # MoE FFN layer 1
  doAssert m.infos.hasKey("blk.1.ffn_gate_inp.weight"), "missing router weight"
  doAssert m.infos.hasKey("blk.1.ffn_up_exps.weight"), "missing expert up weight"
  doAssert m.infos.hasKey("blk.1.ffn_down_exps.weight"), "missing expert down weight"
  doAssert m.infos.hasKey("blk.1.ffn_up_shexp.weight"), "missing shared expert up weight"
  doAssert m.infos.hasKey("blk.1.ffn_down_shexp.weight"), "missing shared expert down weight"
  doAssert m.infos.hasKey("blk.1.exp_probs_b.bias"), "missing router bias"

  let router = m.infos["blk.1.ffn_gate_inp.weight"]
  doAssert router.ne[0] == 2688, &"router cols should be 2688, got {router.ne[0]}"
  doAssert router.ne[1] == 128, &"router rows should be 128, got {router.ne[1]}"

  let upExps = m.infos["blk.1.ffn_up_exps.weight"]
  doAssert upExps.ne[0] == 2688, &"up_exps cols should be 2688, got {upExps.ne[0]}"
  doAssert upExps.ne[1] == 1856, &"up_exps rows should be 1856, got {upExps.ne[1]}"

  let shUp = m.infos["blk.1.ffn_up_shexp.weight"]
  doAssert shUp.ne[0] == 2688, &"shared up cols should be 2688, got {shUp.ne[0]}"
  doAssert shUp.ne[1] == 3712, &"shared up rows should be 3712, got {shUp.ne[1]}"

  echo "[OK] Nemotron-3 30B weight shapes verified"

proc testModelLoad() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  echo "Loading Nemotron-3 30B A3B..."
  let t0 = epochTime()
  var ctx = loadInferenceContext(ModelPath, MaxContextLen)
  let t1 = epochTime()
  defer: unloadInferenceContext(ctx)
  echo &"[OK] Nemotron-3 30B model loaded in {(t1-t0):.1f}s"

proc testInference() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  echo "Loading Nemotron-3 30B A3B for inference..."
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
  echo "[OK] Nemotron-3 30B inference produces output"

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
  echo "[OK] Nemotron-3 30B generate() loop works"

when isMainModule:
  testHParams()
  testLayerTypes()
  testTokenizer()
  testWeightShapes()
  testModelLoad()
  testInference()
  testGenerateLoop()
