## GPU integration test for Qwen3 4B Instruct (F16).
##
## Requires: nim cpp --cc:hipcc -d:backendNaive -d:useMalloc -d:zippyNoSimd
## Requires: Qwen3 4B Instruct model file on disk.

import
  std/[os, strformat, strutils, times, tables],
  hippo_leap/[inference, tokenizer, model, sampling]

const
  ModelPath = "/mnt/steel-chest/LLM/lmstudio/models/monofuel/qwen3-4b-instruct-2507.gguf"
  MaxContextLen = 256

proc testHParams() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found: " & ModelPath
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  let hp = m.hparams
  doAssert hp.arch == "qwen3", "expected arch qwen3, got " & hp.arch
  doAssert hp.nLayer == 36, &"expected 36 layers, got {hp.nLayer}"
  doAssert hp.nEmb == 2560, &"expected 2560 emb, got {hp.nEmb}"
  doAssert hp.nHead == 32, &"expected 32 heads, got {hp.nHead}"
  doAssert hp.nHeadKv == 8, &"expected 8 kv heads, got {hp.nHeadKv}"
  doAssert hp.headDim == 128, &"expected 128 headDim, got {hp.headDim}"
  doAssert hp.nFfn == 9728, &"expected 9728 ffn, got {hp.nFfn}"
  doAssert hp.nVocab == 151936, &"expected 151936 vocab, got {hp.nVocab}"
  doAssert hp.ropeDim == 128, &"expected 128 ropeDim, got {hp.ropeDim}"
  let qDim = hp.nHead * hp.headDim
  doAssert qDim == 4096, &"expected qDim 4096, got {qDim}"
  doAssert qDim != hp.nEmb, "qDim should differ from nEmb for qwen3"
  echo &"[OK] Qwen3 hparams: {hp.nLayer} layers, {hp.nEmb} emb, headDim={hp.headDim}, qDim={qDim}, {hp.nVocab} vocab"

proc testTokenizer() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  var m = loadModel(ModelPath)
  defer: m.close()
  let vocab = loadVocab(m.gguf)

  doAssert vocab.tokens.len == 151936, &"expected 151936 tokens, got {vocab.tokens.len}"

  doAssert vocab.tokenToId.hasKey("<|im_start|>"), "missing <|im_start|> token"
  doAssert vocab.tokenToId.hasKey("<|im_end|>"), "missing <|im_end|> token"

  let imEndId = int32(vocab.tokenToId["<|im_end|>"])
  doAssert imEndId in vocab.stopTokenIds, "<|im_end|> should be a stop token"

  let tokens = vocab.tokenize("Hello world", addSpecial = false)
  doAssert tokens.len > 0, "tokenize returned empty"
  let decoded = vocab.detokenize(tokens)
  doAssert "Hello" in decoded, "detokenize missing Hello, got: " & decoded
  doAssert "world" in decoded, "detokenize missing world, got: " & decoded

  let formatted = vocab.formatChatPrompt("Hello")
  doAssert formatted.contains("<|im_start|>"), "missing <|im_start|> in chat template"
  doAssert formatted.contains("<|im_end|>"), "missing <|im_end|> in chat template"
  doAssert formatted.contains("user"), "missing user role in chat template"
  doAssert formatted.contains("assistant"), "missing assistant role in chat template"
  echo "[OK] Qwen3 tokenizer + ChatML template"

proc testQkNormWeights() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  doAssert m.infos.hasKey("blk.0.attn_q_norm.weight"), "missing attn_q_norm.weight"
  doAssert m.infos.hasKey("blk.0.attn_k_norm.weight"), "missing attn_k_norm.weight"

  let qNormInfo = m.infos["blk.0.attn_q_norm.weight"]
  let kNormInfo = m.infos["blk.0.attn_k_norm.weight"]
  doAssert qNormInfo.ne[0] == 128, &"expected q_norm dim 128, got {qNormInfo.ne[0]}"
  doAssert kNormInfo.ne[0] == 128, &"expected k_norm dim 128, got {kNormInfo.ne[0]}"

  let qNorm = m.getTensor("blk.0.attn_q_norm.weight")
  doAssert qNorm.data.len == 128, &"expected 128 elements, got {qNorm.data.len}"
  echo "[OK] Qwen3 QK norm weights present and loadable"

proc testWeightLoading() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  let t0 = epochTime()
  let wq = m.getTensor("blk.0.attn_q.weight")
  let t1 = epochTime()
  let qDim = m.hparams.nHead * m.hparams.headDim
  # GGUF stores [cols, rows] — shape[0]=inDim, shape[1]=outDim
  doAssert wq.shape[0] == m.hparams.nEmb, &"expected Q cols={m.hparams.nEmb}, got {wq.shape[0]}"
  doAssert wq.shape[1] == qDim, &"expected Q rows={qDim}, got {wq.shape[1]}"
  echo &"[OK] Qwen3 blk.0.attn_q.weight loaded: {wq.shape} in {(t1-t0)*1000:.0f}ms"

  let wo = m.getTensor("blk.0.attn_output.weight")
  doAssert wo.shape[0] == qDim, &"expected O cols={qDim}, got {wo.shape[0]}"
  doAssert wo.shape[1] == m.hparams.nEmb, &"expected O rows={m.hparams.nEmb}, got {wo.shape[1]}"
  echo &"[OK] Qwen3 blk.0.attn_output.weight: {wo.shape}"

proc testModelLoad() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  echo "Loading Qwen3 4B (F16, this may take a minute)..."
  let t0 = epochTime()
  var ctx = loadInferenceContext(ModelPath, MaxContextLen)
  let t1 = epochTime()
  defer: unloadInferenceContext(ctx)
  echo &"[OK] Qwen3 model loaded in {(t1-t0):.1f}s"

proc testInference() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  echo "Loading Qwen3 4B for inference..."
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
  echo "[OK] Qwen3 inference produces output"

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
  echo "[OK] Qwen3 generate() loop works"

when isMainModule:
  testHParams()
  testTokenizer()
  testQkNormWeights()
  testWeightLoading()
  testModelLoad()
  testInference()
  testGenerateLoop()
