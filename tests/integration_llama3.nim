## GPU integration test for Llama 3.2 1B Instruct (Q4_K_M).
##
## Requires: nim cpp --cc:hipcc -d:backendNaive -d:useMalloc
## Requires: Llama 3.2 1B Instruct Q4_K_M model file on disk.

import
  std/[os, strformat, strutils, tables],
  hippo_leap/[inference, tokenizer, model, sampling]

const
  ModelPath = "/mnt/steel-chest/LLM/lmstudio/models/lmstudio-community/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
  MaxContextLen = 256

proc testHParams() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found: " & ModelPath
    return

  var m = loadModel(ModelPath)
  defer: m.close()

  let hp = m.hparams
  doAssert hp.arch == "llama", "expected arch llama, got " & hp.arch
  doAssert hp.nLayer == 16, &"expected 16 layers, got {hp.nLayer}"
  doAssert hp.nEmb == 2048, &"expected 2048 emb, got {hp.nEmb}"
  doAssert hp.nHead == 32, &"expected 32 heads, got {hp.nHead}"
  doAssert hp.nHeadKv == 8, &"expected 8 kv heads, got {hp.nHeadKv}"
  doAssert hp.nVocab == 128256, &"expected 128256 vocab, got {hp.nVocab}"
  echo &"[OK] Llama 3.2 hparams: {hp.nLayer} layers, {hp.nEmb} emb, {hp.nVocab} vocab"

proc testTokenizer() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  var m = loadModel(ModelPath)
  defer: m.close()
  let vocab = loadVocab(m.gguf)

  doAssert vocab.modelType == "gpt2", "expected gpt2, got " & vocab.modelType
  doAssert vocab.tokens.len == 128256, &"expected 128256 tokens, got {vocab.tokens.len}"
  doAssert vocab.mergeRank.len > 0, "expected merges loaded"
  doAssert vocab.bosId == 128000, &"expected bosId 128000, got {vocab.bosId}"

  let tokens = vocab.tokenize("Hello world", addSpecial = false)
  doAssert tokens.len > 0, "tokenize returned empty"
  let decoded = vocab.detokenize(tokens)
  doAssert "Hello" in decoded, "detokenize missing Hello, got: " & decoded
  doAssert "world" in decoded, "detokenize missing world, got: " & decoded

  let formatted = vocab.formatChatPrompt("Hello")
  doAssert formatted.contains("<|start_header_id|>"), "missing start_header_id"
  doAssert formatted.contains("<|begin_of_text|>"), "missing begin_of_text"
  doAssert formatted.contains("<|eot_id|>"), "missing eot_id"
  echo "[OK] Llama 3.2 tokenizer + chat template"

proc testInference() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  echo "Loading Llama 3.2 1B..."
  var ctx = loadInferenceContext(ModelPath, MaxContextLen)
  defer: unloadInferenceContext(ctx)

  echo &"Model: {ctx.model.hparams.arch}, {ctx.model.hparams.nLayer} layers"

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
  echo "[OK] Llama 3.2 inference produces output"

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
  echo "[OK] Llama 3.2 generate() loop works"

when isMainModule:
  testHParams()
  testTokenizer()
  testInference()
  testGenerateLoop()
