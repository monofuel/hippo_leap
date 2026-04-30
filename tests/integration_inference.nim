## GPU integration test: load model, run prefill + decode, verify golden output.
##
## Requires: nim cpp --cc:hipcc -d:backendNaive -d:useMalloc
## Requires: TinyLlama Q2_K model file on disk.

import
  std/[os, strformat],
  hippo_leap/[inference, tokenizer, sampling]

const
  ModelPath = "/mnt/steel-chest/LLM/lmstudio/models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf"
  ShortPrompt = "Write one sentence about Nim."
  MaxContextLen = 128

  ExpectedFirstToken: int32 = 13
  ExpectedTokens: array[8, int32] = [
    13'i32, 13, 13, 13, 6028, 366, 3113, 788
  ]

proc testGoldenOutput() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model file not found: " & ModelPath
    return

  echo "Loading model..."
  var ctx = loadInferenceContext(ModelPath, MaxContextLen)
  defer: unloadInferenceContext(ctx)

  echo &"Model: {ctx.model.hparams.arch}, {ctx.model.hparams.nLayer} layers, {ctx.model.hparams.nVocab} vocab"

  let promptTokens = encodePromptTokens(ctx.vocab, ShortPrompt)
  echo &"Prompt tokens: {promptTokens.len}"

  ctx.cache.curLen = 0
  var logits = forwardPrefill(ctx.model, promptTokens, ctx.cache)
  let firstToken = argmaxLast(logits, ctx.model.hparams.nVocab)

  doAssert firstToken == ExpectedFirstToken,
    &"first token: got {firstToken}, expected {ExpectedFirstToken}"

  var generated: seq[int32]
  var nextToken = firstToken
  for i in 0 ..< 8:
    logits = forwardDecode(ctx.model, nextToken, ctx.cache)
    nextToken = argmaxLast(logits, ctx.model.hparams.nVocab)
    generated.add(nextToken)

  for i in 0 ..< 8:
    doAssert generated[i] == ExpectedTokens[i],
      &"token {i}: got {generated[i]}, expected {ExpectedTokens[i]}"

  let text = ctx.vocab.detokenize(@[firstToken] & generated)
  echo &"Generated: {text}"
  echo "[OK] Golden output matches"

proc testGenerateLoop() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model file not found: " & ModelPath
    return

  var ctx = loadInferenceContext(ModelPath, 256)
  defer: unloadInferenceContext(ctx)

  let result = generate(ctx, "Hello", 16)
  doAssert result.text.len > 0, "generate returned empty text"
  doAssert result.promptTokens > 0, "zero prompt tokens"
  doAssert result.completionTokens > 0, "zero completion tokens"
  doAssert result.elapsedMs > 0.0, "zero elapsed time"
  echo &"Generated {result.completionTokens} tokens in {result.elapsedMs:.1f}ms"
  echo &"Text: {result.text}"
  echo "[OK] generate() loop works"

proc testBenchmark() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model file not found: " & ModelPath
    return

  var ctx = loadInferenceContext(ModelPath, 512)
  defer: unloadInferenceContext(ctx)

  # Warmup (gets GPU clocks up)
  discard generate(ctx, "Hello", 8)

  # Reset cache and run timed benchmark
  ctx.cache.curLen = 0
  let result = generate(ctx, "The quick brown fox", 64)
  echo &"Benchmark: {result.completionTokens} tokens in {result.elapsedMs:.1f}ms = {float64(result.completionTokens) / result.elapsedMs * 1000:.1f} tok/s"

when isMainModule:
  testGoldenOutput()
  testGenerateLoop()
  testBenchmark()
