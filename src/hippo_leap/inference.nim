## Compile-time backend dispatch and generation loop.

import
  std/[locks, times],
  ./[model, tokenizer, sampling, backend_types]

when defined(backendNaive):
  import ./backends/naive
  export naive
elif defined(backendMegakernel):
  import ./backends/megakernel
  export megakernel
else:
  {.error: "No backend selected. Use -d:backendNaive".}

type
  GenerateResult* = object
    text*: string
    promptTokens*: int
    completionTokens*: int
    totalTokens*: int
    elapsedMs*: float64

  InferenceContext* = object
    model*: Model
    vocab*: Vocab
    cache*: KvCache
    maxContextLen*: int
    lock*: Lock
    loaded*: bool

proc loadInferenceContext*(modelPath: string, maxContextLen: int): InferenceContext =
  ## Load a GGUF model, parse vocab, init KV cache, and upload weights to GPU.
  result.model = loadModel(modelPath)
  result.vocab = loadVocab(result.model.gguf)
  result.cache = initKvCache(result.model.hparams, maxContextLen)
  result.maxContextLen = maxContextLen
  initLock(result.lock)
  loadModelBackend(result.model, result.model.hparams)
  result.loaded = true

proc unloadInferenceContext*(ctx: var InferenceContext) =
  ## Release GPU resources and close the model file.
  if ctx.loaded:
    unloadModelBackend()
    ctx.model.close()
    deinitLock(ctx.lock)
    ctx.loaded = false

proc generate*(ctx: var InferenceContext, text: string, maxTokens: int): GenerateResult =
  ## Run prefill + decode loop and return the generated text.
  acquire(ctx.lock)
  defer: release(ctx.lock)

  let t0 = epochTime()

  ctx.cache.curLen = 0

  let promptTokens = encodePromptTokens(ctx.vocab, text)
  if promptTokens.len == 0:
    return GenerateResult(text: "", promptTokens: 0, completionTokens: 0,
                          totalTokens: 0, elapsedMs: 0.0)

  let hp = ctx.model.hparams
  let maxGen = min(maxTokens, ctx.maxContextLen - promptTokens.len)
  if maxGen <= 0:
    return GenerateResult(text: "", promptTokens: promptTokens.len,
                          completionTokens: 0, totalTokens: promptTokens.len,
                          elapsedMs: 0.0)

  var logits = forwardPrefill(ctx.model, promptTokens, ctx.cache)
  var nextToken = argmaxLast(logits, hp.nVocab)

  var generated: seq[int32]
  generated.add(nextToken)

  for i in 1 ..< maxGen:
    if nextToken in ctx.vocab.stopTokenIds:
      break
    logits = forwardDecode(ctx.model, nextToken, ctx.cache)
    nextToken = argmaxLast(logits, hp.nVocab)
    generated.add(nextToken)

  let t1 = epochTime()
  let outText = ctx.vocab.detokenize(generated)

  result = GenerateResult(
    text: outText,
    promptTokens: promptTokens.len,
    completionTokens: generated.len,
    totalTokens: promptTokens.len + generated.len,
    elapsedMs: (t1 - t0) * 1000.0,
  )
