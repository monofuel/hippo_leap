import
  std/[strformat],
  mummy, mummy/routers,
  ./[common, config, handlers, inference]

var ctx: InferenceContext

proc runServer*(cfg: ServerConfig) =
  ## Start the mummy HTTP server with OpenAI-compatible routes.
  echo &"hippo_leap {Version} starting on {cfg.address}:{cfg.port}"

  if cfg.modelPath.len > 0:
    echo &"Loading model: {cfg.modelPath}"
    ctx = loadInferenceContext(cfg.modelPath, cfg.maxContextLen)
    inferCtx = addr ctx
    echo &"Model loaded: {ctx.model.hparams.arch}, {ctx.model.hparams.nLayer} layers, {ctx.model.hparams.nVocab} vocab"
  else:
    echo "WARNING: No model path set. Inference will be unavailable."
    echo "Set HIPPO_LEAP_MODEL=/path/to/model.gguf to enable inference."

  var router: Router
  router.get("/health", healthHandler)
  router.get("/v1/models", listModelsHandler)
  router.post("/v1/chat/completions", chatCompletionsHandler)
  router.notFoundHandler = notFoundHandler

  let server = newServer(router)
  echo &"Serving on http://{cfg.address}:{cfg.port}"
  server.serve(Port(cfg.port), address = cfg.address)
