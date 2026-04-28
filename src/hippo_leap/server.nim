import
  std/[strformat],
  mummy, mummy/routers,
  ./[common, config, handlers]

proc runServer*(cfg: ServerConfig) =
  ## Start the mummy HTTP server with OpenAI-compatible routes.
  echo &"hippo_leap {Version} starting on {cfg.address}:{cfg.port}"

  var router: Router
  router.get("/health", healthHandler)
  router.get("/v1/models", listModelsHandler)
  router.post("/v1/chat/completions", chatCompletionsHandler)
  router.notFoundHandler = notFoundHandler

  let server = newServer(router)
  echo &"Serving on http://{cfg.address}:{cfg.port}"
  server.serve(Port(cfg.port), address = cfg.address)
