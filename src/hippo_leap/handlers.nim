import
  std/[options, times],
  jsony,
  mummy,
  openai_leap/common as oai_common,
  ./common

proc healthHandler*(request: Request) =
  ## Handle GET /health.
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"
  request.respond(200, headers, """{"status": "ok"}""")

proc listModelsHandler*(request: Request) =
  ## Handle GET /v1/models.
  let resp = ListModelResponse(
    data: @[OpenAiModel(
      id: DefaultModel,
      created: epochTime().int,
      `object`: "model",
      owned_by: "hippo_leap",
    )],
    `object`: "list",
  )
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"
  request.respond(200, headers, resp.toJson())

proc chatCompletionsHandler*(request: Request) =
  ## Handle POST /v1/chat/completions.
  let body = request.body
  let req = body.fromJson(CreateChatCompletionReq)
  let resp = CreateChatCompletionResp(
    id: "chatcmpl-stub",
    choices: @[CreateChatMessage(
      finish_reason: "stop",
      index: 0,
      message: some(RespMessage(
        content: "This is a stub response from hippo_leap. Inference is not yet implemented.",
        role: "assistant",
      )),
    )],
    created: epochTime().int,
    model: req.model,
    system_fingerprint: "hippo_leap_stub",
    `object`: "chat.completion",
    usage: Usage(
      prompt_tokens: 0,
      total_tokens: 0,
    ),
  )
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"
  request.respond(200, headers, resp.toJson())

proc notFoundHandler*(request: Request) =
  ## Return a 404 JSON response.
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"
  request.respond(404, headers, """{"error": "not found"}""")
