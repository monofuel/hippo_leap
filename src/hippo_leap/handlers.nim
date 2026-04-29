import
  std/[options, times, strutils],
  jsony,
  mummy,
  openai_leap/common as oai_common,
  ./[common, inference]

var inferCtx*: ptr InferenceContext

proc healthHandler*(request: Request) =
  ## Handle GET /health.
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"
  request.respond(200, headers, """{"status": "ok"}""")

proc listModelsHandler*(request: Request) {.gcsafe.} =
  ## Handle GET /v1/models.
  {.cast(gcsafe).}:
    let modelName = if inferCtx != nil and inferCtx[].loaded:
      inferCtx[].model.hparams.arch & " (" & $inferCtx[].model.hparams.nVocab & " vocab)"
    else:
      DefaultModel
  let resp = ListModelResponse(
    data: @[OpenAiModel(
      id: modelName,
      created: epochTime().int,
      `object`: "model",
      owned_by: "hippo_leap",
    )],
    `object`: "list",
  )
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"
  request.respond(200, headers, resp.toJson())

proc extractText(messages: seq[oai_common.Message]): string =
  ## Extract text content from the last user message.
  for i in countdown(messages.high, 0):
    if messages[i].role == "user" and messages[i].content.isSome:
      for part in messages[i].content.get:
        if part.`type` == "text" and part.text.isSome:
          return part.text.get
  ""

proc chatCompletionsHandler*(request: Request) {.gcsafe.} =
  ## Handle POST /v1/chat/completions.
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"

  let body = request.body
  let req = body.fromJson(CreateChatCompletionReq)

  if req.stream.isSome and req.stream.get:
    request.respond(400, headers,
      """{"error": {"message": "streaming not yet supported", "type": "invalid_request_error"}}""")
    return

  {.cast(gcsafe).}:
    if inferCtx == nil or not inferCtx[].loaded:
      request.respond(503, headers,
        """{"error": {"message": "model not loaded", "type": "server_error"}}""")
      return

    let text = extractText(req.messages)
    if text.len == 0:
      request.respond(400, headers,
        """{"error": {"message": "no user message text found", "type": "invalid_request_error"}}""")
      return

    let maxTok = if req.max_tokens.isSome: req.max_tokens.get else: 256

    var genResult: GenerateResult
    try:
      genResult = generate(inferCtx[], text, maxTok)
    except CatchableError as e:
      request.respond(500, headers,
        """{"error": {"message": """ & e.msg.toJson() & """, "type": "server_error"}}""")
      return

  let resp = CreateChatCompletionResp(
    id: "chatcmpl-" & $epochTime().int,
    choices: @[CreateChatMessage(
      finish_reason: "stop",
      index: 0,
      message: some(RespMessage(
        content: genResult.text,
        role: "assistant",
      )),
    )],
    created: epochTime().int,
    model: req.model,
    system_fingerprint: "hippo_leap",
    `object`: "chat.completion",
    usage: Usage(
      prompt_tokens: genResult.promptTokens,
      total_tokens: genResult.totalTokens,
    ),
  )
  request.respond(200, headers, resp.toJson())

proc notFoundHandler*(request: Request) =
  ## Return a 404 JSON response.
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"
  request.respond(404, headers, """{"error": "not found"}""")
