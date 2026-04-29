## End-to-end server test: start hippo_leap, send HTTP requests, verify responses.
##
## Requires: pre-built ./hippo_leap binary (make build)
## Requires: TinyLlama Q2_K model file on disk.
## Runs with: nim r (plain C backend, no GPU needed for the test client)

import
  std/[os, osproc, strutils, json, times],
  curly,
  webby/httpheaders

const
  ModelPath = "/mnt/steel-chest/LLM/lmstudio/models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf"
  BinaryPath = "./hippo_leap"
  TestPort = 18932
  BaseUrl = "http://127.0.0.1:" & $TestPort

proc waitForHealth(pool: Curly, maxWaitSec: int): bool =
  let deadline = epochTime() + maxWaitSec.float64
  while epochTime() < deadline:
    try:
      let resp = pool.get(BaseUrl & "/health")
      if resp.code == 200:
        return true
    except CatchableError:
      discard
    sleep(500)
  false

proc testHealthEndpoint(pool: Curly) =
  let resp = pool.get(BaseUrl & "/health")
  doAssert resp.code == 200, "health returned " & $resp.code
  let j = parseJson(resp.body)
  doAssert j["status"].getStr() == "ok"
  echo "[OK] /health returns ok"

proc testListModels(pool: Curly) =
  let resp = pool.get(BaseUrl & "/v1/models")
  doAssert resp.code == 200, "/v1/models returned " & $resp.code
  let j = parseJson(resp.body)
  doAssert j["object"].getStr() == "list"
  doAssert j["data"].len > 0
  echo "[OK] /v1/models returns model list"

proc testChatCompletion(pool: Curly) =
  let reqBody = $ %*{
    "model": "tinylama",
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "Say hello."}]}
    ],
    "max_tokens": 8
  }
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"
  let resp = pool.post(BaseUrl & "/v1/chat/completions", headers, reqBody, timeout = 120)
  doAssert resp.code == 200, "chat completions returned " & $resp.code
  let j = parseJson(resp.body)
  doAssert j["object"].getStr() == "chat.completion"
  doAssert j["choices"].len > 0
  let content = j["choices"][0]["message"]["content"].getStr()
  doAssert content.len > 0, "empty response content"
  doAssert j["usage"]["prompt_tokens"].getInt() > 0
  doAssert j["usage"]["total_tokens"].getInt() > 0
  echo "[OK] /v1/chat/completions returns generated text: " & content

proc testStreamingRejected(pool: Curly) =
  let reqBody = $ %*{
    "model": "tinylama",
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
    ],
    "stream": true
  }
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"
  let resp = pool.post(BaseUrl & "/v1/chat/completions", headers, reqBody, timeout = 120)
  doAssert resp.code == 400, "streaming should return 400, got " & $resp.code
  echo "[OK] streaming requests rejected with 400"

proc testNotFound(pool: Curly) =
  let resp = pool.get(BaseUrl & "/nonexistent")
  doAssert resp.code == 404
  echo "[OK] unknown routes return 404"

proc main() =
  if not fileExists(BinaryPath):
    echo "[SKIP] Binary not found: " & BinaryPath
    echo "Run 'make build' first."
    return

  if not fileExists(ModelPath):
    echo "[SKIP] Model file not found: " & ModelPath
    return

  putEnv("HIPPO_LEAP_PORT", $TestPort)
  putEnv("HIPPO_LEAP_MODEL", ModelPath)
  putEnv("HIPPO_LEAP_MAX_CONTEXT", "128")

  echo "Starting hippo_leap server on port " & $TestPort & "..."
  let server = startProcess(
    BinaryPath, args = @["serve"],
    env = nil,
    options = {poStdErrToStdOut}
  )

  defer:
    server.terminate()
    try: server.kill() except CatchableError: discard
    server.close()
    delEnv("HIPPO_LEAP_PORT")
    delEnv("HIPPO_LEAP_MODEL")
    delEnv("HIPPO_LEAP_MAX_CONTEXT")

  let pool = newCurly()

  echo "Waiting for server to become healthy (up to 120s for model load)..."
  let healthy = waitForHealth(pool, 120)
  if not healthy:
    echo "[FAIL] Server did not become healthy within 120 seconds"
    quit(1)

  testHealthEndpoint(pool)
  testListModels(pool)

  if not server.running:
    echo "[FAIL] Server exited unexpectedly before chat test"
    echo "Exit code: " & $server.peekExitCode()
    quit(1)

  testChatCompletion(pool)

  if not server.running:
    echo "[FAIL] Server exited during chat completion test"
    echo "Exit code: " & $server.peekExitCode()
    quit(1)

  testStreamingRejected(pool)
  testNotFound(pool)

  echo "All e2e tests passed."

when isMainModule:
  main()
