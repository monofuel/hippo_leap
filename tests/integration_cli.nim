## Integration test for CLI commands: tokenize, bench, chat.
##
## Requires: compiled hippo_leap binary with GPU backend.
## Requires: TinyLlama Q2_K model file on disk.

import
  std/[os, osproc, strutils, strformat, streams]

const
  ModelPath = "/mnt/steel-chest/LLM/lmstudio/models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf"
  Binary = "./hippo_leap"

proc run(args: string, input = ""): tuple[output: string, exitCode: int] =
  let cmd = Binary & " " & args
  if input.len > 0:
    let p = startProcess(Binary, args = args.splitWhitespace(), options = {poUsePath, poStdErrToStdOut})
    p.inputStream.write(input)
    p.inputStream.close()
    result.output = p.outputStream.readAll()
    result.exitCode = p.waitForExit()
    p.close()
  else:
    result = execCmdEx(cmd)

proc testHelp() =
  let (output, code) = run("--help")
  doAssert code == 0, &"--help exited with {code}"
  doAssert output.contains("tokenize"), "--help missing tokenize"
  doAssert output.contains("bench"), "--help missing bench"
  doAssert output.contains("chat"), "--help missing chat"
  echo "[OK] --help lists all subcommands"

proc testTokenize() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  let (output, code) = run(&"tokenize --model {ModelPath} \"Hello world\"")
  doAssert code == 0, &"tokenize exited with {code}"
  doAssert output.contains("Token count:"), "missing token count"
  doAssert output.contains("15043"), "missing Hello token ID"
  doAssert output.contains("3186"), "missing world token ID"
  echo "[OK] tokenize basic"

proc testTokenizeSpecial() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  let (output, code) = run(&"tokenize --model {ModelPath} --special \"Hello world\"")
  doAssert code == 0, &"tokenize --special exited with {code}"
  doAssert output.contains("(BOS)"), "missing BOS label"
  doAssert output.contains("[1]"), "missing BOS token ID"
  echo "[OK] tokenize --special"

proc testTokenizeIdsOnly() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  let (output, code) = run(&"tokenize --model {ModelPath} --ids-only \"Hello world\"")
  doAssert code == 0, &"tokenize --ids-only exited with {code}"
  var ids: seq[string]
  for line in output.strip().splitLines():
    let s = line.strip()
    if s.len > 0 and s[0].isDigit():
      ids.add(s)
  doAssert ids.len == 2, &"expected 2 token IDs, got {ids.len}: {ids}"
  doAssert ids[0] == "15043", &"expected 15043, got {ids[0]}"
  doAssert ids[1] == "3186", &"expected 3186, got {ids[1]}"
  echo "[OK] tokenize --ids-only"

proc testTokenizeStdin() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  let (output, code) = run(&"tokenize --model {ModelPath} --ids-only", input = "Hello world\n")
  doAssert code == 0, &"tokenize stdin exited with {code}"
  doAssert output.contains("15043"), "missing Hello token from stdin"
  doAssert output.contains("3186"), "missing world token from stdin"
  echo "[OK] tokenize stdin"

proc testBench() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  let (output, code) = run(&"bench --model {ModelPath} --runs 1 --warmup 0 --tokens 8")
  doAssert code == 0, &"bench exited with {code}"
  doAssert output.contains("tok/s"), "missing tok/s in bench output"
  doAssert output.contains("Mean:"), "missing Mean in bench output"
  echo "[OK] bench"

proc testChat() =
  if not fileExists(ModelPath):
    echo "[SKIP] Model not found"
    return

  let (output, code) = run(&"chat --model {ModelPath} --tokens 8", input = "Say hello.\nquit\n")
  doAssert code == 0, &"chat exited with {code}"
  doAssert output.contains("prompt"), "missing prompt stats"
  doAssert output.contains("tok/s"), "missing tok/s in chat output"
  echo "[OK] chat"

when isMainModule:
  if not fileExists(Binary):
    echo "Error: hippo_leap binary not found. Run 'make build' first."
    quit(1)
  testHelp()
  testTokenize()
  testTokenizeSpecial()
  testTokenizeIdsOnly()
  testTokenizeStdin()
  testBench()
  testChat()
