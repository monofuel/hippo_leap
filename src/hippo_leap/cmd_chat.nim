## Interactive chat REPL CLI command.

import
  std/[os, strutils, strformat],
  ./inference

const
  ChatUsage = """Usage: hippo_leap chat [options]

Options:
  --model, -m <path>    GGUF model file (or HIPPO_LEAP_MODEL env)
  --tokens, -n <int>    Max tokens per response (default: 256)
  --context <int>       Max context length (default: 2048)"""

proc cmdChat*(args: seq[string]) =
  var modelPath = getEnv("HIPPO_LEAP_MODEL", "")
  var maxTokens = 256
  var maxContext = 2048

  var i = 0
  while i < args.len:
    case args[i]
    of "--model", "-m":
      inc i
      if i >= args.len:
        echo "Error: --model requires a path"
        quit(1)
      modelPath = args[i]
    of "--tokens", "-n":
      inc i
      if i >= args.len:
        echo "Error: --tokens requires a number"
        quit(1)
      maxTokens = parseInt(args[i])
    of "--context":
      inc i
      if i >= args.len:
        echo "Error: --context requires a number"
        quit(1)
      maxContext = parseInt(args[i])
    of "--help", "-h":
      echo ChatUsage
      quit(0)
    else:
      echo &"Unknown option: {args[i]}"
      echo ChatUsage
      quit(1)
    inc i

  if modelPath.len == 0:
    echo "Error: no model specified. Use --model or set HIPPO_LEAP_MODEL"
    quit(1)
  if not fileExists(modelPath):
    echo &"Error: model file not found: {modelPath}"
    quit(1)

  echo "Loading model..."
  var ctx = loadInferenceContext(modelPath, maxContext)
  defer: unloadInferenceContext(ctx)

  let modelName = modelPath.extractFilename()
  echo &"Model: {modelName}"
  echo &"Context: {maxContext}, Max tokens: {maxTokens}"
  echo "Type 'quit' or 'exit' to stop."
  echo ""

  while true:
    stdout.write("> ")
    stdout.flushFile()
    var line: string
    try:
      if not stdin.readLine(line):
        break
    except EOFError:
      break

    line = line.strip()
    if line.len == 0:
      continue
    if line == "quit" or line == "exit":
      break

    let res = generate(ctx, line, maxTokens)
    echo res.text
    let tps = if res.elapsedMs > 0:
      float64(res.completionTokens) / (res.elapsedMs / 1000.0)
    else:
      0.0
    echo &"({res.promptTokens} prompt, {res.completionTokens} completion, {tps:.1f} tok/s)"
    echo ""
