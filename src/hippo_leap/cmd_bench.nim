## Structured benchmark runner CLI command.

import
  std/[os, strutils, strformat, math],
  ./inference

const
  BenchUsage = """Usage: hippo_leap bench [options]

Options:
  --model, -m <path>    GGUF model file (or HIPPO_LEAP_MODEL env)
  --prompt, -p <text>   Prompt text (default: "Write a short story about a cat.")
  --tokens, -n <int>    Max tokens to generate (default: 128)
  --context <int>       Max context length (default: 2048)
  --runs, -r <int>      Number of benchmark runs (default: 5)
  --warmup <int>        Warmup runs before timing (default: 3)"""

proc cmdBench*(args: seq[string]) =
  var modelPath = getEnv("HIPPO_LEAP_MODEL", "")
  var prompt = "Write a short story about a cat."
  var maxTokens = 128
  var maxContext = 2048
  var runs = 5
  var warmup = 3

  var i = 0
  while i < args.len:
    case args[i]
    of "--model", "-m":
      inc i
      if i >= args.len:
        echo "Error: --model requires a path"
        quit(1)
      modelPath = args[i]
    of "--prompt", "-p":
      inc i
      if i >= args.len:
        echo "Error: --prompt requires text"
        quit(1)
      prompt = args[i]
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
    of "--runs", "-r":
      inc i
      if i >= args.len:
        echo "Error: --runs requires a number"
        quit(1)
      runs = parseInt(args[i])
    of "--warmup":
      inc i
      if i >= args.len:
        echo "Error: --warmup requires a number"
        quit(1)
      warmup = parseInt(args[i])
    of "--help", "-h":
      echo BenchUsage
      quit(0)
    else:
      echo &"Unknown option: {args[i]}"
      echo BenchUsage
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
  echo &"Prompt: \"{prompt}\""
  echo &"Max tokens: {maxTokens}"
  echo &"Context: {maxContext}"
  echo &"Runs: {runs} ({warmup} warmup)"
  echo ""

  for w in 0 ..< warmup:
    echo &"Warmup {w + 1}/{warmup}..."
    discard generate(ctx, prompt, maxTokens)

  echo ""
  echo " Run  Tokens  Time(ms)    tok/s"

  var toksSec: seq[float64]

  for r in 0 ..< runs:
    let res = generate(ctx, prompt, maxTokens)
    let tps = if res.elapsedMs > 0:
      float64(res.completionTokens) / (res.elapsedMs / 1000.0)
    else:
      0.0
    toksSec.add(tps)
    echo &"{r + 1:>4}  {res.completionTokens:>6}  {res.elapsedMs:>9.1f}  {tps:>7.1f}"

  if toksSec.len > 0:
    var sum = 0.0
    var minVal = toksSec[0]
    var maxVal = toksSec[0]
    for v in toksSec:
      sum += v
      if v < minVal: minVal = v
      if v > maxVal: maxVal = v
    let mean = sum / float64(toksSec.len)
    var variance = 0.0
    for v in toksSec:
      variance += (v - mean) * (v - mean)
    let stddev = sqrt(variance / float64(toksSec.len))
    echo ""
    echo &"Mean: {mean:.2f} ± {stddev:.2f} tok/s"
    echo &"Min:  {minVal:.1f} tok/s"
    echo &"Max:  {maxVal:.1f} tok/s"
