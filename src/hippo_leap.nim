import
  std/[os, strformat],
  ./hippo_leap/[common, config, server, cmd_chat, cmd_bench, cmd_tokenize]

const
  Usage = """hippo_leap - OpenAI-compatible LLM inference server

Usage:
  hippo_leap serve           Start the inference server
  hippo_leap chat            Interactive chat REPL
  hippo_leap bench           Run inference benchmarks
  hippo_leap tokenize <text> Show tokenization of text
  hippo_leap --version       Print version
  hippo_leap --help          Show this help

Environment:
  HIPPO_LEAP_MODEL           Path to GGUF model file (required for inference)
  HIPPO_LEAP_PORT            Server port (default: 8080)
  HIPPO_LEAP_ADDRESS         Bind address (default: 0.0.0.0)
  HIPPO_LEAP_MAX_TOKENS      Default max tokens per response (default: 256)
  HIPPO_LEAP_MAX_CONTEXT     Max context length (default: 2048)"""

proc cmdServe() =
  ## Start the HTTP inference server.
  let cfg = loadConfig()
  if cfg.modelPath.len == 0:
    echo "WARNING: HIPPO_LEAP_MODEL not set. Server will start without inference."
  elif not fileExists(cfg.modelPath):
    echo &"Error: model file not found: {cfg.modelPath}"
    quit(1)
  runServer(cfg)

when isMainModule:
  let args = commandLineParams()

  if args.len == 0:
    echo Usage
    quit(0)

  case args[0]
  of "serve":
    cmdServe()
  of "chat":
    cmdChat(args[1..^1])
  of "bench":
    cmdBench(args[1..^1])
  of "tokenize":
    cmdTokenize(args[1..^1])
  of "--version":
    echo Version
  of "--help", "-h":
    echo Usage
  else:
    echo &"hippo_leap: unknown command '{args[0]}'"
    echo Usage
    quit(1)
