import
  std/[os, strformat],
  ./hippo_leap/[common, config, server]

const
  Usage = """hippo_leap - OpenAI-compatible LLM inference server

Usage:
  hippo_leap serve           Start the inference server
  hippo_leap --version       Print version
  hippo_leap --help          Show this help"""

proc cmdServe() =
  ## Start the HTTP inference server.
  let cfg = loadConfig()
  runServer(cfg)

when isMainModule:
  let args = commandLineParams()

  if args.len == 0:
    echo Usage
    quit(0)

  case args[0]
  of "serve":
    cmdServe()
  of "--version":
    echo Version
  of "--help", "-h":
    echo Usage
  else:
    echo &"hippo_leap: unknown command '{args[0]}'"
    echo Usage
    quit(1)
