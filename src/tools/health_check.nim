import
  std/[os, strformat],
  curly

const
  DefaultUrl = "http://127.0.0.1:8080/health"
  Usage = """health_check - Check if hippo_leap server is running

Usage:
  health_check [url]

If no URL is provided, defaults to """ & DefaultUrl

proc main() =
  ## Check if the hippo_leap server is healthy.
  let args = commandLineParams()
  if args.len > 0 and args[0] in ["--help", "-h"]:
    echo Usage
    quit(0)

  let url = if args.len > 0: args[0] else: DefaultUrl
  let pool = newCurly()
  let resp = pool.get(url)
  if resp.code == 200:
    echo &"OK: {url} returned {resp.code}"
    echo resp.body
  else:
    echo &"FAIL: {url} returned {resp.code}"
    quit(1)

when isMainModule:
  main()
