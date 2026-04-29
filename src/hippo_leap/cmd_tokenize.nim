## Tokenization inspector CLI command.

import
  std/[os, strutils, strformat],
  ./[model, tokenizer]

const
  TokenizeUsage = """Usage: hippo_leap tokenize [options] <text>

Options:
  --model, -m <path>    GGUF model file (or HIPPO_LEAP_MODEL env)
  --special              Include BOS/EOS tokens in output
  --ids-only             Print only token IDs, one per line

If no text is given, reads from stdin."""

proc cmdTokenize*(args: seq[string]) =
  var modelPath = getEnv("HIPPO_LEAP_MODEL", "")
  var addSpecial = false
  var idsOnly = false
  var textParts: seq[string]

  var i = 0
  while i < args.len:
    case args[i]
    of "--model", "-m":
      inc i
      if i >= args.len:
        echo "Error: --model requires a path"
        quit(1)
      modelPath = args[i]
    of "--special":
      addSpecial = true
    of "--ids-only":
      idsOnly = true
    of "--help", "-h":
      echo TokenizeUsage
      quit(0)
    else:
      textParts.add(args[i])
    inc i

  if modelPath.len == 0:
    echo "Error: no model specified. Use --model or set HIPPO_LEAP_MODEL"
    quit(1)
  if not fileExists(modelPath):
    echo &"Error: model file not found: {modelPath}"
    quit(1)

  var text = textParts.join(" ")
  if text.len == 0:
    text = stdin.readAll().strip()
  if text.len == 0:
    echo "Error: no input text provided"
    quit(1)

  var m = loadModel(modelPath)
  defer: m.close()
  let vocab = loadVocab(m.gguf)

  let tokens = tokenizeWithSpecial(vocab, text, addSpecial = addSpecial)

  if idsOnly:
    for id in tokens:
      echo id
  else:
    echo &"Token count: {tokens.len}"
    for id in tokens:
      let piece = tokenToPiece(vocab, id)
      var label = ""
      if id == vocab.bosId: label = "  (BOS)"
      elif id == vocab.eosId: label = "  (EOS)"
      echo &"  [{id}]\t{piece}{label}"
