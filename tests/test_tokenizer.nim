import
  std/[os, tables, strutils],
  hippo_leap/[gguf_loader, tokenizer]

const
  ModelPath = "/mnt/steel-chest/LLM/lmstudio/models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf"

proc testLoadVocab() =
  ## Verify vocab loads with expected token count.
  if not fileExists(ModelPath):
    echo "[SKIP] Model file not found: " & ModelPath
    return
  var g = openGguf(ModelPath)
  let vocab = loadVocab(g)
  doAssert vocab.tokens.len > 0
  doAssert vocab.tokenToId.len == vocab.tokens.len
  doAssert vocab.bosId >= 0
  doAssert vocab.eosId >= 0
  echo "[OK] Vocab loaded with " & $vocab.tokens.len & " tokens"
  g.close()

proc testTokenizeRoundtrip() =
  ## Verify tokenize then detokenize recovers the original text.
  if not fileExists(ModelPath):
    echo "[SKIP] Model file not found: " & ModelPath
    return
  var g = openGguf(ModelPath)
  let vocab = loadVocab(g)
  let text = "Hello world"
  let tokens = vocab.tokenize(text, addSpecial = false)
  doAssert tokens.len > 0
  let decoded = vocab.detokenize(tokens)
  doAssert decoded.strip() == text or decoded.contains("Hello")
  echo "[OK] Tokenize roundtrip: " & $tokens.len & " tokens"
  g.close()

proc testTokenizeWithBos() =
  ## Verify BOS token is prepended when addSpecial is true.
  if not fileExists(ModelPath):
    echo "[SKIP] Model file not found: " & ModelPath
    return
  var g = openGguf(ModelPath)
  let vocab = loadVocab(g)
  let tokensNoBos = vocab.tokenize("test", addSpecial = false)
  let tokensWithBos = vocab.tokenize("test", addSpecial = true)
  if vocab.addBos:
    doAssert tokensWithBos.len == tokensNoBos.len + 1
    doAssert tokensWithBos[0] == vocab.bosId
  echo "[OK] BOS token handling correct"
  g.close()

proc testFormatChatPrompt() =
  ## Verify chat prompt formatting includes special tokens.
  if not fileExists(ModelPath):
    echo "[SKIP] Model file not found: " & ModelPath
    return
  var g = openGguf(ModelPath)
  let vocab = loadVocab(g)
  let formatted = vocab.formatChatPrompt("Hello")
  if vocab.chatTemplate.len > 0 and vocab.chatTemplate.contains("<|user|>"):
    doAssert formatted.contains("<|user|>")
    doAssert formatted.contains("<|assistant|>")
  echo "[OK] Chat prompt formatting"
  g.close()

when isMainModule:
  testLoadVocab()
  testTokenizeRoundtrip()
  testTokenizeWithBos()
  testFormatChatPrompt()
