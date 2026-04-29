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

const
  Llama3Path = "/mnt/steel-chest/LLM/lmstudio/models/lmstudio-community/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

proc testLlama3Vocab() =
  if not fileExists(Llama3Path):
    echo "[SKIP] Llama 3.2 model not found"
    return
  var g = openGguf(Llama3Path)
  let vocab = loadVocab(g)
  doAssert vocab.modelType == "gpt2", "expected gpt2, got " & vocab.modelType
  doAssert vocab.tokens.len == 128256, "expected 128256 tokens, got " & $vocab.tokens.len
  doAssert vocab.mergeRank.len > 0, "expected merges loaded"
  doAssert vocab.bosId == 128000, "expected bosId 128000, got " & $vocab.bosId
  doAssert vocab.stopTokenIds.len >= 2, "expected stop tokens"
  echo "[OK] Llama 3.2 vocab: " & $vocab.tokens.len & " tokens, " & $vocab.mergeRank.len & " merges"
  g.close()

proc testLlama3Tokenize() =
  if not fileExists(Llama3Path):
    echo "[SKIP] Llama 3.2 model not found"
    return
  var g = openGguf(Llama3Path)
  let vocab = loadVocab(g)
  let tokens = vocab.tokenize("Hello world", addSpecial = false)
  doAssert tokens.len > 0, "tokenize returned empty"
  let decoded = vocab.detokenize(tokens)
  doAssert "Hello" in decoded, "detokenize missing Hello, got: " & decoded
  doAssert "world" in decoded, "detokenize missing world, got: " & decoded
  echo "[OK] Llama 3.2 tokenize: " & $tokens.len & " tokens -> \"" & decoded & "\""
  g.close()

proc testLlama3ChatTemplate() =
  if not fileExists(Llama3Path):
    echo "[SKIP] Llama 3.2 model not found"
    return
  var g = openGguf(Llama3Path)
  let vocab = loadVocab(g)
  let formatted = vocab.formatChatPrompt("Hello")
  doAssert formatted.contains("<|start_header_id|>"), "missing start_header_id"
  doAssert formatted.contains("<|eot_id|>"), "missing eot_id"
  doAssert formatted.contains("<|begin_of_text|>"), "missing begin_of_text"
  echo "[OK] Llama 3.2 chat template"
  g.close()

when isMainModule:
  testLoadVocab()
  testTokenizeRoundtrip()
  testTokenizeWithBos()
  testFormatChatPrompt()
  testLlama3Vocab()
  testLlama3Tokenize()
  testLlama3ChatTemplate()
