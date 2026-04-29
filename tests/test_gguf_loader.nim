import
  std/[os],
  hippo_leap/gguf_loader

const
  ModelPath = "/mnt/steel-chest/LLM/lmstudio/models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf"

proc testOpenGguf() =
  ## Verify GGUF file opens and has valid metadata.
  if not fileExists(ModelPath):
    echo "[SKIP] Model file not found: " & ModelPath
    return
  var g = openGguf(ModelPath)
  doAssert g.size > 0
  doAssert g.kv.len > 0
  doAssert g.tensors.len > 0
  doAssert g.alignment == 32'u32
  echo "[OK] openGguf loaded " & $g.kv.len & " KV pairs, " & $g.tensors.len & " tensors"
  g.close()

proc testKvLookups() =
  ## Verify KV lookup helpers return expected values.
  if not fileExists(ModelPath):
    echo "[SKIP] Model file not found: " & ModelPath
    return
  var g = openGguf(ModelPath)
  var arch: string
  doAssert g.getKvStr("general.architecture", arch)
  doAssert arch == "llama"

  var nEmb: uint32
  doAssert g.getKvU32("llama.embedding_length", nEmb)
  doAssert nEmb == 2048

  var nLayer: uint32
  doAssert g.getKvU32("llama.block_count", nLayer)
  doAssert nLayer == 22

  var nHead: uint32
  doAssert g.getKvU32("llama.attention.head_count", nHead)
  doAssert nHead == 32

  echo "[OK] KV lookups return correct values"
  g.close()

proc testTensorInfo() =
  ## Verify tensor info has valid names and dimensions.
  if not fileExists(ModelPath):
    echo "[SKIP] Model file not found: " & ModelPath
    return
  var g = openGguf(ModelPath)
  doAssert g.tensors.len > 0
  var foundEmbed = false
  for t in g.tensors:
    doAssert t.name.len > 0
    doAssert t.nDims >= 1 and t.nDims <= 4
    if t.name == "token_embd.weight":
      foundEmbed = true
  doAssert foundEmbed, "expected token_embd.weight tensor"
  echo "[OK] Tensor info is valid"
  g.close()

proc testMissingKv() =
  ## Verify missing KV returns false without raising.
  if not fileExists(ModelPath):
    echo "[SKIP] Model file not found: " & ModelPath
    return
  var g = openGguf(ModelPath)
  var s: string
  doAssert not g.getKvStr("nonexistent.key", s)
  echo "[OK] Missing KV returns false"
  g.close()

when isMainModule:
  testOpenGguf()
  testKvLookups()
  testTensorInfo()
  testMissingKv()
