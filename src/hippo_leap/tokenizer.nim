## Minimal tokenizer for GGUF models (SPM/LLaMA style).

import
  std/[tables, heapqueue, strutils, sequtils, algorithm, unicode],
  ./gguf_loader

const
  TokenModelKey = "tokenizer.ggml.model"
  TokenListKey = "tokenizer.ggml.tokens"
  TokenScoresKey = "tokenizer.ggml.scores"
  TokenTypesKey = "tokenizer.ggml.token_type"
  TokenAddBosKey = "tokenizer.ggml.add_bos_token"
  TokenAddEosKey = "tokenizer.ggml.add_eos_token"
  TokenAddPrefixKey = "tokenizer.ggml.add_space_prefix"
  TokenBosIdKey = "tokenizer.ggml.bos_token_id"
  TokenEosIdKey = "tokenizer.ggml.eos_token_id"
  TokenUnkIdKey = "tokenizer.ggml.unknown_token_id"
  TokenChatTemplateKey = "tokenizer.chat_template"

type
  TokenAttr* = enum
    tokNormal, tokUnknown, tokControl, tokUserDefined, tokUnused, tokByte

  TokenData* = object
    text*: string
    score*: float32
    attr*: TokenAttr

  Vocab* = object
    tokens*: seq[TokenData]
    tokenToId*: Table[string, int]
    mergeRank*: Table[string, int]
    addBos*: bool
    addEos*: bool
    addSpacePrefix*: bool
    bosId*: int32
    eosId*: int32
    unkId*: int32
    modelType*: string
    chatTemplate*: string
    stopTokenIds*: seq[int32]

  Symbol = object
    prev, next: int
    start, len: int

  Bigram = object
    keyScore: float32
    keyLeft: int
    left, right: int
    size: int

proc `<`(a, b: Bigram): bool =
  if a.keyScore == b.keyScore:
    return a.keyLeft < b.keyLeft
  a.keyScore < b.keyScore

proc utf8Len(b: byte): int =
  if b < 0x80: return 1
  if b < 0xE0: return 2
  if b < 0xF0: return 3
  if b < 0xF8: return 4
  1

proc escapeWhitespace(s: string): string =
  result = s.replace(" ", "\xE2\x96\x81")

proc hexByte(ch: byte): string =
  const hex = "0123456789ABCDEF"
  result = "<0x" & $hex[int(ch) shr 4] & $hex[int(ch) and 15] & ">"

proc byteToToken(v: Vocab, ch: byte): int32 =
  let t1 = hexByte(ch)
  if v.tokenToId.hasKey(t1):
    return int32(v.tokenToId[t1])
  let t2 = $char(ch)
  if v.tokenToId.hasKey(t2):
    return int32(v.tokenToId[t2])
  v.unkId

proc loadVocab*(g: GgufFile): Vocab =
  ## Load the tokenizer vocabulary from a GGUF file.
  var modelType: string
  discard g.getKvStr(TokenModelKey, modelType)
  var tokenList: seq[string]
  let okTokens = g.getKvArrStr(TokenListKey, tokenList)
  if not okTokens:
    raise newException(IOError, "tokenizer tokens missing")

  var scores: seq[float32]
  discard g.getKvArrF32(TokenScoresKey, scores)
  if scores.len != tokenList.len:
    scores.setLen(tokenList.len)

  var tokenTypes: seq[int32]
  discard g.getKvArrI32(TokenTypesKey, tokenTypes)
  if tokenTypes.len != tokenList.len:
    tokenTypes.setLen(tokenList.len)

  var addBos = false
  var addEos = false
  var addSpacePrefix = true
  let hasAddBos = g.getKvBool(TokenAddBosKey, addBos)
  discard g.getKvBool(TokenAddEosKey, addEos)
  discard g.getKvBool(TokenAddPrefixKey, addSpacePrefix)
  if modelType == "llama" and not hasAddBos:
    addBos = true

  var bosId: int32 = 1
  var eosId: int32 = 2
  var unkId: int32 = 0
  if not g.getKvI32(TokenBosIdKey, bosId):
    var u: uint32
    if g.getKvU32(TokenBosIdKey, u):
      bosId = int32(u)
  if not g.getKvI32(TokenEosIdKey, eosId):
    var u: uint32
    if g.getKvU32(TokenEosIdKey, u):
      eosId = int32(u)
  if not g.getKvI32(TokenUnkIdKey, unkId):
    var u: uint32
    if g.getKvU32(TokenUnkIdKey, u):
      unkId = int32(u)

  result.modelType = modelType
  result.addBos = addBos
  result.addEos = addEos
  result.addSpacePrefix = addSpacePrefix
  result.bosId = bosId
  result.eosId = eosId
  result.unkId = unkId
  discard g.getKvStr(TokenChatTemplateKey, result.chatTemplate)

  result.tokens.setLen(tokenList.len)
  for i, tok in tokenList:
    result.tokens[i].text = tok
    result.tokens[i].score = scores[i]
    result.tokens[i].attr = tokNormal

  result.tokenToId = initTable[string, int](tokenList.len * 2)
  for i, tok in tokenList:
    result.tokenToId[tok] = i

  if result.modelType == "gpt2":
    var mergeStrs: seq[string]
    if g.getKvArrStr("tokenizer.ggml.merges", mergeStrs):
      result.mergeRank = initTable[string, int](mergeStrs.len * 2)
      for i, m in mergeStrs:
        result.mergeRank[m] = i

    if result.bosId == 0 and result.tokenToId.hasKey("<|begin_of_text|>"):
      result.bosId = int32(result.tokenToId["<|begin_of_text|>"])
    if result.eosId == 0 and result.tokenToId.hasKey("<|end_of_text|>"):
      result.eosId = int32(result.tokenToId["<|end_of_text|>"])

  result.stopTokenIds = @[result.eosId]
  for s in ["<|eot_id|>", "<|end_of_text|>", "</s>", "<|im_end|>"]:
    if result.tokenToId.hasKey(s):
      let id = int32(result.tokenToId[s])
      if id notin result.stopTokenIds:
        result.stopTokenIds.add(id)

proc tokenizeSpm(v: Vocab, text: string): seq[int32] =
  var raw = text
  if v.addSpacePrefix and raw.len > 0:
    raw = " " & raw
  raw = escapeWhitespace(raw)

  var symbols: seq[Symbol] = @[]
  var revMerge = initTable[string, (int, int)]()
  var work = initHeapQueue[Bigram]()

  var index = 0
  var offs = 0
  while offs < raw.len:
    let b = raw[offs].byte
    let n = min(utf8Len(b), raw.len - offs)
    var sym: Symbol
    sym.start = offs
    sym.len = n
    sym.prev = index - 1
    sym.next = (if offs + n == raw.len: -1 else: index + 1)
    symbols.add(sym)
    offs += n
    inc index

  proc tryAddBigram(left, right: int) =
    if left < 0 or right < 0: return
    let a = symbols[left]
    let b = symbols[right]
    if a.len == 0 or b.len == 0: return
    let textLen = a.len + b.len
    let text = raw.substr(a.start, a.start + textLen - 1)
    if not v.tokenToId.hasKey(text): return
    let id = v.tokenToId[text]
    if id < 0 or id >= v.tokens.len: return
    let score = v.tokens[id].score
    work.push(Bigram(
      keyScore: -score,
      keyLeft: left,
      left: left,
      right: right,
      size: textLen
    ))
    revMerge[text] = (left, right)

  for i in 1 ..< symbols.len:
    tryAddBigram(i - 1, i)

  while work.len > 0:
    let bigram = work.pop()
    var leftSym = symbols[bigram.left]
    var rightSym = symbols[bigram.right]
    if leftSym.len == 0 or rightSym.len == 0: continue
    if leftSym.len + rightSym.len != bigram.size: continue

    leftSym.len += rightSym.len
    rightSym.len = 0
    leftSym.next = rightSym.next
    if rightSym.next >= 0:
      symbols[rightSym.next].prev = bigram.left
    symbols[bigram.left] = leftSym
    symbols[bigram.right] = rightSym

    tryAddBigram(leftSym.prev, bigram.left)
    tryAddBigram(bigram.left, leftSym.next)

  proc resegment(idx: int, outp: var seq[int32]) =
    let sym = symbols[idx]
    if sym.len == 0: return
    let text = raw.substr(sym.start, sym.start + sym.len - 1)
    if v.tokenToId.hasKey(text):
      outp.add(int32(v.tokenToId[text]))
      return
    if revMerge.hasKey(text):
      let (l, r) = revMerge[text]
      resegment(l, outp)
      resegment(r, outp)
      return
    for i in 0 ..< sym.len:
      outp.add(byteToToken(v, raw[sym.start + i].byte))

  var outp: seq[int32] = @[]
  var i = 0
  while i >= 0 and i < symbols.len:
    if symbols[i].prev == -1 or i == 0:
      break
    i = symbols[i].prev
  for j in 0 ..< symbols.len:
    if symbols[j].prev == -1:
      i = j
      break
  while i != -1:
    resegment(i, outp)
    i = symbols[i].next

  outp

proc buildByteToUnicode(): array[256, string] =
  ## GPT-2 byte-to-unicode mapping.
  var n = 0
  for b in 0..255:
    if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255):
      var s: string
      let r = Rune(b)
      s.add(r)
      result[b] = s
    else:
      var s: string
      let r = Rune(256 + n)
      s.add(r)
      result[b] = s
      inc n

let byteToUnicode = buildByteToUnicode()

proc buildUnicodeToByte(): Table[Rune, byte] =
  result = initTable[Rune, byte]()
  for b in 0..255:
    var r: Rune
    fastRuneAt(byteToUnicode[b], 0, r, false)
    result[r] = byte(b)

let unicodeToByte = buildUnicodeToByte()

proc tokenizeBpe(v: Vocab, text: string): seq[int32] =
  ## Byte-level BPE tokenizer for GPT-2 style models (Llama 3, etc.).
  if text.len == 0:
    return @[]

  var parts: seq[string]
  for b in text:
    parts.add(byteToUnicode[int(byte(b))])

  while parts.len >= 2:
    var bestRank = high(int)
    var bestIdx = -1
    for i in 0 ..< parts.len - 1:
      let pair = parts[i] & " " & parts[i + 1]
      if v.mergeRank.hasKey(pair):
        let rank = v.mergeRank[pair]
        if rank < bestRank:
          bestRank = rank
          bestIdx = i
    if bestIdx < 0:
      break
    let merged = parts[bestIdx] & parts[bestIdx + 1]
    parts.delete(bestIdx + 1)
    parts[bestIdx] = merged

  for p in parts:
    if v.tokenToId.hasKey(p):
      result.add(int32(v.tokenToId[p]))
    else:
      result.add(v.unkId)

proc tokenize*(v: Vocab, text: string, addSpecial = true): seq[int32] =
  ## Tokenize text, dispatching between SPM and BPE based on model type.
  if v.modelType == "gpt2":
    result = tokenizeBpe(v, text)
  else:
    result = tokenizeSpm(v, text)
  if addSpecial and v.addBos:
    result.insert(v.bosId, 0)
  if addSpecial and v.addEos:
    result.add(v.eosId)

proc tokenizeRaw(v: Vocab, text: string): seq[int32] =
  if v.modelType == "gpt2":
    tokenizeBpe(v, text)
  else:
    tokenizeSpm(v, text)

proc tokenizeWithSpecial*(v: Vocab, text: string, addSpecial = true): seq[int32] =
  ## Tokenize text while preserving special tokens as single token IDs.
  var specials = @["<|user|>", "<|assistant|>", "<|system|>", "</s>", "<s>",
                   "<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>",
                   "<|end_header_id|>", "<|eot_id|>",
                   "<|im_start|>", "<|im_end|>"]
  specials = specials.filterIt(v.tokenToId.hasKey(it))
  if specials.len == 0:
    return tokenize(v, text, addSpecial)

  specials.sort(proc(a, b: string): int = cmp(b.len, a.len))

  var pos = 0
  var outTokens: seq[int32] = @[]
  while pos < text.len:
    var bestIdx = -1
    var bestTok = ""
    for s in specials:
      let i = text.find(s, pos)
      if i >= 0 and (bestIdx == -1 or i < bestIdx or (i == bestIdx and s.len > bestTok.len)):
        bestIdx = i
        bestTok = s
    if bestIdx == -1:
      outTokens.add(tokenizeRaw(v, text.substr(pos)))
      break
    if bestIdx > pos:
      outTokens.add(tokenizeRaw(v, text.substr(pos, bestIdx - 1)))
    outTokens.add(int32(v.tokenToId[bestTok]))
    pos = bestIdx + bestTok.len

  if addSpecial and v.addBos:
    outTokens.insert(v.bosId, 0)
  if addSpecial and v.addEos:
    outTokens.add(v.eosId)
  outTokens

proc tokenToPiece*(v: Vocab, id: int32): string =
  ## Convert a token ID back to its string piece.
  let idx = int(id)
  if idx < 0 or idx >= v.tokens.len:
    return ""
  result = v.tokens[idx].text

proc detokenize*(v: Vocab, tokens: seq[int32]): string =
  ## Convert a sequence of token IDs back to text.
  result = ""
  for t in tokens:
    if t in v.stopTokenIds or t == v.bosId:
      continue
    let piece = tokenToPiece(v, t)
    if piece.startsWith("<|") and piece.endsWith("|>"):
      continue
    if piece == "<s>" or piece == "</s>":
      continue
    if piece.len == 6 and piece.startsWith("<0x") and piece.endsWith(">"):
      let hex = piece[3..4]
      try:
        let b = parseHexInt(hex)
        result.add(char(b))
      except ValueError:
        result.add(piece)
    else:
      result.add(piece)
  if v.modelType == "gpt2":
    var decoded = ""
    for r in result.runes:
      if unicodeToByte.hasKey(r):
        decoded.add(char(unicodeToByte[r]))
      else:
        decoded.add(r)
    result = decoded
  else:
    result = result.replace("\xE2\x96\x81", " ")

proc formatChatPrompt*(v: Vocab, userText: string): string =
  ## Format user text using the model's chat template.
  if v.chatTemplate.len == 0:
    return userText
  if v.chatTemplate.contains("<|start_header_id|>"):
    return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" &
           userText & "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
  if v.chatTemplate.contains("<|im_start|>"):
    return "<|im_start|>user\n" & userText & "<|im_end|>\n<|im_start|>assistant\n"
  if v.chatTemplate.contains("<|user|>") and v.chatTemplate.contains("<|assistant|>"):
    let eosPiece = tokenToPiece(v, v.eosId)
    return "<|user|>\n" & userText & eosPiece & "<|assistant|>"
  userText
