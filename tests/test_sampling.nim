import
  hippo_leap/[tensor, sampling]

proc testArgmaxLastRowMajor() =
  ## Verify argmax on row-major logits (seqLen x nVocab).
  var logits = newTensor(@[1, 4])
  logits.data = @[0.1'f32, 0.5'f32, 0.3'f32, 0.9'f32]
  let best = argmaxLast(logits, 4)
  doAssert best == 3
  echo "[OK] argmaxLast row-major"

proc testArgmaxLastColMajor() =
  ## Verify argmax on column-major logits (nVocab x seqLen).
  var logits = newTensor(@[4, 1])
  logits.data = @[0.2'f32, 0.8'f32, 0.1'f32, 0.4'f32]
  let best = argmaxLast(logits, 4)
  doAssert best == 1
  echo "[OK] argmaxLast col-major"

proc testArgmaxLastMultiPosition() =
  ## Verify argmax picks from the last position in a multi-position tensor.
  var logits = newTensor(@[2, 4])
  logits.data = @[
    0.9'f32, 0.1'f32, 0.1'f32, 0.1'f32,
    0.1'f32, 0.1'f32, 0.1'f32, 0.8'f32
  ]
  let best = argmaxLast(logits, 4)
  doAssert best == 3
  echo "[OK] argmaxLast multi-position"

when isMainModule:
  testArgmaxLastRowMajor()
  testArgmaxLastColMajor()
  testArgmaxLastMultiPosition()
