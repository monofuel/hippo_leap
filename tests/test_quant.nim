import
  std/[math],
  hippo_leap/quant

proc testHalfToFloat() =
  ## Verify known half-precision to float32 conversions.
  doAssert halfToFloat(0x0000'u16) == 0.0'f32
  doAssert halfToFloat(0x3C00'u16) == 1.0'f32
  doAssert halfToFloat(0xBC00'u16) == -1.0'f32
  doAssert halfToFloat(0x4000'u16) == 2.0'f32
  doAssert halfToFloat(0x3800'u16) == 0.5'f32
  doAssert classify(halfToFloat(0x7C00'u16)) == fcInf
  doAssert classify(halfToFloat(0xFC00'u16)) == fcNegInf
  doAssert classify(halfToFloat(0x7C01'u16)) == fcNan
  echo "[OK] halfToFloat known values"

proc testRowSizeCalculations() =
  ## Verify row size calculations for various quant formats.
  doAssert rowSizeQ2K(256) == BlockQ2KSize
  doAssert rowSizeQ2K(512) == 2 * BlockQ2KSize
  doAssert rowSizeQ3K(256) == BlockQ3KSize
  doAssert rowSizeQ3K(512) == 2 * BlockQ3KSize
  doAssert rowSizeQ4K(256) == BlockQ4KSize
  doAssert rowSizeQ4K(512) == 2 * BlockQ4KSize
  doAssert rowSizeQ6K(256) == BlockQ6KSize
  doAssert rowSizeQ6K(512) == 2 * BlockQ6KSize
  doAssert rowSizeQ8_0(32) == BlockQ8_0Size
  doAssert rowSizeQ8_0(64) == 2 * BlockQ8_0Size
  echo "[OK] Row size calculations"

proc testRowSizeInvalidAlignment() =
  ## Verify row size raises on non-256-aligned lengths.
  var caught = false
  try:
    discard rowSizeQ2K(100)
  except ValueError:
    caught = true
  doAssert caught, "expected ValueError for non-256-aligned Q2K row"

  caught = false
  try:
    discard rowSizeQ3K(100)
  except ValueError:
    caught = true
  doAssert caught, "expected ValueError for non-256-aligned Q3K row"

  caught = false
  try:
    discard rowSizeQ4K(100)
  except ValueError:
    caught = true
  doAssert caught, "expected ValueError for non-256-aligned Q4K row"

  caught = false
  try:
    discard rowSizeQ6K(100)
  except ValueError:
    caught = true
  doAssert caught, "expected ValueError for non-256-aligned Q6K row"

  caught = false
  try:
    discard rowSizeQ8_0(100)
  except ValueError:
    caught = true
  doAssert caught, "expected ValueError for non-32-aligned Q8_0 row"
  echo "[OK] Row size invalid alignment raises"

proc testBlockSizeConstants() =
  ## Verify block size constants match expected values.
  doAssert QK_K == 256
  doAssert QK8_0 == 32
  doAssert BlockQ2KSize == 84
  doAssert BlockQ3KSize == 110
  doAssert BlockQ4KSize == 144
  doAssert BlockQ6KSize == 210
  doAssert BlockQ8_0Size == 34
  echo "[OK] Block size constants"

when isMainModule:
  testHalfToFloat()
  testRowSizeCalculations()
  testRowSizeInvalidAlignment()
  testBlockSizeConstants()
