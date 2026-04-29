## WMMA lane mapping discovery and verification for RDNA3+ wave32.
##
## Empirically determined mapping for __builtin_amdgcn_wmma_f32_16x16x16_f16_w32:
##
##   Fragment A: A(tid, i) = A[tid % 16][i]        (thread = row, index = col)
##   Fragment B: B(tid, i) = B[i][tid % 16]        (index = row, thread = col)
##   Fragment C: C(tid, i) = C[2*i + tid/16][tid % 16]  (interleaved rows)
##
## Threads 0-15 and 16-31 redundantly store A and B.
## For C, threads 0-15 cover even rows, 16-31 cover odd rows.
##
## Requires: nim cpp --cc:hipcc -d:useMalloc -d:HippoRuntime:HIP -r

import hippo, std/[strformat, math]

proc wmmaVerifyMappingKernel(
  outData: ptr cfloat,
  errCount: ptr int32
) {.hippoGlobal.} =
  ## Verify the lane mapping by computing a known matmul and checking results.
  ## A[r][c] = r + 1, B[r][c] = c + 1, so C[r][c] = sum_k (r+1)*(c+1) = 16*(r+1)*(c+1)
  ## (since sum_k=0..15 of 1 = 16, factored out the constants)
  ## Actually: C[r][c] = sum_k A[r][k] * B[k][c] = sum_k (r+1)*(k+1) ... no.
  ## Let's use: A[r][c] = r*16 + c (unique), B = identity. C = A * I = A.
  ## More useful: A[r][c] = float(r), B[r][c] = float(c). C[r][c] = sum_k r*c = 16*r*c.
  ## Hmm, let's just do A = sequential, B = identity col 0.
  ##
  ## Simpler: A[r][c] = 1 for all, B[r][c] = 1 for all. C[r][c] = 16.
  ## Already tested. Let's do the actual mapping verification.

  let tid = cint(threadIdx.x)
  let outArr = cast[ptr UncheckedArray[cfloat]](outData)
  let errPtr = errCount

  # Fill A: A[tid%16][i] = (tid%16)*100 + i  (row*100 + col gives unique per-position)
  var fragA {.noinit.}: HippoWmmaHalf16
  let myRow = tid mod 16'i32
  for i in 0'i32 ..< 16'i32:
    let val = cfloat(myRow * 100'i32 + i)
    hippoWmmaSetF16(fragA, i, hippoFloatToHalf(val))

  # Fill B: B[i][tid%16] = broadcast x. Set B so all cols are same: x[k] = k+1
  # B[row][col] = row + 1 for all col → fragB[i] = i+1 for all threads
  var fragB {.noinit.}: HippoWmmaHalf16
  for i in 0'i32 ..< 16'i32:
    hippoWmmaSetF16(fragB, i, hippoFloatToHalf(cfloat(i + 1'i32)))

  var fragC = hippoWmmaZeroF32()
  fragC = hippoWmmaF32_16x16x16_f16(fragA, fragB, fragC)

  # Expected: C[r][c] = sum_k A[r][k] * B[k][c] = sum_k (r*100+k) * (k+1)
  # = r*100 * sum(k+1) + sum(k*(k+1))
  # = r*100 * 136 + sum(k^2 + k) = r*100*136 + 1240 + 120 = 13600*r + 1360
  # All columns same since B broadcast.

  # Verify C output matches expected using the mapping C(tid,i) = C[2*i + tid/16][tid%16]
  var localErr = 0'i32
  for i in 0'i32 ..< 8'i32:
    let cRow = 2'i32 * i + tid div 16'i32
    let expected = 13600.0'f32 * cfloat(cRow) + 1360.0'f32
    let got = hippoWmmaGetF32(fragC, i)
    outArr[tid * 8'i32 + i] = got
    let diff = got - expected
    if (if diff < 0.0'f32: -diff else: diff) > 1.0'f32:
      localErr = localErr + 1'i32

  discard hippoAtomicAdd(errPtr, localErr)

proc wmmaGemvTestKernel(
  xData, outData: ptr cfloat
) {.hippoGlobal.} =
  ## Test the GEMV pattern: 16 rows of weights × x vector.
  ## Weights: W[r][c] = r + c (simple pattern).
  ## x[c] = c + 1.
  ## Expected: y[r] = sum_c (r+c)*(c+1) = sum_c (rc + r + c^2 + c)
  ##   = r*sum(c+1) + sum(c^2+c) = r*136 + 1360
  let tid = cint(threadIdx.x)
  let xArr = cast[ptr UncheckedArray[cfloat]](xData)
  let outArr = cast[ptr UncheckedArray[cfloat]](outData)

  let myRow = tid mod 16'i32

  var fragC = hippoWmmaZeroF32()

  # Single tile: 16×16 weight matrix × 16-element x vector
  var fragA {.noinit.}: HippoWmmaHalf16
  for i in 0'i32 ..< 16'i32:
    let wVal = cfloat(myRow + i)
    hippoWmmaSetF16(fragA, i, hippoFloatToHalf(wVal))

  # B = x broadcast: B[k][j] = x[k] for all j
  # B(tid, i) = B[i][tid%16] = x[i]
  var fragB {.noinit.}: HippoWmmaHalf16
  for i in 0'i32 ..< 16'i32:
    hippoWmmaSetF16(fragB, i, hippoFloatToHalf(xArr[i]))

  fragC = hippoWmmaF32_16x16x16_f16(fragA, fragB, fragC)

  # Write all thread results for debugging
  for i in 0'i32 ..< 8'i32:
    outArr[tid * 8'i32 + i] = hippoWmmaGetF32(fragC, i)

proc main() =
  echo "=== WMMA Lane Mapping Verification ==="
  echo ""
  echo "Confirmed mapping (RDNA3 wave32):"
  echo "  A(tid, i) = A[tid % 16][i]"
  echo "  B(tid, i) = B[i][tid % 16]"
  echo "  C(tid, i) = C[2*i + tid/16][tid % 16]"
  echo ""

  # Test 1: Mapping verification
  block:
    echo "--- Test 1: Verify mapping with known matmul ---"
    var hostOut: array[256, cfloat]
    var hostErr: int32 = 0

    let devOut = hippoMalloc(256 * sizeof(cfloat))
    let devErr = hippoMalloc(1 * sizeof(int32))
    hippoMemcpy(devErr, addr hostErr, cint(sizeof(int32)), HippoMemcpyHostToDevice)

    let grid = newDim3(1'u32)
    let blk = newDim3(32'u32)
    hippoLaunchKernel(wmmaVerifyMappingKernel, gridDim = grid, blockDim = blk,
                      args = hippoArgs(devOut.p, devErr.p))
    discard hipDeviceSynchronize()

    hippoMemcpy(addr hostErr, devErr, cint(sizeof(int32)), HippoMemcpyDeviceToHost)
    hippoMemcpy(addr hostOut[0], devOut, cint(256 * sizeof(cfloat)), HippoMemcpyDeviceToHost)

    if hostErr == 0:
      echo "  PASS: All C elements match expected values"
    else:
      echo &"  FAIL: {hostErr} mismatches"
      for tid in 0 ..< 32:
        for i in 0 ..< 8:
          let cRow = 2 * i + tid div 16
          let expected = 13600.0'f32 * float32(cRow) + 1360.0'f32
          let got = hostOut[tid * 8 + i]
          if abs(got - expected) > 1.0:
            echo &"    C(t{tid},i{i}) = {got}, expected {expected} (row={cRow})"

  # Test 2: GEMV pattern verification
  block:
    echo "--- Test 2: GEMV pattern (W[r][c]=r+c, x[c]=c+1) ---"
    var hostX: array[16, cfloat]
    var hostOut: array[256, cfloat]

    for i in 0 ..< 16:
      hostX[i] = cfloat(i + 1)

    let devX = hippoMalloc(16 * sizeof(cfloat))
    let devOut = hippoMalloc(256 * sizeof(cfloat))
    hippoMemcpy(devX, addr hostX[0], cint(16 * sizeof(cfloat)), HippoMemcpyHostToDevice)

    let grid = newDim3(1'u32)
    let blk = newDim3(32'u32)
    hippoLaunchKernel(wmmaGemvTestKernel, gridDim = grid, blockDim = blk,
                      args = hippoArgs(devX.p, devOut.p))
    discard hipDeviceSynchronize()

    hippoMemcpy(addr hostOut[0], devOut, cint(256 * sizeof(cfloat)), HippoMemcpyDeviceToHost)

    # C(tid,i) = C[2*i + tid/16][tid%16]
    # For GEMV broadcast, all columns should be equal
    echo "  Full C dump (row via mapping):"
    for tid in 0 ..< 32:
      var line = &"  tid={tid:2d} (col={tid mod 16:2d}): "
      for i in 0 ..< 8:
        let cRow = 2 * i + tid div 16
        line.add(&" r{cRow:2d}={hostOut[tid * 8 + i]:8.1f}")
      echo line

    # Verify using column 0 (tid=0 and tid=16)
    var pass = true
    for i in 0 ..< 8:
      let evenRow = 2 * i
      let oddRow = 2 * i + 1
      let evenExpected = float32(evenRow) * 136.0 + 1360.0
      let oddExpected = float32(oddRow) * 136.0 + 1360.0
      let evenGot = hostOut[0 * 8 + i]   # tid=0
      let oddGot = hostOut[16 * 8 + i]   # tid=16
      if abs(evenGot - evenExpected) > 1.0:
        echo &"  FAIL: y[{evenRow}] = {evenGot}, expected {evenExpected}"
        pass = false
      if abs(oddGot - oddExpected) > 1.0:
        echo &"  FAIL: y[{oddRow}] = {oddGot}, expected {oddExpected}"
        pass = false
    if pass:
      echo "  PASS: All 16 GEMV outputs match expected (column 0)"

  echo ""
  echo "=== Done ==="

main()
