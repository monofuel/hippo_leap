import
  hippo_leap/common

proc testVersion() =
  ## Verify version constant is set.
  doAssert Version == "0.1.0"
  echo "[OK] Version constant is correct"

proc testDefaults() =
  ## Verify default constants.
  doAssert DefaultPort == 8080
  doAssert DefaultAddress == "0.0.0.0"
  doAssert DefaultModel == "stub"
  echo "[OK] Default constants are correct"

when isMainModule:
  testVersion()
  testDefaults()
