import
  std/[os],
  hippo_leap/[common, config]

proc testDefaultConfig() =
  ## Verify defaultConfig returns expected default values.
  let cfg = defaultConfig()
  doAssert cfg.port == DefaultPort
  doAssert cfg.address == DefaultAddress
  doAssert cfg.modelPath == ""
  echo "[OK] defaultConfig returns expected defaults"

proc testEnvOverride() =
  ## Verify loadConfig reads environment variables.
  putEnv("HIPPO_LEAP_PORT", "9090")
  putEnv("HIPPO_LEAP_ADDRESS", "127.0.0.1")
  putEnv("HIPPO_LEAP_MODEL", "/tmp/test.gguf")
  let cfg = loadConfig()
  doAssert cfg.port == 9090
  doAssert cfg.address == "127.0.0.1"
  doAssert cfg.modelPath == "/tmp/test.gguf"
  delEnv("HIPPO_LEAP_PORT")
  delEnv("HIPPO_LEAP_ADDRESS")
  delEnv("HIPPO_LEAP_MODEL")
  echo "[OK] loadConfig reads environment variables"

when isMainModule:
  testDefaultConfig()
  testEnvOverride()
