import
  std/[os],
  hippo_leap/[common, config]

proc testDefaultConfig() =
  ## Verify defaultConfig returns expected default values.
  let cfg = defaultConfig()
  doAssert cfg.port == DefaultPort
  doAssert cfg.address == DefaultAddress
  doAssert cfg.modelPath == ""
  doAssert cfg.maxTokens == 256
  doAssert cfg.maxContextLen == 2048
  echo "[OK] defaultConfig returns expected defaults"

proc testEnvOverride() =
  ## Verify loadConfig reads environment variables.
  putEnv("HIPPO_LEAP_PORT", "9090")
  putEnv("HIPPO_LEAP_ADDRESS", "127.0.0.1")
  putEnv("HIPPO_LEAP_MODEL", "/tmp/test.gguf")
  putEnv("HIPPO_LEAP_MAX_TOKENS", "512")
  putEnv("HIPPO_LEAP_MAX_CONTEXT", "4096")
  let cfg = loadConfig()
  doAssert cfg.port == 9090
  doAssert cfg.address == "127.0.0.1"
  doAssert cfg.modelPath == "/tmp/test.gguf"
  doAssert cfg.maxTokens == 512
  doAssert cfg.maxContextLen == 4096
  delEnv("HIPPO_LEAP_PORT")
  delEnv("HIPPO_LEAP_ADDRESS")
  delEnv("HIPPO_LEAP_MODEL")
  delEnv("HIPPO_LEAP_MAX_TOKENS")
  delEnv("HIPPO_LEAP_MAX_CONTEXT")
  echo "[OK] loadConfig reads environment variables"

when isMainModule:
  testDefaultConfig()
  testEnvOverride()
