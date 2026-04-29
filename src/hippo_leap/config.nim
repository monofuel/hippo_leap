import
  std/[os, strutils],
  ./common

type
  ServerConfig* = object
    port*: int
    address*: string
    modelPath*: string
    maxTokens*: int
    maxContextLen*: int

proc defaultConfig*(): ServerConfig =
  ## Return the default server configuration.
  result = ServerConfig(
    port: DefaultPort,
    address: DefaultAddress,
    modelPath: "",
    maxTokens: 256,
    maxContextLen: 2048,
  )

proc loadConfig*(): ServerConfig =
  ## Load server configuration from environment variables.
  result = defaultConfig()
  let portEnv = getEnv("HIPPO_LEAP_PORT", "")
  if portEnv.len > 0:
    result.port = parseInt(portEnv)
  let addrEnv = getEnv("HIPPO_LEAP_ADDRESS", "")
  if addrEnv.len > 0:
    result.address = addrEnv
  let modelEnv = getEnv("HIPPO_LEAP_MODEL", "")
  if modelEnv.len > 0:
    result.modelPath = modelEnv
  let maxTokEnv = getEnv("HIPPO_LEAP_MAX_TOKENS", "")
  if maxTokEnv.len > 0:
    result.maxTokens = parseInt(maxTokEnv)
  let maxCtxEnv = getEnv("HIPPO_LEAP_MAX_CONTEXT", "")
  if maxCtxEnv.len > 0:
    result.maxContextLen = parseInt(maxCtxEnv)
