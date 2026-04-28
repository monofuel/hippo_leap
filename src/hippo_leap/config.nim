import
  std/[os, strutils],
  ./common

type
  ServerConfig* = object
    port*: int
    address*: string
    modelPath*: string

proc defaultConfig*(): ServerConfig =
  ## Return the default server configuration.
  result = ServerConfig(
    port: DefaultPort,
    address: DefaultAddress,
    modelPath: "",
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
