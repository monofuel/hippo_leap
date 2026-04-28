import
  std/[os, strutils]

export os, strutils

const
  TestServerPort* = 18080
  TestServerAddress* = "127.0.0.1"
  TestServerUrl* = "http://" & TestServerAddress & ":" & $TestServerPort
