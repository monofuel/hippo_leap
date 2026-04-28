const
  Version* = "0.1.0"
  DefaultPort* = 8080
  DefaultAddress* = "0.0.0.0"
  DefaultModel* = "stub"

type
  HippoLeapError* = object of CatchableError
