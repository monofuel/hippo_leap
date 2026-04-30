## Compile-time machine and model configs for the megakernel backend.
## Selected via -d:targetMachine=azem -d:targetModel=tinyllama_q2k.

type
  GpuSpec* = object
    arch*: string
    cuCount*, warpSize*, ldsBytes*: int
    vramBandwidthGBs*: int

  MachineConfig* = object
    hostname*: string
    gpu*: GpuSpec

  TransformerConfig* = object
    nEmb*, nHead*, nHeadKv*, headDim*: int
    ffnDim*, nLayers*, nVocab*, ropeDim*: int
    ropeTheta*: float64
    rmsEps*: float32

# --- Machine configs ---

const Azem* = MachineConfig(
  hostname: "azem",
  gpu: GpuSpec(arch: "gfx1151", cuCount: 16, warpSize: 32,
               ldsBytes: 65536, vramBandwidthGBs: 256))

const HighSteel* = MachineConfig(
  hostname: "high-steel",
  gpu: GpuSpec(arch: "gfx1100", cuCount: 96, warpSize: 32,
               ldsBytes: 65536, vramBandwidthGBs: 960))

# --- Model configs ---

const TinyLlama_1_1B_Q2K* = TransformerConfig(
  nEmb: 2048, nHead: 32, nHeadKv: 4, headDim: 64,
  ffnDim: 5632, nLayers: 22, nVocab: 32000,
  ropeDim: 64, ropeTheta: 10000.0, rmsEps: 1e-5'f32)

# --- Compile-time selection ---

const TargetMachine* {.strdefine.} = ""
const TargetModel* {.strdefine.} = ""

const Machine* = when TargetMachine == "azem": Azem
                 elif TargetMachine == "high-steel": HighSteel
                 else: {.error: "Unknown target machine: '" & TargetMachine & "'. Use -d:targetMachine=azem".}

const ModelCfg* = when TargetModel == "tinyllama_q2k": TinyLlama_1_1B_Q2K
                  else: {.error: "Unknown target model: '" & TargetModel & "'. Use -d:targetModel=tinyllama_q2k".}

# --- Derived constants ---

const
  QDim* = ModelCfg.nHead * ModelCfg.headDim
  KvDim* = ModelCfg.nHeadKv * ModelCfg.headDim
  HeadsPerKvGroup* = ModelCfg.nHead div ModelCfg.nHeadKv

  NumBlocks* = Machine.gpu.cuCount
  BlockSize* = 256
  WarpSize* = Machine.gpu.warpSize
  WarpsPerBlock* = BlockSize div WarpSize
  TotalWarps* = NumBlocks * WarpsPerBlock

  MaxDim* = max(max(ModelCfg.nEmb, ModelCfg.ffnDim),
                max(ModelCfg.nVocab, QDim))
