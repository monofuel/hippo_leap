# HippoLeap

OpenAI-compatible LLM inference server in Nim with GPU acceleration via HIP.

The name is a play on [monofuel/hippo](https://github.com/monofuel/hippo) (GPU library for CUDA/HIP in Nim) and [openai_leap](https://github.com/monofuel/openai_leap) (originally a play on llama_leap).

## Quick start

```bash
# Install dependencies
nimby sync -g nimby.lock

# Build (requires hipcc on an AMD GPU machine)
make build

# Run the server
HIPPO_LEAP_MODEL=/path/to/model.gguf ./hippo_leap serve

# Test it
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tinylama","messages":[{"role":"user","content":[{"type":"text","text":"Write one sentence about Nim."}]}],"max_tokens":32}'
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HIPPO_LEAP_MODEL` | (none) | Path to GGUF model file (required for inference) |
| `HIPPO_LEAP_PORT` | 8080 | Server port |
| `HIPPO_LEAP_ADDRESS` | 0.0.0.0 | Bind address |
| `HIPPO_LEAP_MAX_TOKENS` | 256 | Default max tokens per response |
| `HIPPO_LEAP_MAX_CONTEXT` | 2048 | Max context length |

## Building

Requires AMD GPU with ROCm/HIP toolchain (`hipcc`). Tested on gfx1100 (7900 XTX) and gfx1151 (Strix Halo).

```bash
# GPU build (default)
make build

# Run CPU-only unit tests (no GPU needed)
make test

# Run GPU integration tests (verifies golden output against reference)
make integration-test

# Run end-to-end server tests (starts server, sends HTTP requests)
make e2e-test
```

The build uses `nim cpp --cc:hipcc -d:useMalloc -d:backendNaive -d:zippyNoSimd`. The `-d:zippyNoSimd` flag works around hipcc trying to compile x86 SSE intrinsics for the GPU target (see `docs/split-gpu-build.md` for a future fix).

## Architecture

```
src/hippo_leap/
  # Shared (no GPU, testable with nim c):
  gguf_loader.nim      GGUF file format parser (MemFile-based)
  tensor.nim           CPU float32 tensor ops
  quant.nim            Q2_K/Q3_K/Q6_K dequantization
  model.nim            HParams, Model, lazy tensor loading
  tokenizer.nim        SentencePiece tokenize/detokenize
  sampling.nim         argmax, prompt encoding

  # GPU backend (nim cpp --cc:hipcc):
  backend_types.nim    GpuTensor, KvCache, GpuContext types
  backends/
    naive.nim          All GPU kernels (quantized GEMV, attention, RoPE, etc.)

  # Server:
  inference.nim        Compile-time backend dispatch, generate() loop
  handlers.nim         OpenAI-compatible HTTP handlers
  server.nim           Mummy HTTP server setup
  config.nim           Environment-based configuration
```

Backends are selected at compile time with `-d:backendNaive`. Each backend exports four procs: `loadModelBackend`, `unloadModelBackend`, `forwardPrefill`, `forwardDecode`. No vtable, full inlining.

## Dependencies

- [hippo](https://github.com/monofuel/hippo) - Nim GPU library for CUDA/HIP
- [mummy](https://github.com/guzba/mummy) - HTTP server
- [openai_leap](https://github.com/monofuel/openai_leap) - OpenAI API type definitions
- [jsony](https://github.com/treeform/jsony) - JSON serialization
- [curly](https://github.com/guzba/curly) - HTTP client (used in tests/tools)

## Models

GGUF models are stored at `/mnt/steel-chest/LLM/lmstudio/models/`:
- `TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf` - benchmark reference model
- `lmstudio-community/Llama-3.2-1B-Instruct-GGUF/` - small test models (Q4_K_M, Q8_0)

Currently supports Q2_K, Q3_K, and Q6_K quantization formats.

## Reference repos

- `../tinylama` - reference performant LLM inference using hippo (371 tok/s on 7900 XTX)
- `../hippo` - Nim HIP/CUDA library (maintained by monofuel)
- `../openai_leap` - OpenAI API client/types
- `../scriptorium` - nimby.lock and Makefile conventions
