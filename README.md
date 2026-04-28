# HippoLeap


- hippo_leap is an openai-compatible LLM inference server purely in Nim.
- the name is a play on my `monofuel/hippo` gpu library for cuda/hip in Nim, along with my openai_leap library (which was originally a play on llama_leap when I initially had an ollama-compatible API client)


- dependencies
  - hippo for GPU usage
  - mummy for API server
  - treeform/whisky for websockets (or treeform/ws if we have issues)
  - openai_leap to borrow the API structures, and for API integration tests

- models
  - GGUF models are stored at `/mnt/steel-chest/LLM/lmstudio/models/`
  - `TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf` — benchmark reference model from tinylama
  - `lmstudio-community/Llama-3.2-1B-Instruct-GGUF/` — small test models (Q4_K_M, Q8_0)

- reference repos
  - refer to ../scriptorium/ for the nimby.lock structure and makefile structure. we want unit, integration and e2e tests.
  - refer to ../tinylama for a reference initial performant implementation of llm inference using hippo
  - refer to ../openai_leap/ for the API structure
  - we can refer to ../andrewlytics, which has examples of providing an LLM gateway API (note: we want to be using mummy, but mummy does not support streaming, I think we had an example of how to do this properly on andrewlytics)
  - refer to ../hippo for our nim hip library
    - I maintain the library, we can easily add more cpp headers for more features or intrinsics.

- I plan to have this project adopted by scriptorium for better automation.
  - probably have a docker-compose to containerize it, but also pass the GPU through?
