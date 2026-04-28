# hippo_leap

- OpenAI-compatible LLM inference server in Nim.
- Uses hippo for GPU compute, mummy for HTTP serving, openai_leap for API types.

## Project layout

- `src/hippo_leap.nim` — main binary entry point (CLI dispatch)
- `src/hippo_leap/` — library modules (common, config, server, handlers)
- `src/tools/` — standalone CLI tool binaries (health_check)
- `tests/test_*.nim` — unit tests
- `tests/integration_*.nim` — integration tests
- `tests/e2e_*.nim` — end-to-end tests
- `tests/helpers.nim` — shared test utilities
- `docs/` — design docs, benchmarks, future optimization ideas

## Dependencies

- Nim >= 2.0.0
- hippo — GPU library for CUDA/HIP operations (monofuel/hippo)
- mummy — HTTP server
- openai_leap — OpenAI-compatible API types and client (monofuel/openai_leap)
- jsony — JSON serialization
- curly — HTTP client
- ws — WebSocket support

All dependencies are pinned in `nimby.lock` with exact versions and commit hashes.

## Nimby

This project uses nimby for dependency management instead of nimble.

- `nimby.lock` pins every dependency to a specific git commit
- Run `nimby sync -g nimby.lock` to clone/update all dependencies and generate `nim.cfg`
- The generated `nim.cfg` contains `--path:` entries pointing to `~/.nimby/pkgs/<name>/src`
- After syncing, `nim c` and `nim r` will find all dependencies automatically
- To add or update a dependency, edit `nimby.lock` and re-run sync

## Build

- `make build` — compile the main `hippo_leap` binary
- `make tools` — compile CLI tools (health_check)
- Binary output lands in the project root (gitignored)

## Tests

- `make test` — run all unit tests (`tests/test_*.nim`) in parallel
- `make integration-test` — run integration tests (`tests/integration_*.nim`) in parallel
- `make e2e-test` — run end-to-end tests (`tests/e2e_*.nim`) sequentially
- Individual test files can be run with `nim r tests/test_*.nim`

## Models

- GGUF models are stored on the NFS share at `/mnt/steel-chest/LLM/lmstudio/models/`
- `TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf` — benchmark reference model from tinylama
- `lmstudio-community/Llama-3.2-1B-Instruct-GGUF/` — small test models (Q4_K_M, Q8_0)

## Reference repos

- `../scriptorium/` — nimby.lock structure, Makefile patterns, test organization
- `../tinylama` — reference performant LLM inference implementation using hippo
- `../openai_leap/` — OpenAI API types and client
- `../andrewlytics` — mummy HTTP server examples, streaming workarounds
- `../hippo` — Nim HIP/CUDA GPU library (maintained by monofuel)

## Nim best practices

**Prefer letting errors bubble up naturally** - Nim's stack traces are excellent for debugging:

Default approach - let operations fail with full context:
```nim
# Simple and clear - if writeFile fails, we get a full stack trace
writeFile(filepath, content)

# Database operations - let them fail with complete error information
db.exec(sql"INSERT INTO users (name) VALUES (?)", username)
```

For validation and early returns, check conditions explicitly:
```nim
# Check preconditions and exit early with clear messages
if not fileExists(parentDir):
  error "Parent directory does not exist"
  quit(1)

if username.len == 0:
  error "Username cannot be empty"
  quit(1)

# Now proceed with the operation
writeFile(filepath, content)
```

This approach ensures full stack traces in CI environments and makes debugging straightforward.

### Nim Imports

- std imports should be first, then libraries, and then local imports
- use [] brackets to group when possible
- split imports on newlines
for example,
```
import
  std/[strformat, strutils],
  debby/[pools, postgres],
  ./[models, logs, llm] 
```

### Nim Procs

- do not put comments before functions! comments go inside functions.
- every proc should have a nimdoc comment
- nimdoc comments start with ##
- nimdoc comments should be complete sentences followed by punctuation
for example,
```
proc sumOfMultiples(limit: int): int =
  ## Calculate the sum of all multiples of 3 or 5 below the limit.
  var total = 0
  for i in 1..<limit:
    if i mod 3 == 0 or i mod 5 == 0:
      total += i
  return total
```

### Nim Properties

- if an object property is the same name as a nim keyword, you must wrap it in backticks
```
  DeleteModelResponse* = ref object
    id*: string
    `object`*: string
    deleted*: bool
```

### Variables

- please group const, let, and var variables together.
- please prefer const over let, and let over var.
- please use capitalized camelCase for consts
- use regular camelcase for var and let
- do not place 'magic variables' in the code, instead make them a const and pull them up to the top of the file
- for example:

```
const
  Version = "0.1.0"
  Model = "llama3.2:1b"
let
  embeddingModel = "nomic-embed-text"
```

## Programming

- Don't use try/catch unless you have a very, very good reason to be handling the error at this level.
- never mask errors with catch: discard
- it's OK to allow errors to bubble up. we want things to be easy to debug and fail fast.
- returning in the middle of files is confusing, avoid doing it.
  - early returns at the start of the file is ok.
- try to make things as idempotent as possible. if a job runs every day, we should make sure it can be robust.
- never use booleans for 'success' or 'error'. If a function was successful, return nothing and do not throw an error. if a function failed, throw an error.

### Comments

- functions should have doc comments
- however code should otherwise not need comments. functions should be named properly and the code should be readable.
- comments may be ok for 'spooky at a distance' things in rare cases.
- comments should be complete sentences that are followed with a period.
