.PHONY: test integration-test e2e-test build build-megakernel build-naive tools bench bench-gpu

nim.cfg: nimby.lock
	nimby sync -g nimby.lock

# Shared flags for all GPU builds
NIM_COMMON_FLAGS = --cc:hipcc -d:release -d:useMalloc -d:zippyNoSimd -d:HippoRuntime:HIP

# --- Build targets ---

# Megakernel backend (fast, compile-time specialized per machine+model)
# This is the primary backend for benchmarking and inference.
build-megakernel: nim.cfg
	nim cpp $(NIM_COMMON_FLAGS) \
		-d:backendMegakernel -d:useIndividualLaunches \
		-d:targetMachine=azem -d:targetModel=tinyllama_q2k \
		-o:hippo_leap src/hippo_leap.nim

# Naive backend (general purpose, no compile-time specialization)
build-naive: nim.cfg
	nim cpp $(NIM_COMMON_FLAGS) \
		-d:backendNaive \
		-o:hippo_leap_naive src/hippo_leap.nim

# Default build is megakernel
build: build-megakernel

tools: nim.cfg
	nim c -o:health_check src/tools/health_check.nim

# --- Benchmark targets ---

build-graph: nim.cfg
	nim cpp $(NIM_COMMON_FLAGS) \
		-d:backendMegakernel -d:useIndividualLaunches -d:useGraphCapture \
		-d:targetMachine=azem -d:targetModel=tinyllama_q2k \
		-o:hippo_leap src/hippo_leap.nim

bench: build-megakernel
	./hippo_leap bench -m /mnt/steel-chest/LLM/lmstudio/models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf

bench-gpu: nim.cfg
	nim cpp $(NIM_COMMON_FLAGS) \
		-d:backendMegakernel -d:useIndividualLaunches \
		-d:targetMachine=azem -d:targetModel=tinyllama_q2k \
		-r tests/bench_gpu.nim

NIM_TEST_FLAGS ?= --hints:off --warnings:off

# CPU-only unit tests (no GPU needed)
test: nim.cfg
	@files=$$(ls tests/test_*.nim 2>/dev/null); \
	if [ -z "$$files" ]; then \
		echo "No unit tests found in tests/test_*.nim"; \
		exit 0; \
	fi; \
	fail=0; \
	pids=""; \
	for f in $$files; do \
		( nim r $(NIM_TEST_FLAGS) "$$f" 2>&1 | sed "s|^|[$$f] |" ) & \
		pids="$$pids $$!"; \
	done; \
	for pid in $$pids; do \
		wait $$pid || fail=1; \
	done; \
	exit $$fail

# GPU integration tests (requires nim cpp + hipcc)
integration-test: nim.cfg
	@export TMPDIR=/dev/shm; \
	files=$$(ls tests/integration_*.nim 2>/dev/null); \
	if [ -z "$$files" ]; then \
		echo "No integration tests found in tests/integration_*.nim"; \
		exit 0; \
	fi; \
	fail=0; \
	pids=""; \
	for f in $$files; do \
		( nim cpp $(NIM_GPU_FLAGS) $(NIM_TEST_FLAGS) --run "$$f" 2>&1 | sed "s|^|[$$f] |" ) & \
		pids="$$pids $$!"; \
	done; \
	for pid in $$pids; do \
		wait $$pid || fail=1; \
	done; \
	exit $$fail

e2e-test: nim.cfg
	@found=0; \
	for f in tests/e2e_*.nim; do \
		[ -e "$$f" ] || continue; \
		found=1; \
		echo "--- $$f ---"; \
		nim r $(NIM_TEST_FLAGS) "$$f" || exit 1; \
	done; \
	if [ $$found -eq 0 ]; then \
		echo "No e2e tests found in tests/e2e_*.nim"; \
	fi
