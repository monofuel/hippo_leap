# GPU Fault Diagnosis and Recovery (amdgpu / gfx1151)

## Symptoms

When the GPU enters a bad state, you'll see some or all of:

- **Degraded performance**: inference runs slower than expected or produces incorrect results
- **Kernel page faults in dmesg**:
  ```
  amdgpu: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:32770)
  amdgpu: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031
  amdgpu:   Faulty UTCL2 client ID: TCP (0x8)
  amdgpu:   PERMISSION_FAULTS: 0x3
  ```
- **Multiple faults at sequential addresses** (0x...c02000, 0x...c04000, etc.) — the GPU is walking through a freed page table

## Root Cause

A GPU process crashes or is killed while it has active HIP allocations. The amdgpu driver
doesn't always cleanly recover the GPU's L2 page table (UTCL2) state. Subsequent kernel
launches still work but run degraded — possibly because the command processor stalls
recovering between dispatches, or because VMID recycling hits dirty entries.

The `PERMISSION_FAULTS: 0x3` indicates the GPU tried to read memory that was already unmapped.
`TCP (0x8)` = Texture Cache Pipe, meaning shader loads (our GEMV weight reads) are hitting
invalid page table entries.

## Diagnosis

```bash
# Check for GPU page faults
npsh azem sudo dmesg --level=err,warn | grep -i "page fault\|PERMISSION_FAULTS\|amdgpu"

# Check if any GPU processes are still running
npsh azem fuser -v /dev/dri/renderD128 2>&1

# Check GPU reset status (may not exist on all kernels)
npsh azem cat /sys/kernel/debug/dri/0/amdgpu_gpu_recover 2>/dev/null

# Check uptime to know when last reboot was
npsh azem uptime
```

## Recovery

### Reboot (reliable)

The only reliable fix is rebooting the node. The amdgpu driver reinitializes all GPU state
at boot, clearing the corrupted page tables.

After reboot, re-pin GPU clocks before benchmarking:
```bash
npsh azem bash -c 'echo "high" | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level'
```

### GPU reset without reboot (unreliable)

Some kernels expose `amdgpu_gpu_recover` in debugfs, but it's not available on all
configurations and may not fully clear UTCL2 state. On azem's kernel (6.12.x), this
file does not exist.

## Prevention

- **Clean shutdown of GPU processes**: avoid `kill -9` on processes with active HIP
  allocations. Let them exit normally or use `hipDeviceReset()` before exit.
- **Signal handlers**: consider adding a cleanup handler that calls `hipDeviceReset()`
  on SIGSEGV/SIGABRT so that a crash at least attempts to release GPU resources.
- **Integration tests**: the `integration_inf` test binary is the most common culprit.
  If it crashes mid-kernel, the GPU page tables are left dirty.

## Incident Log

### 2026-05-01: integration_inf page fault storm

- **Trigger**: `integration_inf` (pid 1359506) crashed with active GPU allocations
- **Effect**: ~10 sequential page faults at 2KB-spaced addresses in the 0x7f8cc1c00000 range
- **Impact**: GPU page table state corrupted, subsequent launches unreliable
- **Resolution**: node reboot required
- **Root cause**: likely a bug in the test causing an out-of-bounds GPU memory access, which
  cascaded into a page fault storm that corrupted the driver's VMID/page table state
