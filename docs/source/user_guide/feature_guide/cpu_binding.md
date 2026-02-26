# CPU Binding

CPU binding pins threads within vLLM Ascend worker processes to specific CPU cores to reduce scheduling jitter, improve cache locality, and keep NPU interrupts on dedicated CPU cores. It is enabled by default on ARM hosts and automatically skipped on x86.

## Overview

The binding logic runs during worker initialization and performs three actions:

1. Build a CPU pool per running NPU (affinity-based or NUMA-balanced).
2. Bind worker threads (`main`, `acl_thread`, `release_thread`) to assigned CPUs.
3. Bind NPU IRQs (`sq_send_trigger_irq` and `cq_update_irq`) to reserved CPUs and optionally migrate memory toward the target NUMA nodes.

The implementation is in [vllm_ascend/cpu_binding.py](../../../vllm_ascend/cpu_binding.py).

## How it works

### CPU pool construction

CPU pools are built per NPU in `build_cpu_pools()`:

- **Affinity mode (default)**: uses `npu-smi info -t topo` affinity, filtered by `Cpus_allowed_list`.
- **NUMA-balanced mode**: used on A3, or when affinity info is missing.

The pool is expanded to the neighbor NUMA node to improve elasticity when all CPUs map to a single NUMA node.

### CPU allocation

For each NPU pool (`pool`):

- `pool[0]` and `pool[1]` are **reserved for IRQ binding**.
- `pool[-2]` is assigned to `acl_thread`.
- `pool[-1]` is assigned to `release_thread`.
- `pool[2:-2]` is assigned to the main process.

This is logged by `print_plan()` during startup.

> Note
> - The allocator currently only checks `len(pool) >= 3`. With a very small pool (e.g., 3 CPUs), `acl_thread` may share a core with IRQ binding. For best isolation, provide at least **4 CPUs per NPU**.

### Thread binding

`bind_threads()` uses `taskset`:

- `main` PID is bound to `pool[2:-2]`.
- `acl_thread` PID is bound to `pool[-2]` and memory migration is triggered.
- `release_thread` PID is bound to `pool[-1]`.

### Memory binding

`bind_memory()` migrates the ACL thread memory to nearby NUMA nodes using `migratepages`:

- Target NUMA is derived from the ACL CPU.
- Migration is skipped if `migratepages` is not available.

### IRQ binding

`bind_npu_irq()` pins NPU IRQs:

- Stops `irqbalance` if running (via `systemctl`).
- Detects `sq_send_trigger_irq` from `/proc/interrupts`.
- Finds the NPU PCI device via `npu-smi info -t board -i` and `/sys/bus/pci/devices/.../msi_irqs`.
- Binds `sq_send_trigger_irq` to `pool[0]` and `cq_update_irq` to `pool[1]` by writing to `/proc/irq/*/smp_affinity`.

If `/proc/irq` is not writable, IRQ binding is skipped.

## Usage

> **Note**: CPU binding is **only enabled on ARM hosts**. On x86 hosts, it is automatically skipped.

CPU binding is enabled by default. To run with explicit configuration:

### Online Inference

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
    --additional_config '{"enable_cpu_binding": true}' \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --trust-remote-code
```

To disable CPU binding:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
    --additional_config '{"enable_cpu_binding": false}' \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --trust-remote-code
```

### Offline Inference

```python
from vllm import LLM, SamplingParams

# Enabled by default
llm = LLM(
    model="Qwen/Qwen2.5-14B-Instruct",
    tensor_parallel_size=2,
    additional_config={"enable_cpu_binding": True},
)

# Or disable it
llm = LLM(
    model="Qwen/Qwen2.5-14B-Instruct",
    tensor_parallel_size=2,
    additional_config={"enable_cpu_binding": False},
)

prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
```

### Configuration

CPU binding is controlled by `additional_config.enable_cpu_binding` (default: `True`):

```python
additional_config = {
    "enable_cpu_binding": True,  # Enable CPU binding (default)
}
```

## Requirements

- **ARM hosts only**: binding is skipped on x86 (`is_arm_cpu()` guard).
- **Tools**: `taskset`, `npu-smi`; `migratepages` is optional.
- **Permissions**: IRQ binding needs write access to `/proc/irq/*/smp_affinity` (typically root).

## Troubleshooting

- **No binding occurred**: check that the host is ARM and `enable_cpu_binding` is true.
- **IRQ binding skipped**: ensure `/proc/irq` is writable and `irqbalance` is not locked down by policy.
- **Memory binding skipped**: install `migratepages` or ignore if not needed.
