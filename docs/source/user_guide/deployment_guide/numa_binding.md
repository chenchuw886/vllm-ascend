# NUMA Binding for A2/A3 (CPU/Memory Affinity)

This guide documents the standard NUMA binding policy and the acceptance checklist for mixed-deploy and large EP scenarios on A2/A3.

## Topology Assumption (A3)
- 4 sockets (CPU0-CPU3), each socket contains 2 NUMA nodes (2 dies).
- 8 NUMA nodes total (NUMA0-7).
- 40 cores per NUMA node (10 clusters, 4 cores per cluster).

## Binding Policy (A3)
1. **Process group layout**
- Each socket hosts 4 scheduling process groups.
- A scheduling process group includes scheduler/worker/rpc_proxy (and related TCP processes).

2. **CPU binding**
- Each group binds to the first NUMA node of its socket.
- Forward threads in the group bind to the second NUMA node of the socket.
- Motor/Servicer and other background processes bind to Socket 2, first NUMA node.

3. **Memory binding**
- Per group, use `interleave` across the socket's two NUMA nodes to balance bandwidth.
- Memory policy is applied at process start or via cgroup cpuset `mems`.

## Standardized Script
Use `tools/ascend_numa_bind.py` to build and visualize a binding plan, and apply CPU binding for running services.

## Generic Topology Script
For non-A3 environments, use the generic topology-aware script that prioritizes NUMA locality and adapts to the host layout.

```bash
tools/ascend_numa_bind_generic.py \
  --group-pattern "scheduler|worker|rpc_proxy|tcp" \
  --other-pattern "Motor|Servicer"
```

```bash
tools/ascend_numa_bind_generic.py \
  --group-pattern "scheduler|worker|rpc_proxy|tcp" \
  --other-pattern "Motor|Servicer" \
  --apply
```

### Dry Run (plan + report)
```bash
tools/ascend_numa_bind.py \
  --group-pattern "scheduler|worker|rpc_proxy|tcp" \
  --other-pattern "Motor|Servicer"
```

### Apply CPU Binding
```bash
tools/ascend_numa_bind.py \
  --group-pattern "scheduler|worker|rpc_proxy|tcp" \
  --other-pattern "Motor|Servicer" \
  --apply
```

### Config File (optional)
```json
{
  "groups_per_socket": 4,
  "others_socket_index": 2,
  "group_patterns": ["scheduler", "worker", "rpc_proxy", "tcp"],
  "other_patterns": ["Motor", "Servicer"],
  "forward_thread_patterns": ["forward", "prefill", "decode"],
  "report": "workflow_visualization/numa_binding_report.md"
}
```

```bash
tools/ascend_numa_bind.py --config configs/numa_bind.json --apply
```

## Acceptance Checklist

### A. Topology and Preconditions
- `lscpu -e=CPU,NODE,SOCKET` shows 8 NUMA nodes and 4 sockets.
- `numactl --hardware` shows memory size per NUMA node.
- CPU and memory isolation is enabled for target processes (no unexpected cgroup cpuset limits).

### B. Binding Verification
- Run the script in dry-run mode and confirm the plan matches the expected socket/NUMA mapping.
- Apply CPU binding and verify with `taskset -cp <pid>`.
- For forward threads, verify `ps -T -p <pid> -o tid,comm` and `taskset -cp <tid>` reflect the secondary NUMA CPU list.
- If using memory interleave, verify `numactl --show --pid <pid>` or `/proc/<pid>/numa_maps` reflects interleave across the two NUMA nodes.

### C. Performance Acceptance
- Baseline without binding: record token latency for a fixed workload.
- After binding: run the same workload and confirm:
  - Token-to-token latency fluctuation <= 1 ms.
  - No obvious CPU jitter on critical worker threads (use `perf top` or `pidstat -t`).
- Keep log/report from `workflow_visualization/numa_binding_report.md` as evidence.

## Notes
- Host-side IRQ binding requires privileges and is not suitable inside containers. If needed, bind IRQs at host level and pin softirq handling CPUs per socket.
- For memory binding on running processes, prefer starting the service with `numactl --interleave=<nodes>` or using a cpuset cgroup with `cpuset.mems`.
