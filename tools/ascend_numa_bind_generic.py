#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    comm: str
    args: str


@dataclass(frozen=True)
class GroupPlan:
    name: str
    socket: int
    group_index: int
    primary_node: int
    secondary_node: int
    primary_cpus: list[int]
    secondary_cpus: list[int]
    mem_nodes: list[int]


def run_cmd(cmd: list[str]) -> tuple[str, int]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate(timeout=30)
    if proc.returncode != 0 and err:
        out = out + "\n" + err
    return out.strip(), proc.returncode


def compress_cpus(cpus: Iterable[int]) -> str:
    items = sorted(set(int(x) for x in cpus))
    if not items:
        return ""
    ranges: list[str] = []
    start = prev = items[0]
    for val in items[1:]:
        if val == prev + 1:
            prev = val
            continue
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = val
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(ranges)


def chunk_list(items: list[int], chunks: int) -> list[list[int]]:
    if chunks <= 0:
        return [items]
    total = len(items)
    base = total // chunks
    extra = total % chunks
    result: list[list[int]] = []
    start = 0
    for i in range(chunks):
        take = base + (1 if i < extra else 0)
        result.append(items[start:start + take])
        start += take
    return result


def parse_lscpu() -> dict[int, dict[int, list[int]]]:
    # lscpu is stable across distributions and exposes CPU->NUMA->socket mapping.
    out, code = run_cmd(["lscpu", "-e=CPU,NODE,SOCKET"])
    if code != 0:
        raise RuntimeError("failed to run lscpu -e=CPU,NODE,SOCKET")
    socket_map: dict[int, dict[int, list[int]]] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        cpu, node, socket = (int(parts[0]), int(parts[1]), int(parts[2]))
        socket_map.setdefault(socket, {}).setdefault(node, []).append(cpu)
    for socket_nodes in socket_map.values():
        for node, cpus in socket_nodes.items():
            socket_nodes[node] = sorted(cpus)
    return socket_map


def parse_node_distances(node: int) -> list[int]:
    path = f"/sys/devices/system/node/node{node}/distance"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        values = [int(x) for x in f.read().strip().split() if x.isdigit()]
    return values


def pick_secondary_node(primary: int, nodes: list[int]) -> int:
    if len(nodes) <= 1:
        return primary
    distances = parse_node_distances(primary)
    best_node = None
    best_distance = None
    for node in nodes:
        if node == primary:
            continue
        if distances and node < len(distances):
            distance = distances[node]
        else:
            distance = None
        if distance is None:
            # Fallback: prefer next node in order.
            best_node = nodes[(nodes.index(primary) + 1) % len(nodes)]
            break
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_node = node
    return best_node if best_node is not None else primary


def list_processes() -> list[ProcessInfo]:
    out, code = run_cmd(["ps", "-eo", "pid,comm,args"])
    if code != 0:
        raise RuntimeError("failed to run ps -eo pid,comm,args")
    result: list[ProcessInfo] = []
    for line in out.splitlines():
        line = line.rstrip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid = int(parts[0])
        comm = parts[1]
        args = parts[2]
        result.append(ProcessInfo(pid=pid, comm=comm, args=args))
    return result


def match_processes(processes: list[ProcessInfo], patterns: list[str]) -> list[ProcessInfo]:
    if not patterns:
        return []
    regexes = [re.compile(p) for p in patterns]
    matched: list[ProcessInfo] = []
    for proc in processes:
        hay = f"{proc.comm} {proc.args}"
        if any(r.search(hay) for r in regexes):
            matched.append(proc)
    return matched


def list_threads(pid: int) -> list[tuple[int, str]]:
    out, code = run_cmd(["ps", "-T", "-p", str(pid), "-o", "tid,comm"])
    if code != 0:
        return []
    threads: list[tuple[int, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        threads.append((int(parts[0]), parts[1]))
    return threads


def taskset_bind(pid: int, cpus: list[int], all_threads: bool) -> tuple[str, int]:
    cpu_list = compress_cpus(cpus)
    if not cpu_list:
        return "", 0
    if all_threads:
        # Bind all threads for a process (scheduler/worker/rpc_proxy) to keep group locality.
        cmd = ["taskset", "-acp", cpu_list, str(pid)]
    else:
        # Bind a single thread (e.g. forward) to secondary NUMA CPUs.
        cmd = ["taskset", "-cp", cpu_list, str(pid)]
    return run_cmd(cmd)


def numa_show(pid: int) -> str:
    out, _ = run_cmd(["numactl", "--show", "--pid", str(pid)])
    return out


def build_group_plans(
    socket_map: dict[int, dict[int, list[int]]],
    groups_per_socket: int,
) -> list[GroupPlan]:
    plans: list[GroupPlan] = []
    for socket in sorted(socket_map):
        nodes = sorted(socket_map[socket])
        if not nodes:
            continue
        group_count = max(groups_per_socket, 1)
        node_to_group_count: dict[int, int] = {node: 0 for node in nodes}
        for i in range(group_count):
            node = nodes[i % len(nodes)]
            node_to_group_count[node] += 1
        node_chunks: dict[int, list[list[int]]] = {}
        for node in nodes:
            node_chunks[node] = chunk_list(socket_map[socket][node], node_to_group_count[node])

        group_index = 0
        for node in nodes:
            for chunk in node_chunks[node]:
                primary_node = node
                secondary_node = pick_secondary_node(primary_node, nodes)
                secondary_cpus = socket_map[socket][secondary_node]
                if secondary_node == primary_node:
                    secondary_cpus = chunk
                mem_nodes = [primary_node] if secondary_node == primary_node else [primary_node, secondary_node]
                plans.append(
                    GroupPlan(
                        name=f"socket{socket}_group{group_index}",
                        socket=socket,
                        group_index=group_index,
                        primary_node=primary_node,
                        secondary_node=secondary_node,
                        primary_cpus=chunk,
                        secondary_cpus=secondary_cpus,
                        # Memory interleave across two nearest NUMA nodes when available.
                        mem_nodes=mem_nodes,
                    )
                )
                group_index += 1
    return plans


def format_plan(plans: list[GroupPlan]) -> str:
    lines = ["Binding Plan (per group)"]
    for plan in plans:
        lines.append(
            f"- {plan.name}: socket={plan.socket} primary_node={plan.primary_node} "
            f"secondary_node={plan.secondary_node} primary_cpus={compress_cpus(plan.primary_cpus)} "
            f"forward_cpus={compress_cpus(plan.secondary_cpus)} mem_nodes={','.join(map(str, plan.mem_nodes))}"
        )
    return "\n".join(lines)


def assign_processes_to_groups(processes: list[ProcessInfo], groups: list[GroupPlan]) -> dict[int, GroupPlan]:
    assignment: dict[int, GroupPlan] = {}
    if not processes:
        return assignment
    groups_sorted = list(groups)
    # Deterministic mapping by PID keeps repeated runs stable.
    for idx, proc in enumerate(sorted(processes, key=lambda p: p.pid)):
        assignment[proc.pid] = groups_sorted[idx % len(groups_sorted)]
    return assignment


def write_report(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def build_report(
    socket_map: dict[int, dict[int, list[int]]],
    plans: list[GroupPlan],
    group_processes: list[ProcessInfo],
    other_processes: list[ProcessInfo],
    assignments: dict[int, GroupPlan],
    before: dict[int, str],
    after: dict[int, str],
    forward_matches: dict[int, list[int]],
    report_title: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# {report_title}")
    lines.append("")
    lines.append("## Topology")
    for socket in sorted(socket_map):
        lines.append(f"- socket {socket}")
        for node in sorted(socket_map[socket]):
            lines.append(f"- node {node}: cpus {compress_cpus(socket_map[socket][node])}")
    lines.append("")
    lines.append("## Plan")
    lines.append(format_plan(plans))
    lines.append("")
    lines.append("## Process Matches")
    if not group_processes and not other_processes:
        lines.append("- no processes matched")
    if group_processes:
        lines.append("- group processes")
        for proc in group_processes:
            lines.append(f"- pid {proc.pid}: {proc.comm} {proc.args}")
    if other_processes:
        lines.append("- others processes")
        for proc in other_processes:
            lines.append(f"- pid {proc.pid}: {proc.comm} {proc.args}")
    lines.append("")
    lines.append("## Binding Results")
    if not assignments and not other_processes:
        lines.append("- no binding actions")
    for pid, plan in assignments.items():
        lines.append(
            f"- pid {pid} -> {plan.name} primary={compress_cpus(plan.primary_cpus)} "
            f"forward={compress_cpus(plan.secondary_cpus)} mem_nodes={','.join(map(str, plan.mem_nodes))}"
        )
        tids = forward_matches.get(pid, [])
        if tids:
            lines.append(f"- forward threads: {','.join(map(str, tids))}")
        if pid in before:
            lines.append(f"- before: {before[pid]}")
        if pid in after:
            lines.append(f"- after: {after[pid]}")
    for proc in other_processes:
        if proc.pid not in assignments and proc.pid in after:
            lines.append(f"- pid {proc.pid} (others) -> {after[proc.pid]}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NUMA binding helper (generic topology)")
    parser.add_argument("--groups-per-socket", type=int, default=0)
    parser.add_argument("--others-socket-index", type=int, default=0)
    parser.add_argument("--group-pattern", action="append", default=[])
    parser.add_argument("--other-pattern", action="append", default=[])
    parser.add_argument("--forward-thread-pattern", action="append", default=["forward", "prefill", "decode"])
    parser.add_argument("--group-pid", action="append", type=int, default=[])
    parser.add_argument("--other-pid", action="append", type=int, default=[])
    parser.add_argument("--report", default="workflow_visualization/numa_binding_report.md")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--config")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> None:
    if not args.config:
        return
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    args.groups_per_socket = int(cfg.get("groups_per_socket", args.groups_per_socket))
    args.others_socket_index = int(cfg.get("others_socket_index", args.others_socket_index))
    args.group_pattern = list(cfg.get("group_patterns", args.group_pattern))
    args.other_pattern = list(cfg.get("other_patterns", args.other_pattern))
    args.forward_thread_pattern = list(cfg.get("forward_thread_patterns", args.forward_thread_pattern))
    args.group_pid = list(cfg.get("group_pids", args.group_pid))
    args.other_pid = list(cfg.get("other_pids", args.other_pid))
    args.report = cfg.get("report", args.report)


def main() -> int:
    args = parse_args()
    load_config(args)

    socket_map = parse_lscpu()

    processes = list_processes()
    group_processes = match_processes(processes, args.group_pattern)
    other_processes = match_processes(processes, args.other_pattern)

    if args.group_pid:
        pid_set = {p.pid for p in group_processes}
        pid_set.update(args.group_pid)
        group_processes = [p for p in processes if p.pid in pid_set]

    if args.other_pid:
        pid_set = {p.pid for p in other_processes}
        pid_set.update(args.other_pid)
        other_processes = [p for p in processes if p.pid in pid_set]

    other_pids = {p.pid for p in other_processes}
    group_processes = [p for p in group_processes if p.pid not in other_pids]

    if args.groups_per_socket <= 0:
        num_sockets = max(len(socket_map), 1)
        args.groups_per_socket = max((len(group_processes) + num_sockets - 1) // num_sockets, 1)

    plans = build_group_plans(socket_map, args.groups_per_socket)
    assignments = assign_processes_to_groups(group_processes, plans)

    forward_regexes = [re.compile(p) for p in args.forward_thread_pattern]

    before: dict[int, str] = {}
    after: dict[int, str] = {}
    forward_matches: dict[int, list[int]] = {}

    for proc in group_processes + other_processes:
        if proc.pid in before:
            continue
        out, _ = run_cmd(["taskset", "-cp", str(proc.pid)])
        before[proc.pid] = out.replace("\n", " ").strip()

    if args.apply:
        for proc in group_processes:
            plan = assignments.get(proc.pid)
            if not plan:
                continue
            taskset_bind(proc.pid, plan.primary_cpus, all_threads=True)
            tids = []
            for tid, name in list_threads(proc.pid):
                if any(r.search(name) for r in forward_regexes):
                    taskset_bind(tid, plan.secondary_cpus, all_threads=False)
                    tids.append(tid)
            forward_matches[proc.pid] = tids
        if other_processes:
            socket_nodes = socket_map.get(args.others_socket_index)
            if socket_nodes:
                first_node = sorted(socket_nodes)[0]
                other_cpus = socket_nodes[first_node]
                for proc in other_processes:
                    taskset_bind(proc.pid, other_cpus, all_threads=True)
        time.sleep(0.2)

    for proc in group_processes + other_processes:
        if proc.pid in after:
            continue
        out, _ = run_cmd(["taskset", "-cp", str(proc.pid)])
        after[proc.pid] = out.replace("\n", " ").strip()

    report = build_report(
        socket_map=socket_map,
        plans=plans,
        group_processes=group_processes,
        other_processes=other_processes,
        assignments=assignments,
        before=before,
        after=after,
        forward_matches=forward_matches,
        report_title="NUMA Binding Report (Generic)",
    )
    write_report(args.report, report)

    print(report)
    print(f"Report written to {args.report}")
    if not args.apply:
        print("Dry run only. Use --apply to enforce taskset bindings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
