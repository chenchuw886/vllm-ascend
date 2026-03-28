# vLLM-Ascend CPU 绑核性能优化案例分析

## 1. 问题背景

### 1.1 现象描述

在 vLLM-Ascend 8 卡 TP 推理场景下，观察到以下性能异常：

| 配置 | 首次性能 | 稳态表现 | 性能上限 |
|------|----------|----------|----------|
| **不绑核** | 差 | 越跑越好，逐渐稳定 | 中等 |
| **绑核（原方案）** | 差 | 不收敛，持续波动 | **最差** |

**核心矛盾**：绑核理论上应该提升 NUMA locality，但实测反而导致性能上限下降。

### 1.2 原始绑核方案

```python
# worker.py - _init_device()
def _init_device(self):
    torch.npu.set_device(device)
    self._init_worker_distributed_environment()  # HCCL 初始化
    
    # 绑核在这里执行 ← 问题根源
    if get_ascend_config().enable_cpu_binding:
        bind_cpus(self.local_rank)
    
    return device
```

绑核操作包含：
1. `taskset -acp`：将进程所有线程绑定到目标 NUMA 的 CPU
2. `migratepages`：迁移已有内存页到目标 NUMA

---

## 2. 问题分析

### 2.1 内存分布诊断

通过 `numastat -p` 采集各 Worker 进程的内存分布：

```
=== Worker_TP0 (Node 6) ===  本地: 2294 MB  远程: 718 MB  (本地化率 76%)
=== Worker_TP1 (Node 7) ===  本地: 2820 MB  远程: 239 MB  (本地化率 92%)
...
共性远程页:
- Node 2: ~140 MB (所有 Worker)  ← PyTorch dispatch table
- Node 4: ~77 MB (所有 Worker)   ← 框架页
- Node 7: ~480 MB (所有 Worker)  ← EngineCore fork 继承页
```

### 2.2 启动时序分析

绘制完整的内存分配时序：

```
T0: _init_device()
    ├─ torch.npu.set_device()      ← ACL context
    ├─ _init_distributed()         ← HCCL 缓冲
    └─ bind_cpus()                 ← ⚠️ 绑核点（太早）

T1: NPUModelRunner.__init__()
    └─ NPUInputBatch()             ← pinned buffers（绑核后才分配）

T2: load_model()
    └─ 权重张量                     ← （绑核后才分配）

T3: profile_run()
    └─ attention metadata          ← （绑核后才分配）

T4: capture_model()
    └─ ACLGraphEntry, graph pool   ← （绑核后才分配）

T5: _warm_up_atb()
    └─ ATB 内部缓冲                ← （绑核后才分配）
```

### 2.3 根因定位

#### 根因 1：`migratepages` 时机太早

T0 时执行 `migratepages`，但真正的热页（T1-T5）都还不存在：

| T0 时 migratepages 能迁的页 | 推理 steady-state 热页 |
|---------------------------|----------------------|
| Python runtime pages | ACLGraphEntry (T4) |
| PyTorch framework pages | NPUInputBatch pinned (T1) |
| ACL context init pages | attention metadata (T3) |
| HCCL ring buffer headers | graph pool buffers (T4) |

**结论**：迁移的都是冷页，热页根本不存在。

#### 根因 2：`taskset -acp` 抑制了 AutoNUMA

Linux 内核的 AutoNUMA (`numa_balancing`) 会自动迁移热页：

- **不绑核**：AutoNUMA 持续工作，逐步优化那 ~700MB 远程页 → "越跑越好"
- **早期绑核**：`taskset` 信号告诉内核"我已手动优化"，AutoNUMA 降低积极性 → 远程页永远无法迁移

#### 根因 3：共享页无法被 migratepages 迁移

那 ~700MB 远程页来自 fork：
```
api_server (Node 7) → fork → EngineCore → fork → Workers
                              ↓
                    COW 共享页，migratepages 无法迁移
```

---

## 3. 解决方案

### 3.1 核心思路

**延迟绑核时机**：将 `bind_cpus()` 移到所有内存分配完成之后。

### 3.2 代码修改

```python
# worker.py - 修改前
def _init_device(self):
    ...
    if get_ascend_config().enable_cpu_binding:
        bind_cpus(self.local_rank)  # ← 太早
    return device

# worker.py - 修改后
def _init_device(self):
    ...
    # NOTE: CPU binding is deferred to compile_or_warm_up_model()
    return device

def compile_or_warm_up_model(self) -> float:
    # 1. warmup
    for size in sorted(warmup_sizes, reverse=True):
        self.model_runner._dummy_run(size)
    
    # 2. graph capture
    if not self.model_config.enforce_eager:
        self.model_runner.capture_model()
    
    # 3. ATB warmup
    if get_ascend_device_type() != AscendDeviceType.A5:
        self._warm_up_atb()
    
    # 4. CPU binding - 所有热页已分配完毕 ← 最佳位置
    if get_ascend_config().enable_cpu_binding:
        try:
            bind_cpus(self.local_rank)
        except Exception as e:
            logger.warning(f"Bind cpus failed: {e}")
    
    set_random_seed(self.model_config.seed)
    return self.vllm_config.compilation_config.compilation_time
```

### 3.3 时序对比

```
修改前:                              修改后:
─────────────────────────────────    ─────────────────────────────────
_init_device()                       _init_device()
  └─ bind_cpus() ← T0 太早             └─ (无绑核)

NPUModelRunner()                     NPUModelRunner()
  └─ NPUInputBatch ← 绑核后             └─ NPUInputBatch ← first-touch ✓

load_model() ← 绑核后                load_model() ← first-touch ✓

profile_run() ← 绑核后               profile_run() ← first-touch ✓

capture_model() ← 绑核后             capture_model() ← first-touch ✓

_warm_up_atb() ← 绑核后              _warm_up_atb() ← first-touch ✓
                                       └─ bind_cpus() ← 最后执行 ✓
```

---

## 4. 验证结果

### 4.1 内存分布改善

| Worker | 修改前本地化率 | 修改后本地化率 | 改善 |
|--------|---------------|---------------|------|
| TP0 (Node 6) | 76% | 76% | - |
| **TP1 (Node 7)** | 92% | **99.997%** | ✅ |
| TP2 (Node 4) | 79% | 76% | - |
| ... | ... | ... | ... |

**关键变化**：
- Node 2 的 ~140MB 远程页消失
- Node 4 的 ~77MB 减少到 ~10MB
- TP1 几乎达到完美本地化

### 4.2 性能表现

| 指标 | 修改前 | 修改后 | 不绑核 |
|------|--------|--------|--------|
| 首次性能 | 差 | 略差 | 差 |
| 稳态收敛 | ❌ 不收敛 | ✅ 第2次稳定 | ✅ 慢收敛 |
| **稳态性能** | **最差** | **最好** | 中等 |

---

## 5. 技术原理总结

### 5.1 Linux NUMA 内存分配策略

1. **First-Touch Policy**：页面首次访问时，分配在当前 CPU 所在 NUMA
2. **AutoNUMA (numa_balancing)**：内核自动检测并迁移频繁访问的远程页
3. **taskset 的副作用**：向内核发出"已手动优化"信号，可能降低 AutoNUMA 积极性

### 5.2 Fork 与 COW 共享页

```
Parent Process (EngineCore)
    │
    │  fork()
    ▼
Child Process (Worker)
    │
    └─ 共享父进程的所有页面 (COW)
       │
       └─ migratepages 无法迁移共享页
          只有写入触发 COW 后，新页才能正确分配
```

### 5.3 绑核时机选择原则

| 时机 | 效果 |
|------|------|
| 太早（初始化前） | 热页通过 first-touch 在正确位置，但 AutoNUMA 被抑制，共享页无法优化 |
| 太晚（推理中） | 引入运行时抖动 |
| **最佳（所有分配后、推理前）** | first-touch 保证新页正确，migratepages 迁移存量远程页，不影响推理 |

---

## 6. vLLM-Ascend 启动链路全景图

```
api_server (main process)
    │
    ├─ EngineCore.__init__()
    │       │
    │       ├─ Executor._init_executor()
    │       │       └─ WorkerProc ─── fork ──→ Worker 进程
    │       │                                      │
    │       │   ┌──────────────────────────────────┴───────────────────────┐
    │       │   │  [1] init_worker()           Python 对象 (COW 共享)       │
    │       │   │  [2] init_device()           ACL/HCCL/NPUInputBatch       │
    │       │   │  [3] load_model()            权重张量                     │
    │       │   │  [READY]                                                  │
    │       │   └──────────────────────────────────────────────────────────┘
    │       │
    │       └─ _post_init_executor()
    │
    ├─ _initialize_kv_caches()
    │       ├─ [4] determine_available_memory()    profile_run
    │       ├─ [5] initialize_from_config()        KV cache 元数据
    │       └─ [6] compile_or_warm_up_model()      warmup + capture
    │                   │
    │                   ├─ _dummy_run()
    │                   ├─ capture_model()
    │                   ├─ _warm_up_atb()
    │                   └─ bind_cpus()  ← ✅ 最佳位置
    │
    └─ [Start serving requests]
```

---

## 7. 经验教训

### 7.1 性能优化方法论

1. **先量化，后优化**：用 `numastat -p` 等工具量化内存分布
2. **理解时序**：绘制完整的初始化链路，找出所有内存分配点
3. **理解内核行为**：AutoNUMA、first-touch、COW 等机制
4. **A/B 对比验证**：修改前后对比内存分布和性能

### 7.2 常见陷阱

| 陷阱 | 表现 | 解决 |
|------|------|------|
| 绑核时机太早 | 抑制 AutoNUMA，共享页无法优化 | 延迟到分配完成后 |
| 依赖 migratepages | COW 共享页无法迁移 | 依赖 first-touch + AutoNUMA |
| taskset -acp 过度约束 | 所有线程被绑定，包括 HCCL 通信线程 | 考虑只绑定关键线程 |

### 7.3 调试工具

```bash
# NUMA 内存分布
numastat -p <pid>

# AutoNUMA 状态
cat /proc/sys/kernel/numa_balancing

# 进程 CPU 亲和性
taskset -p <pid>

# 内存页迁移统计
cat /proc/vmstat | grep numa

# 查看进程内存映射
cat /proc/<pid>/numa_maps
```

---

## 8. 后续优化方向

1. **消除 ~700MB COW 共享页**：
   - 使用 `spawn` 替代 `fork` 启动 Worker
   - 或在绑核后主动触发 COW 写入

2. **精细化绑核**：
   - 只绑定 `acl_thread`、`release_thread`，不绑定 HCCL 通信线程
   - 图模式下只绑 IRQ，不绑进程

3. **动态调整**：
   - 根据 `enforce_eager` 模式选择不同绑核策略
   - eager 模式全绑，graph 模式轻量绑定

---

## 附录：关键代码位置

| 文件 | 功能 |
|------|------|
| `vllm_ascend/cpu_binding.py` | 绑核实现 (`bind_cpus`, `bind_threads`, `bind_memory`) |
| `vllm_ascend/worker/worker.py` | Worker 初始化链路 |
| `vllm_ascend/worker/model_runner_v1.py` | NPUInputBatch、ACLGraph 相关 |
| `vllm_ascend/worker/npu_input_batch.py` | pinned CPU buffer 定义 |
| `vllm/v1/executor/multiproc_executor.py` | Worker 进程 fork 逻辑 |
| `vllm/v1/engine/core.py` | EngineCore 初始化流程 |

---

**作者**：vLLM-Ascend Team  
**日期**：2026-03-19  
**版本**：1.0
