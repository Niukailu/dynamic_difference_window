# 共享 B300 上的第二轮优化（2026-09-21）

> 这是 Python/CuPy 阶段的研究记录。旧命令需先安装 `reference/python`；当前 Rust 入口见[根目录 README](../README.md)。结果链接已按新目录更新。

所有工作在 `gpt-6-astra` 完成，未独占 GPU 或调整其他训练进程。比较对象是上一版 GPU 重构提交 `0f62945`，不是原始 CPU 程序。

## 单卡完整流程

SIMON64 差分、输入 `(0,1)`、w15、23 轮、GPU 0，固定包含已有同宽窗口；仍执行窗口评分，因此包含选窗、转移、统计、归一化和内存管理。新旧进程交替运行，顺序为旧/新、新/旧、旧/新。每轮计时均在 GPU 同步后结束；不计解释器启动，首轮可能包含 JIT。

| 实现 | 第一次 | 第二次 | 第三次 | 中位数 |
|---|---:|---:|---:|---:|
| 上一版 | 7.207 s | 7.285 s | 7.918 s | 7.285 s |
| 新版 | 2.561 s | 1.702 s | 1.637 s | 1.702 s |

中位数比值 **4.28 倍**。仅统计第 6–23 轮，中位数由 6.897 s 降到 1.511 s，约 4.57 倍。所有运行的每轮 log₂ 最大概率与原窗口计划误差不超过 1.4211e-14。原始文件在 [hardware-benchmark](../results/benchmarks/hardware-benchmark/) 的 `before-*` / `after-*`。

这与之前 2.211 秒的 coset 基准不是同一口径：之前只重放、不评分，GPU 编号和共享负载也不同。不能把两批时间直接作加速比。

复现交替对照：

```bash
git worktree add --detach /tmp/ddw-baseline-0f62945 0f62945
python reference/python/scripts/benchmark_hardware.py --baseline-root /tmp/ddw-baseline-0f62945 \
  --device 0 --repeats 3 --output-dir results/my-hardware-benchmark
```

若该临时 worktree 已存在，复用即可。脚本验证基线提交及受跟踪文件未修改。

## 具体改动

1. **窗口评分**：原 FP64 GEMM 为每个概率重复做多个物理位的乘加。分层内核复用共享内存归约树，一次读取计算所有坐标位的边缘质量，再映射到物理位。保留 FP64，不依赖低精度 Tensor Core。
2. **陪集求和**：warp 子组分别拥有互不相交的陪集，Gray 次序遍历子空间，以寄存器归约替代 FP64 原子加和大缓存清零。
3. **输出查询**：把目标坐标到可行性综合征/陪集编号的仿射映射编译为 8 位分块查表。每个输出只需少量读取和 XOR，替代重复消元。
4. **统计与归一化**：在输出内核中顺便计算峰值、首次 argmax 和总质量；避免重新扫完整矩阵。归一化用一次标量倒数加逐元素乘法，替代逐元素 FP64 除法。
5. **分配**：窗口大小稳定后复用两张概率矩阵，减少大块分配和内存池反复清理。

`results/benchmarks/profile-*.json` 是逐步实现时的诊断快照，包含 CUDA event 的阶段和内核时间。它们对应不同开发阶段，不能当成最终代码的严格消融实验。`reference/python/scripts/profile_gpu.py` 可重测阶段；其中 `--fused` 指独立的合并统计扫描，完整 CLI 还进一步将统计融合进输出写入。性能结论使用上面的完整流程交替对照。

## 真正的多卡单任务

实现 `python -m ddw.multigpu`：将输入矩阵按左窗口坐标分成连续行，每卡生成不相交的输出列。计算完成后，各卡通过 peer access 直接读取所需远端列，重排为下一轮行分片，并融合归一化。仅峰值、总质量等标量回主机；不会把重叠概率相加。

SIMON128 差分 w16、同一固定窗口、45 轮，两次交替测量（八卡/单卡/八卡/单卡）：

| 实现 | 第一次 | 第二次 | 中位数 |
|---|---:|---:|---:|
| 新版单卡 GPU 0 | 8.350 s | 7.608 s | 7.979 s |
| 新版八卡 GPU 0–7 | 3.244 s | 4.321 s | 3.782 s |

此规模约 **2.11 倍**加速，远非八倍。八卡每次交换合计 1.053 / 0.759 秒，计算合计 2.111 / 3.484 秒；共享负载和最慢卡等待仍然显著。原始记录为 `hardware-benchmark/single-*` / `eight-*`。全部 45 轮的 log₂ 最大概率与原 w16 结果完全一致，45 轮仍为 **−127.761229169588**。

早期中间缓冲版本一次运行总计 4.461 秒，其中交换 2.140 秒；直接读取版本第一次总计 3.683 秒，其中交换 0.786 秒。两次不是交替重复对照，仅用于说明优化方向；文件分别为 `multigpu-8-w16.jsonl` / `multigpu-8-w16-peer.jsonl`。前者记录产生在增加 `exchange` 元数据之前，实际为 staged；`multigpu-2-w10.jsonl` 同样为 staged。

复现单卡和八卡：

```bash
python -m ddw --device 0 --word-bits 64 --left 0x1000 --right 0x4440 \
  --width 16 --rounds 45 --memory-gib 80 \
  --replay-windows results/baselines/simon128-diff-w16.jsonl --output results/my-single16.jsonl
python -m ddw.multigpu results/baselines/simon128-diff-w16.jsonl \
  --devices 0,1,2,3,4,5,6,7 --memory-gib 20 --output results/my-eight16.jsonl
python tools/check_results.py results/my-eight16.jsonl \
  --reference results/baselines/simon128-diff-w16.jsonl --tolerance 1e-10
```

## 正确性与剩余空间

16 项测试通过，包括真值表、CPU 完整分布对照、四种转移内核、四种位统计实现、argmax 平局、融合统计、输出缓冲复用，以及两卡下差分/线性完整矩阵多轮对照。peer 与 staged 两种交换均测试；日志检查另覆盖全部长轮运行的峰值、端点概率和质量守恒约束。

没有测出硬件上限：没有独占基准或硬件计数器证据，不能声称达到峰值带宽或最佳占用率。多卡当前只重放已有窗口，跨卡动态选窗尚未实现。继续扩大规模时还可研究跨卡位统计归约、跨轮布局交替来减少重排、降低 Python 调度和同步开销，以及更紧凑的概率表示。必须保持小概率累积和全局归一化的正确性，不能只为了利用 Tensor Core 降低精度。

本轮提升的是运算效率和单任务容量，未声称进一步增加区分器轮数。
