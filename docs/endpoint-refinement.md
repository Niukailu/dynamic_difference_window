# 用最终端点指导窗口优化（2026-09-21）

本轮不再用下一轮质量或峰值代替最终目标，而是固定原结果的最后一轮端点，利用前向分布和反向权重优化整条窗口计划。所有运行仍在共享 B300 环境进行，未调整训练任务。

## 新结果

| 实验 | 优化前 log₂ 概率 | 优化后 log₂ 概率 | 变化 |
|---|---:|---:|---:|
| SIMON64，23 轮，w10；从已有束搜索结果出发 | −66.288552381355 | **−65.590628604365** | 约 1.62 倍 |
| SIMON128，45 轮，w14；相同宽度优化 | −128.665486583123 | **−127.859300186727** | 约 1.75 倍 |
| SIMON128，45 轮，w17；新计划扩窗，对比此前最强 w17 | −127.725579424791 | **−127.574478192099** | 约 1.1104 倍 |

SIMON128 输入和输出保持 `(0x1000, 0x4440)` → `(0x444040, 0x100000)`。新的 w17 结果比 2^-128 高约 **0.42552 bit**。没有增加轮数，也没有进行真实密码长轮数的明文对统计；这些仍是论文模型下的路径聚集估计，不宣称新的公开最佳纪录。

同宽 w14 已达到 45 轮超过 2^-128 的判据，并优于之前 w15 的 −127.912010。w14 单张满矩阵为 2 GiB，w17 为 128 GiB；这说明窗口选择的改进能明显降低达到给定概率所需的状态规模。

原始记录：

- [SIMON64 优化](../results/simon64-w10-endpoint-refined.jsonl)、[gather 复核](../results/simon64-w10-endpoint-refined-check.jsonl)
- [SIMON128 w14 优化](../results/simon128-w14-endpoint-refined.jsonl)、[gather 复核](../results/simon128-w14-endpoint-refined-check.jsonl)
- [SIMON128 w17 扩窗](../results/simon128-w17-endpoint-expanded.jsonl)、[coset 复核](../results/simon128-w17-endpoint-expanded-check.jsonl)

优化日志包含每次接受的窗口修改、修改前后的最终端点 log₂ 概率和最终重放结果。最早两份优化日志继承了源配置中的旧 kernel 名称；已显式记录 `metadata_correction` 并改为实际使用的 `coset_lut`，数值和计时未改动。

## 为什么反向权重有效

前向一步从 `(l,r)` 到 `(t,l)`，转移权重只依赖 `l` 和 `t XOR r`。因此固定窗口算子的转置可通过交换目标/右窗口、前后转置矩阵复用已有精确转移内核：

`adjoint(H, L, R, T) = step(H.T, L, T, R).T`

这里的转置是线性算子的伴随，不是在猜逆向密码差分。终点处放一个值为 1 的指示函数，逐轮应用转置算子，得到每个中间状态沿指定后续窗口到达终点的权重。反向权重也逐轮按最大值归一化，单独累计 log₂ 尺度。

改变第 r 轮窗口会改变第 r、r+1 轮的状态坐标；向前计算三次后，第 r+2 轮恢复为相同的两侧窗口。因此每个候选只需三次局部传播，再与缓存的后续反向权重做内积，就能准确评价最终端点概率。末尾不足三轮时直接读取目标端点。

优化从前到后逐轮进行：保留当前窗口作为候选，另外由已有位评分提出若干窗口，只接受最终端点概率增加的修改。一个扫描期间未来窗口还未改变，所以反向缓存始终对应当前正在评价的后续计划；下一次扫描重新构造缓存。每次扫描结束还会用实际前向结果检查目标概率没有下降。

这是候选集合上的坐标优化，不保证全局最优。它不合并互相重叠的窗口历史，不通过重复计数增加概率。

## 实现与开销

新增 `Engine.adjoint` 和 `python -m ddw.refine`。优化过程在单卡执行；之后可以用已有八卡动态扩窗扩大获胜计划。

- w10：提出 12 个候选，加上当前窗口；最多 3 次扫描，实测第二次无改进后停止，优化约 9.28 秒。
- w14：提出 8 个候选，加上当前窗口；2 次扫描接受 39 次修改，优化约 122.02 秒。
- w14 反向缓存放主机内存，约 84.03 GiB；GPU 内存池预算 20 GiB，主机预算 128 GiB。设备缓存也可选，运行前检查保守容量估计。
- 将获胜 w14 计划扩到 w17 使用 8 卡、每卡内存池预算 60 GiB；本次约 15.40 秒，期间另有复核任务共享 GPU，因此不作为性能基准。

这是花更多搜索时间换更好窗口，不是对完整搜索宣称加速。缓存空间随轮数和窗口状态数增加，主机缓存只减少 GPU 常驻容量，不消除总内存需求。

## 验证

19 项测试通过，包括：

- CPU 前向算子与 GPU 伴随的内积恒等式，覆盖 SIMON/SIMECK 和差分/线性；
- 每个候选的局部传播加反向内积评分，与 CPU 从第一轮到最后一轮的完整重放一致，包含最后几轮、端点不在候选窗口等情况；
- 主机缓存与设备缓存的全部反向消息和最终窗口选择一致；
- 优化后目标概率不下降，以及已有单卡/多卡完整分布测试。

实际结果另用不同转移内核重放完整计划：SIMON64 w10 的所有轮峰值完全一致；SIMON128 w14 的最大 log₂ 误差为 1.4211e-14；w17 的最大 log₂ 误差同为 1.4211e-14。不同转移内核仍共享部分基构造代码，不等于完全独立的密码分析程序。

## 复现

```bash
python -m ddw.refine results/simon64-diff-w10-beam-replay.jsonl \
  --device 0 --candidates 12 --passes 3 --output results/my-refined64.jsonl

python -m ddw.refine results/simon128-diff-w14.jsonl \
  --device 1 --cache host --host-memory-gib 128 --memory-gib 20 \
  --candidates 8 --passes 2 --output results/my-refined128.jsonl

python -m ddw.multigpu results/my-refined128.jsonl --width 17 \
  --memory-gib 60 --output results/my-expanded128.jsonl

python -m ddw.multigpu results/my-expanded128.jsonl --kernel coset \
  --memory-gib 60 --devices 7,6,5,4,3,2,1,0 --output results/my-expanded128-check.jsonl
python scripts/check_results.py results/my-expanded128-check.jsonl \
  --reference results/my-expanded128.jsonl --tolerance 1e-10
```

单纯优化窗口时，中间轮的峰值或旧端点概率可能下降，因此不对优化前后使用等值校验；应比较固定最终端点。获胜计划的另一内核重放才使用 `--reference` 做等值校验。
