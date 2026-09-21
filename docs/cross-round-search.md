# 跨轮选窗与新增路径搜索

本轮实现相邻两轮联合选窗，以及相对于固定参考计划的新增路径评分。计算仍为 Rust + CUDA；`tools/block_sweep.py` 仅调度八张 GPU 的独立搜索任务。目标端点固定为 SIMON64 线性 25 轮 `(0x44400, 0x1000) → (0x1000, 0x44400)`。

## 为什么改目标函数

参考宽窗计划 A 已经覆盖大部分高权重路径。继续最大化候选 B 的总权重，容易重复找到 A 已有的路径。现在直接最大化 `P(B \ A)`，允许 B 自身总权重下降，只要对参考集合的补充增多。

`src/tagged.rs` 在每个状态保存两个非负量：始终满足 A 的路径质量 I，以及曾经离开 A 的路径质量 O。设 T 为当前 B 窗口下的转移，M 为输出属于本轮 A 窗口的指示函数：

```text
I' = M · T(I)
O' = T(O) + (1 − M) · T(I)
```

在固定终点读取 O 就得到新增路径权重。反向消息使用对应伴随递推，终端条件为 `(H_I, H_O) = (0, 目标点指示函数)`。前后向点积用于局部候选评分。搜索过程不通过两个接近的大数相减求新增量；另行重放交集，验证 `P(B) = P(B∩A) + P(B\A)`。

目前参考是**单个 A**，并未直接优化 `P(B \ (A∪反射A))`。最终结果必须另算 A、反射 A、B、反射 B 的完整 15 项容斥，不能把新增量简单加到旧并集上。

## 如何联合选窗

`block-refine --round r` 同时修改第 r、r+1 轮窗口，宽度保持不变。先用 GPU 后验统计评价全部单比特替换，再为每个窗口保留原窗口、高分候选及不同新增自由位的代表。本轮每边 32 个候选，穷举 1,024 个组合；单独评分较差的候选也有机会进入组合。

固定前缀与局部块之后的反向消息复用，每个第一窗口候选的首步转移也复用。选出的结果必须通过完整带标签重放、普通端点重放和独立交集重放。这里仅保证候选池内最优，不是所有窗口或所有历史的全局最优。

每次八卡 sweep 从同一源计划出发，只接受最佳完整计划，再开始下一次；不拼接各卡独立修改而跳过核算。

## 实验

参考 A 是 `dedicated-expansion/gpu-0-w20.jsonl`，候选 B 从 `simon64-linear25-posterior-w14.jsonl` 出发，始终保持 w14。下表均为 log₂ 权重。

| 阶段 | B 的目标权重 | B 在 A 之外的新增权重 | 含反射的完整并集 |
|---|---:|---:|---:|
| 原候选 B | −64.263484114764 | −74.060602440675 | 未在本轮核算 |
| 第一批，第 12–13 轮 | −64.627224956918 | −71.462930723662 | −64.094060240130 |
| 第二批，第 10–11 轮 | −64.650925274318 | −70.746669250771 | −64.088105604252 |
| 第三批，第 13–14 轮 | −64.867444619230 | −70.462383169268 | −64.088155335019 |

此前 A∪反射 A 为 −64.104083953419。最终保留第二批为最佳 **−64.088105604252**，总体权重提高约 **1.11%**，距 −64 阈值仍差 **0.088106 bit**。第三批的单参考新增评分更高，完整并集反而略差，说明优化目标与最终评价确有差异。最佳记录见[完整并集](../results/best/simon64-linear25-novel-union.json)。第一批另有四个普通端点目标对照（第 10、11、12、13 轮起始块），均未改善。新增路径目标则找到互补计划。新增路径量的倍数提升不代表总体权重的同倍数提升，也不能据此断言联合优化优于所有逐轮优化策略。

原始记录在 `results/exploration/cross-round-search/`、`novel-block-sweep/`、`novel-block-sweep-next/`。实验后段其他训练重新占用各卡约 120–164 GiB，最后一批将每进程预算改为 64 GiB；没有停止其他任务。这些耗时不能作为独占性能结论。

仍然没有证明 25 轮达到理论极限，也没有突破 −64 阈值。这些数值是论文模型下的平方相关路径聚集权重，不是固定密钥下带符号线性相关性的证明。

## 复现

输出目录/文件必须不存在。以下内存预算适用于此 w14 候选，不能直接推广到更宽窗口。

```bash
./target/release/ddw block-refine \
  results/exploration/simon64-linear25-posterior-w14.jsonl \
  --reference results/exploration/dedicated-expansion/gpu-0-w20.jsonl \
  --round 12 --pool 32 --device 0 --memory-gib 64 \
  --output results/my-novel.jsonl

python tools/block_sweep.py results/my-novel.jsonl \
  --reference results/exploration/dedicated-expansion/gpu-0-w20.jsonl \
  --binary ./target/release/ddw --devices 0,1,2,3,4,5,6,7 \
  --rounds 7,8,9,10,11,12,13,14 --pool 32 --iterations 1 \
  --memory-gib 64 --output results/my-novel-sweep

./target/release/ddw path-union \
  results/exploration/dedicated-expansion/gpu-0-w20.jsonl \
  results/my-novel.jsonl --reflect --device 0 --memory-gib 64 \
  --output results/my-novel-union.json
```

## 验证范围

- 独立 CPU 参考穷举小字长候选池的全部组合：SIMON 差分/线性、普通/新增目标、三个起始位置，共 12 个案例，涵盖有后缀与末尾块。
- GPU 带标签分布逐状态对照普通分布与 CPU 参考窗口投影，覆盖 SIMON/SIMECK、差分/线性，并检查各截断位置前后向恒等式。
- 稀有路径测试同时放入 0.5 的旧质量与 2^-80 的新增质量，确认新增量未因相减消失。
- JSONL 保留真实 `log2_max` 和峰值端点，另记 `target_output`、`log2_target`；并集读取声明的目标而非误用峰值。独立 CPU 测试覆盖峰值与目标不同的计划。
- 结果审计见 `results/validation/cross-round-checks.json`。FP64 容斥与数值容差检查不是区间算术证明。
