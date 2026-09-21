# 动态差分窗口 · CUDA 重构

研究 SIMON / SIMECK 的差分聚集与线性壳。基于《动态聚集效应及其在 SIMON 算法上的应用》的转移模型，提供 GPU 搜索、历史结果复现、窗口扩展和替代搜索策略。

新实现位于 `ddw/`，原始 C++ 实现归档在 `legacy/`，41 份原始实验日志保留在 `experiments/`。运行实验不会提交 Git、上传结果或调用云服务。

## 安装

Python ≥ 3.10，需要 NVIDIA GPU 和匹配的 CUDA 环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda13]'
# CUDA 12 环境改用：pip install -e '.[cuda12]'
```

CUDA 内核由 CuPy / NVRTC 首次运行时编译。CPU 数学参考与日志工具仅依赖 NumPy；完整搜索需要 GPU。

## 运行

```bash
# SIMON64，输入差分 (0,1)，10 位窗口，23 轮
python -m ddw --device 0 --word-bits 32 --left 0 --right 1 \
  --width 10 --rounds 23 --output results/my-run.jsonl

# 线性模式：CLI 的输入和输出统一使用真实左右分支顺序
python -m ddw --device 0 --mode linear --left 0x44400 --right 0x1000 \
  --width 14 --rounds 25 --output results/my-linear.jsonl
```

`--word-bits` 指半个分组的字长；SIMON128 使用 64。线性模式内部交换分支以复用递推，但输入输出不需要手动交换。`--cipher simeck` 切换算法。

输出 JSONL 包含配置、每轮最大概率的 log₂、输出端点、窗口、保留总质量及同步后的墙钟耗时。拒绝覆盖已有文件。第一轮耗时可能包含 JIT；共享 GPU 上的时间不是独占性能基准。

## 复现与扩大论文窗口

```bash
python scripts/import_legacy_log.py \
  experiments/SIMON128_DIFFERENCE_FROM_0x1000_0x4440_PRECISION_14/info.log \
  results/my-reference.jsonl

# 相同宽度重放历史窗口
python -m ddw --device 0 --word-bits 64 --left 0x1000 --right 0x4440 \
  --width 14 --rounds 45 --replay-windows results/my-reference.jsonl \
  --output results/my-replay.jsonl

# 每一轮包含原窗口，并增加一个活跃位
python -m ddw --device 1 --word-bits 64 --left 0x1000 --right 0x4440 \
  --width 15 --rounds 45 --memory-gib 24 \
  --include-windows results/my-reference.jsonl --output results/my-wider.jsonl

python scripts/check_results.py results/my-replay.jsonl
python scripts/check_results.py results/my-wider.jsonl
```

新实现修正了旧代码中可能误删合法转移的消元问题，因此相同窗口下的概率可能略高于旧日志。`--replay-windows` 固定窗口，`--include-windows` 用于扩展。`check_results.py --reference` 则用于新内核之间的等值回归。

嵌套窗口保留旧搜索的全部路径，因此在精确算术下，同一端点的概率不会下降。实现额外记录旧端点上的新概率；验证脚本检查这一性质。浮点实现使用 FP64，不能视作严格区间算术证明。

矩阵空间仍随窗口宽度指数增长。相邻两轮都达到宽度 `w` 时，两张 FP64 矩阵合计：

| w | 两张矩阵 |
|---|---:|
| 10 | 16 MiB |
| 14 | 4 GiB |
| 15 | 16 GiB |
| 16 | 64 GiB |
| 17 | 256 GiB |

还需要陪集聚合缓存（最坏可达一张输入矩阵）、内核数据、统计缓存和 CUDA 上下文。`--memory-gib` 限制 CuPy 内存池，并在转移前检查矩阵预算和可用显存。它不预留 GPU，也不能保证其他进程增加显存后仍有空间。固定窗口计划支持 NVLink 多卡分片重放，见下文；动态选窗和束搜索目前仍在单卡执行。

## 探索其他策略

```bash
# 基于真实单比特输出边缘分布选窗
python -m ddw --device 0 --width 10 --rounds 23 --strategy marginal \
  --output results/my-marginal.jsonl

# 对单比特交换候选实际转移，按保留质量选窗
python -m ddw --device 0 --width 10 --rounds 23 \
  --lookahead-candidates 8 --lookahead-objective mass \
  --output results/my-local-search.jsonl

# 保留多个窗口历史，延缓贪心决策；各束之间不累加概率
python -m ddw.search --device 0 --width 10 --rounds 23 \
  --beam-size 8 --candidates 8 --objective peak --output results/my-beam.jsonl
```

替代策略属于实验功能，没有普遍优于原策略的保证。束搜索为控制显存限制宽度至 12。默认 `--kernel coset_lut` 按陪集归约后通过仿射查表查询输出，避免指数级枚举和逐输出消元；`--kernel coset` 保留上一版实现用于对照。`--kernel gather` 按输出求前像，`--kernel scatter` 按输入分发；四者在固定窗口下计算相同的转移。默认 `--statistics hierarchical` 用共享内存归约树计算位边缘分布，`gemm` 保留矩阵乘法对照。

## 验证

```bash
DDW_TEST_DEVICE=0 python -m unittest discover -s tests -v
python scripts/benchmark_legacy.py --width 10 --rounds 12 --output results/my-cpu.jsonl
```

测试覆盖差分穷举真值表、线性 Walsh 谱、四种 GPU 内核与 CPU 的数值一致性、64 位窗口索引、历史日志及非法结果检测。没有 CuPy/GPU 时 GPU 测试明确跳过。CPU 基准在临时目录编译旧递推，关闭检查点和发布操作。

[实测结果](docs/results.md)汇总速度与概率改进；[实现与研究判断](docs/research-notes.md)记录模型、数值边界和后续方向。实测原始记录放在 `results/`，修正前的探索记录单独存放在 `results/prototype/`。

## NVLink 多卡重放

```bash
python -m ddw.multigpu results/simon128-diff-w16.jsonl \
  --devices 0,1,2,3,4,5,6,7 --memory-gib 20 \
  --output results/my-eight-gpu.jsonl
python scripts/check_results.py results/my-eight-gpu.jsonl \
  --reference results/simon128-diff-w16.jsonl --tolerance 1e-10

# 多卡完整分布测试，可用任意支持 peer access 的两卡
DDW_TEST_DEVICE=1 DDW_TEST_DEVICES=1,2 python -m unittest discover -s tests -v
```

设备数必须为 2 的幂，且支持彼此的 CUDA peer access。输入按行分片，每卡计算不相交的输出列；下一轮通过 NVLink 直接读取远端列并重排为行分片，同时归一化。概率不跨卡重复计数。小窗口先用首卡运行，窗口足够大后启用全部指定卡。分片后不支持窗口缩小到小于卡数。

默认 `--exchange peer` 使用直接远端读取；`--exchange staged` 保留先复制到中间缓冲的对照实现。`--memory-gib` 是每卡预算，默认实现需约两张矩阵分片加陪集缓存与工作区，中间缓冲模式再增加一张分片。当前实现重放已有计划，不执行跨卡动态选窗。共享训练负载下的实测与具体限制见 [硬件优化记录](docs/hardware-optimization.md)。
