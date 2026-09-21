# 动态聚集效应 · Rust + CUDA

SIMON / SIMECK 差分聚集与线性壳研究，基于《动态聚集效应及其在 SIMON 算法上的应用》。Rust 负责搜索、显存管理、结果记录和多卡调度；CUDA 负责 FP64 转移、选窗统计、投影质量评估和 NVLink 数据交换。主程序不依赖 Python。

当前已验证的 SIMON128 差分结果为 **45 轮，log₂ 概率 −127.442713104012**；尚未突破到 46 轮。这里计算的是论文模型下保留路径的聚集权重，线性模式使用平方相关路径权重，不等同于固定密钥的有符号相关性。见[研究结果](docs/endpoint-refinement.md)、[后续轮数探索](docs/round-search.md)和[本轮对照](docs/window-selection.md)。

## 重要结果对照

按论文中的 15 个初始点逐项比较，每项只保留最好结果。比较轮数为原始记录中刚未达到阈值的那一轮；数值均为 log₂ 路径聚集权重，越大越好。diff = 优化后 − 优化前，按表中六位小数计算。**粗体表示该轮已过线。**

| 密码 | 方法 | 初始点 | 比较轮数 | 优化前 | 优化后 | diff |
|---|---|---|---:|---:|---:|---:|
| SIMON64 | 差分 | `(0x4000000, 0x11000000)` | 24 | -65.520874 | -65.286416 | +0.234458 |
| SIMON64 | 差分 | `(0x440, 0x1880)` | 25 | -68.820868 | -68.582667 | +0.238201 |
| SIMON64 | 差分 | `(0x1, 0x40000004)` | 24 | -65.520921 | -65.286878 | +0.234043 |
| SIMON64 | 差分 | `(0x80000, 0x222000)` | 25 | -65.732964 | -65.509391 | +0.223573 |
| SIMON64 | 线性 | `(0x40000004, 0x1)` | 24 | -64.164774 | **-63.937636** | +0.227138 |
| SIMON64 | 线性 | `(0x44400, 0x1000)` | 25 | -64.398116 | -64.088106 | +0.310010 |
| SIMON96 | 差分 | `(0x4000, 0x11101)` | 33 | -96.003572 | **-95.407092** | +0.596480 |
| SIMON96 | 线性 | `(0x400000004044, 0x1)` | 34 | -100.206918 | -99.932627 | +0.274291 |
| SIMON96 | 线性 | `(0x400000000044, 0x1)` | 34 | -98.716786 | -98.349747 | +0.367039 |
| SIMON96 | 线性 | `(0x1, 0x0)` | 36 | -99.185423 | -98.839309 | +0.346114 |
| SIMON128 | 差分 | `(0x1000, 0x4440)` | 45 | -128.665612 | **-127.442713** | +1.222899 |
| SIMON128 | 线性 | `(0x4000000000000004, 0x1)` | 45 | -131.069584 | -130.650251 | +0.419333 |
| SIMON128 | 线性 | `(0x4000000000000044, 0x1)` | 46 | -131.751124 | -131.290196 | +0.460928 |
| SIMON128 | 线性 | `(0x1, 0x0)` | 44 | -131.706222 | -131.217020 | +0.489202 |
| SIMECK64 | 差分 | `(0x0, 0x1)` | 32 | -64.483982 | **-63.936070** | +0.547912 |

15 项均已完成新版补跑；其中 3 项保留此前更大预算下的更强结果。原始窗口大小均为 14，优化后可使用更大的窗口与去重路径并集。历史数值差异、计算配置及每项结果来源见[实验记录](docs/important-results.md)和[结果清单](results/best/important-results.json)。

本轮新增达到：**SIMON64 线性 24 轮、SIMON96 差分 33 轮、SIMECK64 差分 32 轮**（对应表中初始点）。

线性模式计算平方相关路径权重；过线指超过 `2^−分组长度`，不等于固定密钥带符号相关性的直接验证。

## 构建与运行

需要 近期稳定版 Rust（已验证 1.98.1）、NVIDIA 驱动和 CUDA Toolkit。B300 环境使用 CUDA 13。CUDA 源码嵌入可执行文件，在启动时由 NVRTC 按实际 GPU 架构编译。

```bash
CUDA_HOME=/usr/local/cuda cargo build --release

# SIMON64：字长为半个分组长度
./target/release/ddw search --word-bits 32 --left 0 --right 1 \
  --width 10 --rounds 23 --devices 0 --output results/my-search.jsonl

# 线性模式的输入输出使用真实左右分支顺序
./target/release/ddw search --mode linear --left 0x44400 --right 0x1000 \
  --width 14 --rounds 25 --output results/my-linear.jsonl
```

共享文件系统上建议设置 `CARGO_TARGET_DIR=/tmp/ddw-target`，随后从该目录运行二进制。所有命令拒绝覆盖已有输出。程序不会提交 Git、上传结果或调整其他 GPU 任务。

## 重放、扩窗与继续搜索

```bash
# 固定窗口重放：配置从 JSONL 读取
./target/release/ddw replay results/baselines/simon64-diff-w10.jsonl \
  --output results/my-replay.jsonl
./target/release/ddw validate results/my-replay.jsonl \
  --reference results/baselines/simon64-diff-w10.jsonl

# 八卡重放最佳 45 轮计划，预算按每卡计算
./target/release/ddw replay results/best/simon128-w17-endpoint-expanded.jsonl \
  --devices 0,1,2,3,4,5,6,7 --memory-gib 60 \
  --output results/my-eight-gpu.jsonl

# 扩窗：每轮包含参考窗口，新增位由当前分布选取
./target/release/ddw replay results/best/simon128-w14-endpoint-refined.jsonl \
  --width 15 --memory-gib 28 --output results/my-expanded.jsonl

# 保留已有前缀，继续搜索新轮次
./target/release/ddw replay results/best/simon128-w14-endpoint-refined.jsonl \
  --rounds 46 --extend --candidates 8 --output results/my-46-rounds.jsonl

# 用反向权重优化最终端点，而非仅优化下一轮
./target/release/ddw refine results/baselines/simon128-diff-w14.jsonl \
  --device 0 --memory-gib 28 --host-memory-gib 128 \
  --candidates 8 --passes 2 --output results/my-refined.jsonl
```

`--candidates` 比较局部换位候选；`--objective mass` 通过投影直接计算保留质量，`peak` 计算完整候选分布。搜索和扩窗均可多卡执行，端点优化目前单卡执行并在主机缓存反向权重。贪心、局部优化和扩窗均不保证增加轮数。

多卡数量必须为 2 的幂，并支持两两 CUDA peer access。小窗口先在首卡运行；足够大后按行分片，下一轮通过远端读取重排，窗口缩小时可退回单卡。没有概率重复计数。

满窗口单张 FP64 矩阵大小为 `8 × 2^(2w)` 字节：w14 为 2 GiB，w15 为 8 GiB，w17 为 128 GiB。还需要输出矩阵、陪集与统计缓存。`--memory-gib` 限制每卡由程序分配的缓冲，并检查剩余显存；预算不是独占预留。计时来自共享 B300，不能当作硬件上限。

## 更宽窗口与端点后验搜索

新增 `compressed` 无损隐式矩阵后端：保持 FP64，把转移输出保存为陪集值和仿射映射，而非完整矩阵。已在共享单卡上完成 w19；对应稠密单状态为 2 TiB，实际表示约 14 GiB，另需工作区和前后状态。它降低显存需求，计算仍随窗口宽度指数增长。

`refine --strategy posterior` 用最终端点后验质量评价全部单比特换位；`mirror` 构造对称计划；`symmetric-union` / `path-union` 对同端点路径集合进行精确去重。SIMON64 线性 25 轮经跨轮选窗与新增路径搜索达到 **−64.088105604252**，仍低于 −64 阈值。最新结果、验证与命令见[跨轮与新增路径搜索](docs/cross-round-search.md)；后端说明见[隐式矩阵搜索](docs/implicit-search.md)。

已实现按端点贡献自适应分配自由位：只加宽三轮即可优于原均匀 w15 方案，详见[选窗算法对照](docs/window-selection.md)。新增路径评分已实现；进一步优化方案见[算法优化方向](docs/algorithm-directions.md)。

## 仓库结构

| 目录 | 用途 |
|---|---|
| `src/` | Rust CLI、CUDA 驱动封装、转移调度、多卡和端点优化 |
| `cuda/` | 唯一一份转移内核，以及 Rust 运行时辅助内核 |
| `tests/` | Rust GPU 集成测试；基础测试随 Rust 模块维护 |
| `reference/python/` | CPU 数学参考、独立 GPU 对照和实验性束搜索/路径并集 |
| `tools/` | 历史日志导入与独立 JSONL 检查 |
| `experiments/` | 41 份原始实验日志，保持原样 |
| `results/` | 按用途分类的原始结果，见[索引](results/README.md) |
| `docs/` | [实现说明](docs/architecture.md)、验证与研究记录 |

旧 C++ 主程序和中间原型已从当前分支移除，可从 `master` 或重构前提交恢复。论文文件保留在本地。Python 参考实现的[使用方法](reference/python/README.md)与生产入口分开维护。

## 验证

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
# 显式指定共享测试卡；测试只分配少量显存
DDW_TEST_DEVICE=0 DDW_TEST_DEVICES=0,1 cargo test -- --ignored --test-threads=1

# 独立 CPU / Python / CUDA 回归
python -m pip install -e 'reference/python[cuda13]'
DDW_RUST="$PWD/target/release/ddw" DDW_TEST_DEVICES=0,1 \
  PYTHONPATH=.:reference/python python -m unittest discover -s reference/python/tests -v
```

默认 Rust 测试不启动 GPU。完整回归覆盖独立差分真值表、线性 Walsh 谱、各字长和模式、转移内核、伴随恒等式、多卡完整分布和续搜前缀。FP64 结果通过数值回归验证，不是区间算术证明。此次迁移的具体实测见[重构验证记录](docs/rust-migration.md)。
