# 动态聚集效应 · Rust + CUDA

SIMON / SIMECK 差分聚集与线性壳研究，基于《动态聚集效应及其在 SIMON 算法上的应用》。Rust 负责搜索、显存管理、结果记录和多卡调度；CUDA 负责 FP64 转移、选窗统计、投影质量评估和 NVLink 数据交换。主程序不依赖 Python。

当前已验证的 SIMON128 差分结果为 **45 轮，log₂ 概率 −127.574478192099**；尚未突破到 46 轮。这里计算的是论文模型下保留路径的聚集权重，线性模式使用平方相关路径权重，不等同于固定密钥的有符号相关性。见[研究结果](docs/endpoint-refinement.md)和[后续轮数探索](docs/round-search.md)。

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
DDW_TEST_DEVICES=0,1 cargo test --test gpu -- --ignored --test-threads=1

# 独立 CPU / Python / CUDA 回归
python -m pip install -e 'reference/python[cuda13]'
DDW_RUST="$PWD/target/release/ddw" DDW_TEST_DEVICES=0,1 \
  PYTHONPATH=.:reference/python python -m unittest discover -s reference/python/tests -v
```

默认 Rust 测试不启动 GPU。完整回归覆盖独立差分真值表、线性 Walsh 谱、各字长和模式、转移内核、伴随恒等式、多卡完整分布和续搜前缀。FP64 结果通过数值回归验证，不是区间算术证明。此次迁移的具体实测见[重构验证记录](docs/rust-migration.md)。
