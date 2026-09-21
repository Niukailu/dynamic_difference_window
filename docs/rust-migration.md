# Rust + CUDA 重构验证（2026-09-21）

主程序已迁移为原生 Rust host，通过 CUDA Driver API 和 NVRTC 调用嵌入的 CUDA 内核，不启动 Python 子进程。支持单卡/多卡搜索、固定计划重放、嵌套扩窗、前缀续搜、投影候选质量评估、端点伴随优化和 JSONL 验证。

Python 的 CPU 数学参考、独立 gather/scatter/coset GPU 算法与实验性束搜索/路径并集保留在 `reference/python/`。旧 C++ 主程序、临时 C++ 原型、重复 CUDA 文件、无效原型结果和生成物已清理。41 份原始实验日志保持不变，论文文件保留在本地。修改只发生在 `gpt-6-astra`，`master` 仍为 `f06150bbab115ad55f308bb1541d9c15f3549602`。

## 数值验证

- Rust 单卡重放 SIMON64 w10、23 轮，与原计划逐轮 log₂ 峰值完全一致。
- [Rust 八卡重放](../results/validation/rust-eight-gpu-w17.jsonl) SIMON128 w17、45 轮，与已有最佳计划逐轮完全一致；最后一轮为 −127.574478192099。
- Rust GPU 集成测试覆盖 SIMON/SIMECK × 差分/线性：投影质量与完整转移一致，伴随内积恒等式成立，矩阵转置往返一致，两卡与单卡完整分布一致，并验证收缩回单卡。
- Python 交叉测试共 24 项全部通过，其中 Rust CLI 测试覆盖 16/24/32/48/64 字长 × 两密码 × 两模式的 20 种组合，并用独立 CPU 转移逐轮核对峰值和质量；另检查续搜前缀保留，以及差分/线性模式下的多卡扩窗与续搜。
- [Rust 端点优化](../results/validation/rust-simon64-refined.jsonl)把 SIMON64 w10 的 23 轮从 −66.288552381355 提高到 −65.613252139494；[Python gather 复核](../results/validation/rust-simon64-refined-gather.jsonl)一致。它未超过已有 Python 搜索的 −65.590628604365，因此没有替换 `best/` 中的计划。浮点归约顺序与并列候选处理可能改变局部搜索轨迹，等值保证针对固定计划。

## 性能实测

在同一张共享 B300 上，对同一 SIMON64 w15、23 轮计划交替运行 Rust 与 Python，各三次。每次的逐轮峰值误差均不超过 1.43×10^-14。原始记录和汇总见 [benchmarks/rust-python](../results/benchmarks/rust-python/summary.json)。

| 中位数 | Rust | Python |
|---|---:|---:|
| 进程墙钟时间，包含启动/编译 | 3.4622 s | 3.3605 s |
| 各轮累计时间 | 1.0149 s | 0.9079 s |

这组测试没有证明 Rust 更快；轮内累计时间约高 11.8%，进程时间约高 3.0%。共享训练负载存在波动，不能据此判断独占硬件峰值。CUDA 转移核心相同，主机语言变化本身不改变转移复杂度。前期临时 C++ 原型的性能记录单独保留，不能套用到 Rust。

复现：

```bash
python tools/benchmark_hosts.py results/baselines/simon64-diff-w15.jsonl \
  --rust ./target/release/ddw --device 0 --repeats 3 \
  --output-dir results/my-host-benchmark
```

当前仍可优化 Driver API 同步/分配、持久工作线程、NVRTC 缓存和归一化融合。研究侧未突破新的轮数，后续重点见[轮数探索](round-search.md)。
