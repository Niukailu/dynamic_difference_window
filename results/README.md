# 实验结果索引

| 目录 | 内容 |
|---|---|
| `baselines/` | 历史日志转换、CPU 对照和扩窗基线 |
| `best/` | 当前最佳端点优化/路径并集；验证范围见对应研究文档 |
| `validation/` | 独立内核、Rust 和多卡复核 |
| `benchmarks/` | 共享 GPU 下的计时原始记录、阶段诊断与旧 C++ 原型对照 |
| `exploration/` | 候选搜索、输入扫描、续搜、路径并集及失败尝试 |

重点记录：

- [SIMON128，45 轮，w19，−127.442713104012](best/simon128-w19-endpoint-expanded.jsonl)。
- [SIMON64，25 轮，w20 反射并集，−64.104083953419](best/simon64-linear25-dedicated-union.json)，仍低于 −64 阈值。
- [自适应宽度选择](exploration/adaptive-width-w14/summary.json)与[同规模选窗对照](../docs/window-selection.md)。

- [SIMON128，45 轮，w17，−127.574478192099](best/simon128-w17-endpoint-expanded.jsonl)；[Rust 八卡逐轮复核](validation/rust-eight-gpu-w17.jsonl)。
- [SIMON128，45 轮，w14，−127.859300186727](best/simon128-w14-endpoint-refined.jsonl)。
- [SIMON64，23 轮，w10，−65.590628604365](best/simon64-w10-endpoint-refined.jsonl)。
- [SIMON128 继续至 48 轮](exploration/simon128-best-continuation48.jsonl)：46 轮未达到 2^-128。
- [SIMON64 线性 25 轮最新并集](best/simon64-linear25-implicit-union.json)：**−64.105767763312**，未达到 2^-64；[此前结果](exploration/simon64-linear25-symmetric-union.json)为 −64.134564031949。方法与验证见[隐式矩阵搜索](../docs/implicit-search.md)。

JSONL 首行为配置，其后是逐轮记录，可带末尾汇总。`legacy-*` 的概率来自原日志六位小数；其余结果使用修正后的投影消元。`exploration/simon64-diff-w10-marginal.jsonl` 第 18 轮丢失全部路径，是失败证据，不能当作完整 23 轮运行。

整理目录时未改写既有记录中的配置、概率、计时和源码哈希。元数据中的输入路径可能是搬迁前的位置，可按文件名在上述分类查找。消元修正前的 `prototype/` 无效中间结果已移除，可从提交 `c260b76` 恢复。性能诊断快照不代表最终实现的严格消融实验；共享训练负载下的计时不能视为独占峰值。
