# Python 数值参考与研究工具

主程序是仓库根目录的 Rust `ddw`。本目录保留独立 CPU 数学参考、CuPy 对照内核和尚处于实验阶段的束搜索、反射路径并集、输入扫描，便于检验 Rust/CUDA 的结果。

在**仓库根目录**执行：

```bash
python -m pip install -e 'reference/python[cuda13]'
PYTHONPATH=.:reference/python python -m unittest discover -s reference/python/tests -v
python -m ddw --help
python -m ddw.search --help
python -m ddw.refine --help
python -m ddw.path_union --help
```

没有 GPU 时 CPU 参考仍可使用，GPU 测试明确跳过；`DDW_RUST=/absolute/path/to/ddw` 启用 Rust CLI 与 CPU 的逐轮交叉测试。`DDW_TEST_DEVICE` 选择单卡，`DDW_TEST_DEVICES` 启用多卡验证。

本包仅支持当前仓库内的 editable 安装：CUDA 源码统一读取根目录 `cuda/kernels.cu`，不复制另一份。控制台基础入口命名为 `ddw-reference`，避免与 Rust 二进制冲突。旧研究文档中的 `python -m ddw ...` 命令仍指此参考实现。

`scripts/` 保留独立内核基准、阶段性能分析、候选质量基准和输入扫描。导入原始日志使用根目录 `tools/import_legacy_log.py`；结果检查使用 `tools/check_results.py`。旧 CPU 与临时 C++ 原型的构建脚本已删除，历史实现可由 Git 恢复。
