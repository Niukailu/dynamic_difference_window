# 动态差分窗口

## 用法

### 1. 修改超参数
修改[hyperparameter.hpp](./src/hyperparameter.hpp)，来确定初始点等信息

### 2. 运行程序
```bash
bash scripts/run.sh {name}
```
> 这个脚本会自动编译程序、运行程序、将结果上传到github以及阿里云，但是不会更新实验表格，所以请注意更新README表格！

### 3. 分析程序结果

可以通过`experiments/{name}/info.log`来查看运行的日志，或者运行`python3 help.py`来对结果进行分析。

### 4. 结果

点击这里查询<a href="https://ia9zk56a6c.feishu.cn/sheets/shtcnjMqiCpIk6EXckGuzb8gbKb" target="_blank">实验结果</a>。
