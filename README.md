# 动态差分窗口

[思路证明](https://ia9zk56a6c.feishu.cn/docs/doccnndtjyX7nHOSvjqdzRbc9rh)

## 用法

### 1. 修改超参数
修改[hyperparameter.hpp](./src/hyperparameter.hpp)，来确定初始点等信息

### 2. 编译程序
```bash
g++ main.cpp -Wall -Wextra -O3 -march=native -fopenmp -lm -o bin/{name}
```

### 3. 运行程序

```bash
./bin/{name}
```

### 4. 将结果上传到云端，并更新README的表格
```bash
python3 backup.py {name}
```
> 请注意更新README表格

### 5. 分析程序结果

可以通过`experiments/{name}/info.log`来查看运行的日志，或者运行`python3 help.py`来对结果进行分析。

### 6. 结果

[实验结果](https://ia9zk56a6c.feishu.cn/sheets/shtcnjMqiCpIk6EXckGuzb8gbKb)
