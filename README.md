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