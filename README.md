# 动态差分窗口

[思路证明](https://ia9zk56a6c.feishu.cn/docs/doccnndtjyX7nHOSvjqdzRbc9rh)

## 用法

### 1. 修改超参数
修改[hyperparameter.hpp](./src/hyperparameter.hpp)，来确定初始点等信息

### 2. 编译程序
```bash
# simon
g++ simon.cpp -Wall -Wextra -O3 -march=native -fopenmp -lm -o bin/simon

# simeck
g++ simeck.cpp -Wall -Wextra -O3 -march=native -fopenmp -lm -o bin/simeck
```

### 3. 运行程序

```bash
# simon
mkdir experiments/{name}/
./bin/simon 2>&1 | tee experiments/{name}/info.log

# simeck
mkdir experiments/{name}/
./bin/simeck 2>&1 | tee experiments/{name}/info.log
```