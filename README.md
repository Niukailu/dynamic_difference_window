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

### 4. 分析程序结果

可以通过`experiments/{name}/info.log`来查看运行的日志，或者运行`python3 help.py`来对结果进行分析。

### 5. 结果

<table>
	<tr>
	    <th rowspan="2" style="text-align: center;">密码版本</th>
        <th rowspan="2" style="text-align: center;">初始点</th>
	    <th colspan="3" style="text-align: center;">原始对比实验</th>
	    <th colspan="3" style="text-align: center;">本文实验</th>
	</tr>
    <tr>
        <th style="text-align: center;">终止点</th>
        <th style="text-align: center;">概率</th>
        <th style="text-align: center;">轮数</th>
        <th style="text-align: center;">终止点</th>
        <th style="text-align: center;">概率</th>
        <th style="text-align: center;">轮数</th>
    </tr>
    <!-- SIMON64 -->
    <tr>
	    <td rowspan="3">SIMON64</td>
	    <td>(0x4000000, 0x11000000)</td> <td>(0x11000000, 0x4000000)</td> <td>-57.57</td> <td>21</td> <td>(0x11000001, 0x40000000)</td> <td>-60.44</td> <td>23</td>
	</tr>
    <tr>
	    <td>(0x440, 0x1880)</td> <td>(0x440, 0x100)</td> <td>-61.32</td> <td>22</td> <td>(0x4440, 0x1000)</td> <td>-63.74</td> <td>24</td>
	</tr>
    <tr>
	    <td>(0x1, 0x40000004)</td> <td>(0x40000044, 0x10)</td> <td>-61.93</td> <td>23</td> <td>(0x40000044, 0x10)</td> <td>-60.44</td> <td>23</td>
	</tr>
    <!-- SIMON96 -->
    <tr>
	    <td rowspan="2">SIMON96</td>
	    <td>(0x100000, 0x444040)</td> <td>(0x10100, 0x4440)</td> <td>-92.2</td> <td>30</td> <td>(0x100000, 0x44040)</td> <td>-94.00</td> <td>32</td>
	</tr>
    <tr>
	    <td>(0x4000, 0x11101)</td> <td>(0x1101, 0x404)</td> <td>-95.34</td> <td>31</td> <td>(0x4000, 0x1101)</td> <td>-94.00</td> <td>32</td>
	</tr>
</table>
