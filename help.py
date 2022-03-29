import os
import re
import math
import numpy as np


def get_bit_by_num(experiment: str):
    # 假设严格按照命名规范
    experiment = experiment.replace("SIMON", "").replace("SIMECK", "")
    if experiment.startswith("32"):    # uint16
        return 2
    if experiment.startswith("48") or experiment.startswith("64"):  # uint32
        return 4
    if experiment.startswith("96") or experiment.startswith("128"):  # uint64
        return 8


def mask_to_list(mask):
    ''' mask to list '''
    mask = re.findall(r'.{1}', bin(mask)[2:])[::-1]
    window = []
    for idx, bit in enumerate(mask):
        if bit == '1':
            window.append(idx)
    return window


def load(experiment: str, step: int):
    ''' load '''
    root_path = f"experiments/{experiment}/{step}/"
    bit = get_bit_by_num(experiment)
    # 加载基础信息
    with open(os.path.join(root_path, "info.txt"), "r") as f:
        round = int(f.readline().strip())
        left_size, right_size = [int(v) for v in f.readline().strip().split()]
        left_base, left_mask, _ = [int(v) for v in f.readline().strip().split()]
        left_window = mask_to_list(left_mask)
        right_base, right_mask, _ = [int(v) for v in f.readline().strip().split()]
        right_window = mask_to_list(right_mask)
    # 加载左边
    with open(os.path.join(root_path, "left"), "rb") as f:
        left = np.frombuffer(f.read(bit * left_size), dtype=eval(f'np.uint{bit*8}'))
    # 加载右边
    with open(os.path.join(root_path, "right"), "rb") as f:
        right = np.frombuffer(f.read(bit * right_size), dtype=eval(f'np.uint{bit*8}'))
    # 加载概率
    with open(os.path.join(root_path, "prob"), "rb") as f:
        data = f.read(8 * left_size * right_size)    # double
        prob = np.frombuffer(data, dtype=np.float64).reshape((left_size, right_size))
    return {
        "round": round,
        "left": left,
        "right": right,
        "prob": prob,
        "left_window": left_window,
        "right_window": right_window,
        "left_base": left_base,
        "right_base": right_base,
    }


def clear_console():
    ''' clear the console '''
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'
    os.system(command)


def choice_experiments():
    ''' choice experiments '''
    clear_console()
    print("实验目录：")
    experiments = os.listdir('experiments/')
    # print(f"0.\t退出")
    print(f"0.\t\033[0;31;40m退出\033[0m")
    for idx, experiment in enumerate(experiments):
        print(f"{idx+1:d}.\t{experiment}")
    command = int(input("请输入实验编号："))
    if(command == 0):
        exit()
    return experiments[command - 1]


def generate_log(experiment, max_step):
    clear_console()
    info = load(experiment, 0)
    print(f"\033[0;34;40mInput: ({info['left'][0]:#x}, {info['right'][0]:#x})\033[0m")
    print(f"\033[0;34;40mMax: {info['prob'][0,0]:06f} ({info['left'][0]:#x}, {info['right'][0]:#x})\033[0m")
    for step in range(1, max_step+1):
        del info
        print("==============")
        print(f"\033[0;34;40mRound: {step}\033[0m")
        info = load(experiment, step)
        print(f"\033[0;34;40mwindow is {info['left_window']}\033[0m")
        max_prob = info['prob'].max()
        print(f"\033[0;34;40mMax: {max_prob:06f}\033[0m", end="")
        xs, ys = np.where(info['prob'] >= max_prob + math.log2(0.999999))
        for x, y in zip(xs, ys):
            print(f"\033[0;34;40m ({info['left'][x]:#x},{info['right'][y]:#x})\033[0m", end="")
        print("")
    del info
    input("\n按回车继续...")


def tuple_active_bit(tp):
    l, r = tp
    return len(mask_to_list(l)) + len(mask_to_list(r))


def analysis_experiment(experiment, step):
    ''' analysis_experiment '''
    clear_console()
    print(f"当前实验：\033[0;32;40m{experiment}\033[0m，正在分析第\033[0;32;40m{step}\033[0m步")
    print("0.\t\033[0;31;40m返回上一级\033[0m")
    print("1.\t输出窗口")
    print("2.\t输出base")
    print("3.\t输出概率最大值，活跃位从少到多排序")
    print("4.\t输出概率最大值附近，活跃位从少到多排序")
    command = int(input("请输入实验编号："))
    if command == 0:
        return False
    elif command == 1:
        info = load(experiment, step)
        print(f"\033[0;34;40m左窗口一共{len(info['left_window'])}位，为：{info['left_window']}\033[0m")
        print(f"\033[0;34;40m右窗口一共{len(info['right_window'])}位，为：{info['right_window']}\033[0m")
    elif command == 2:
        info = load(experiment, step)
        print(f"\033[0;34;40m左窗口的base为：{bin(info['left_base'])}\033[0m")
        print(f"\033[0;34;40m右窗口的base为：{bin(info['right_base'])}\033[0m")
    elif command == 3:
        info = load(experiment, step)
        max_prob = info['prob'].max()
        print(f"\033[0;34;40mMax: {max_prob:06f}\033[0m", end="")
        xs, ys = np.where(info['prob'] >= max_prob)
        pos = list(zip(xs, ys))
        pos.sort(key=tuple_active_bit)
        for x, y in zip(xs, ys):
            print(f"\033[0;34;40m ({info['left'][x]:#x},{info['right'][y]:#x})\033[0m", end="")
        print("")
    elif command == 4:
        info = load(experiment, step)
        max_prob = info['prob'].max()
        print(f"\033[0;34;40mMax: {max_prob:06f}\033[0m")
        p = float(input("寻找最大值周围多少？（\033[0;32;40m[0, 1]之间代表max_prob * input，小于0代表真实概率下界）\033[0m："))
        if 0 <= p <= 1:
            xs, ys = np.where(info['prob'] >= max_prob + math.log2(p))
        else:
            xs, ys = np.where(info['prob'] >= math.log2(p))
        pos = list(zip(xs, ys))
        pos.sort(key=tuple_active_bit)
        for x, y in zip(xs, ys):
            print(f"\033[0;34;40m({info['left'][x]:#x},{info['right'][y]:#x})\033[0m", end=" ")
        print("")
    else:
        print("暂时还没有那么多功能")
    input("\n按回车继续...")
    return True
    


def choice_experiment_action(experiment):
    ''' choice experiment action '''
    clear_console()
    steps = sorted([int(step) for step in os.listdir(f"experiments/{experiment}") if step != "info.log"])
    print(f"当前实验：\033[0;32;40m{experiment}\033[0m，共运行\033[0;32;40m{steps[-1]}\033[0m步")
    print("0.\t\033[0;31;40m返回上一级\033[0m")
    print("1.\t生成实验日志")
    print(f"2.\t单独分析某步结果（\033[0;32;40m请输入2-x来选择，如2-{steps[-1]}\033[0m）")
    command = input("请输入实验编号：")
    if command == '0':
        return -2
    if command == '1':
        generate_log(experiment, steps[-1])
        return -1
    return int(command[2:])
        
    
def main():
    while True:
        experiment = choice_experiments()
        while True:
            step = choice_experiment_action(experiment)
            if step == -2:
                break
            if step >= 0:
                ctn = True
                while ctn:
                    ctn = analysis_experiment(experiment, step)


if __name__ == '__main__':
    main()
