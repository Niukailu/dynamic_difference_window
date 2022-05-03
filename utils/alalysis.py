''' analysis '''
import os
import math
import functools
import numpy as np
from .misc import get_bit_by_num, mask_to_list, tuple_active_bit, bits
from .zip import unzip


def load(experiment: str, step: int):
    ''' load '''
    root_path = f"experiments/{experiment}/{step}"
    bit = get_bit_by_num(experiment)

    # 解压缩实验文件
    unzip(root_path, '/tmp')
    # 加载基础信息
    with open(os.path.join('/tmp', root_path, "info.txt"), "r") as f:
        round = int(f.readline().strip())
        left_size, right_size = [int(v) for v in f.readline().strip().split()]
        left_base, left_mask, _ = [int(v) for v in f.readline().strip().split()]
        left_window = mask_to_list(left_mask)
        right_base, right_mask, _ = [int(v) for v in f.readline().strip().split()]
        right_window = mask_to_list(right_mask)
    # 加载左边
    with open(os.path.join('/tmp', root_path, "left"), "rb") as f:
        left = np.frombuffer(f.read(bit * left_size), dtype=eval(f'np.uint{bit*8}'))
    # 加载右边
    with open(os.path.join('/tmp', root_path, "right"), "rb") as f:
        right = np.frombuffer(f.read(bit * right_size), dtype=eval(f'np.uint{bit*8}'))
    # 加载概率
    with open(os.path.join('/tmp', root_path, "prob"), "rb") as f:
        data = f.read(8 * left_size * right_size)    # double
        prob = np.frombuffer(data, dtype=np.float64).reshape((left_size, right_size))
    # 删除tmp文件
    assert os.system(f'rm {os.path.join("/tmp", root_path)} -r') == 0, f'rm {os.path.join("/tmp", root_path)} -r'
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


def wait(func):
    ''' wait '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        input("\n按回车继续...")
    return wrapper


@wait
def print_window_info(info):
    ''' print_window_info '''
    print(f"\033[0;34;40m左窗口一共{len(info['left_window'])}位，为：{info['left_window']}\033[0m")
    print(f"\033[0;34;40m右窗口一共{len(info['right_window'])}位，为：{info['right_window']}\033[0m")


@wait
def print_base_info(info):
    ''' print_base_info '''
    print(f"\033[0;34;40m左窗口的base为：{bin(info['left_base'])}\033[0m")
    print(f"\033[0;34;40m右窗口的base为：{bin(info['right_base'])}\033[0m")


@wait
def print_max_info(info):
    ''' print_max_info '''
    max_prob = info['prob'].max()
    print(f"\033[0;34;40mMax: {max_prob:06f}\033[0m", end="")
    xs, ys = np.where(info['prob'] >= max_prob)
    pos = list(zip(xs, ys))
    pos.sort(key=tuple_active_bit)
    for x, y in zip(xs, ys):
        print(f"\033[0;34;40m ({info['left'][x]:#x},{info['right'][y]:#x})\033[0m", end="")
    print("")


@wait
def print_max_info_with_min_porb(info):
    ''' print_max_info_with_min_porb '''
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


@wait
def print_max_info_with_max_act(info):
    ''' print_max_info_with_max_act '''
    left_bits = bits(info['left'])
    right_bits = bits(info['right'])
    left_bits = np.tile(left_bits, (len(right_bits), 1)).T
    right_bits = np.tile(right_bits, (len(left_bits), 1))
    info_bits = left_bits + right_bits
    bit = int(input(f"已知最少活跃位数为{info_bits.min()}，求最多的活跃位数？："))
    map1, map2 = info_bits > bit, info_bits <= bit
    info_bits[map1] = -1000000000
    info_bits[map2] = 0
    info['prob'] = info['prob'] + info_bits
    max_prob = info['prob'].max()
    print(f"\033[0;34;40mMax: {max_prob:06f}\033[0m")
    xs, ys = np.where(info['prob'] >= max_prob + math.log2(0.999999))
    pos = list(zip(xs, ys))
    pos.sort(key=tuple_active_bit)
    for x, y in zip(xs, ys):
        print(f"\033[0;34;40m({info['left'][x]:#x},{info['right'][y]:#x})\033[0m", end=" ")
    print("")


@wait
def print_prob_info(info):
    ''' print_prob_info '''
    left = int(input("请输入left的值(16进制): "), 16)
    right = int(input("请输入right的值(16进制): "), 16)
    left = np.where(info['left'] == left)[0]
    right = np.where(info['right'] == right)[0]
    if len(left) == 0 or len(right) == 0:
        print("-Inf")
    else:
        print(info['prob'][left[0], right[0]])

@wait
def generate_log(experiment):
    ''' generate_log '''
    max_step = sorted([int(step[:-3]) for step in os.listdir(f"experiments/{experiment}") if step.endswith('.7z')])[-1]
    poss = []
    while True:
        command = input('是否继续添加监测点，输入(left, right)添加监测点，输入no停止: ')
        if command.upper() in ['N', 'NO']:
            break
        try:
            command = command.replace('(', '')
            command = command.replace(')', '')
            command = command.replace(',', ' ')
            command = command.replace('，', ' ')
            left, right = command.split()
            poss.append((int(left, 16), int(right, 16)))
        except Exception as e:
            print(f"输入不合法: {e}, 请继续输入")
    info = load(experiment, 0)
    print(f"\033[0;34;40mInput: ({info['left'][0]:#x}, {info['right'][0]:#x})\033[0m")
    print(f"\033[0;34;40m\tMax: {info['prob'][0,0]:06f} ({info['left'][0]:#x},{info['right'][0]:#x})\033[0m")
    for step in range(1, max_step+1):
        del info
        print(f"\033[0;34;40mRound {step}:\033[0m")
        info = load(experiment, step)
        print(f"\033[0;34;40m\twindow is {info['left_window']}\033[0m")
        max_prob = info['prob'].max()
        print(f"\033[0;34;40m\tMax: {max_prob:06f}\033[0m", end="")
        xs, ys = np.where(info['prob'] >= max_prob + math.log2(0.999999))
        for x, y in zip(xs, ys):
            print(f"\033[0;34;40m ({info['left'][x]:#x},{info['right'][y]:#x})\033[0m", end="")
        print("")
        for pos in poss:
            left = np.where(info['left'] == pos[0])[0]
            right = np.where(info['right'] == pos[1])[0]
            if len(left) == 0 or len(right) == 0:
                prob = -math.inf
            else:
                prob = info['prob'][left[0], right[0]]
            print(f"\033[0;34;40m\tprob[{hex(pos[0])}, {hex(pos[1])}] = {prob}\033[0m")
    print("\033[0;34;40mover.\033[0m")
    del info
