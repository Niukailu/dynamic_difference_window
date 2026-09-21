''' misc '''
import os
import re
import numpy as np


def get_bit_by_num(experiment: str):
    ''' get_bit_by_num '''
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


def tuple_active_bit(tp):
    ''' tuple_active_bit '''
    l, r = tp
    return len(mask_to_list(l)) + len(mask_to_list(r))


def bits(array):
    ''' bits '''
    return np.array([bin(x).count('1') for x in array])


def clear_console():
    ''' clear the console '''
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'
    os.system(command)


def select_list(title, *args):
    ''' select_list '''
    print(title)
    for id, arg in enumerate(args):
        print(f'{id}.\t{arg}')
    command = None
    while command is None:
        try:
            command = int(input("请输入实验编号："))
            assert 0 <= command < len(args), '输入编号越界'
        except Exception as e:
            print(f'输入错误: [{e}]. 请重新输入！')
    return command
