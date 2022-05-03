''' help '''
import os
import sys
sys.path.append(sys.path[0].replace('/scripts', ''))
from utils import clear_console, select_list   # pylint: disable=wrong-import-position
from utils.alalysis import load, generate_log, print_window_info, print_base_info, print_max_info, print_max_info_with_min_porb, print_max_info_with_max_act, print_prob_info  # pylint: disable=wrong-import-position


def choice_experiment_step_action(experiment, step):
    ''' choice_experiment_step_action '''
    clear_console()
    command = select_list(
        f"当前实验：\033[0;32;40m{experiment}\033[0m，正在分析第\033[0;32;40m{step}\033[0m步",
        "\033[0;31;40m返回上一级\033[0m",
        "输出窗口",
        "输出base",
        "输出概率最大值，活跃位从少到多排序",
        "输出概率最大值附近，活跃位从少到多排序",
        "输出概率最大值，活跃位从少到多排序(限定活跃位数)",
        "查询固定位置概率",
    )
    functions = [None, print_window_info, print_base_info, print_max_info, print_max_info_with_min_porb, print_max_info_with_max_act, print_prob_info]
    action_function = functions[command]
    return command, action_function


def choice_experiment_step(experiment):
    ''' choice experiment step '''
    clear_console()
    steps = sorted([int(step[:-3]) for step in os.listdir(f"experiments/{experiment}") if step.endswith('.7z')])
    steps_info = [f'第{step}步' for step in steps[1:]]
    command = select_list(
        f"当前实验：\033[0;32;40m{experiment}\033[0m，共运行\033[0;32;40m{steps[-1]}\033[0m步",
        "\033[0;31;40m返回上一级\033[0m",
        *steps_info
    )
    return command


def choice_experiment_action(experiment):
    ''' choice experiment action '''
    clear_console()
    command = select_list(
        f"当前实验：\033[0;32;40m{experiment}\033[0m",
        "\033[0;31;40m返回上一级\033[0m",
        "生成实验日志",
        "单独分析某步结果",
    )
    return command


def choice_experiments():
    ''' choice experiments '''
    clear_console()
    experiments = sorted(os.listdir('experiments/'))
    command = select_list('实验目录：', '\033[0;31;40m退出\033[0m', *experiments)
    command = None if command == 0 else experiments[command - 1]
    return command


def main():
    ''' main '''
    experiment = choice_experiments()
    while experiment is not None:
        experiment_action = choice_experiment_action(experiment)
        while experiment_action > 0:
            if experiment_action == 1:
                generate_log(experiment)
            elif experiment_action == 2:
                step = choice_experiment_step(experiment)
                while step != 0:
                    info = load(experiment, step)
                    step_action, action_function = choice_experiment_step_action(experiment, step)
                    while step_action != 0:
                        action_function(info)
                        step_action, action_function = choice_experiment_step_action(experiment, step)
                    step = choice_experiment_step(experiment)
            experiment_action = choice_experiment_action(experiment)
        experiment = choice_experiments()


if __name__ == '__main__':
    main()
