from sko.GA import GA
import torch
import torch.nn as nn
import numpy as np
import time
import sys

sys.path.append("..") 
import utils

sys.path.append("../train_model") 

sys.path.append("../model_data_generation")
from cross_shape_kpa import calculate_k_vertical, calculate_k_horizontal, q0, d

last_Time = None # 初始化上次计算的时间
total_runtime = 0.0 # 初始化总的运行时长
iteration_count = 0 # 初始化迭代次数


target_k_horizontal = 58
target_k_vertical = 368


def worker(d_param):
    d1 = d_param[0]
    d3 = d_param[1]
    d2 = d-d1-d3
    d4 = d_param[2]
    d6 = d_param[3]
    d5 = d-d4-d6
    d7 = d_param[4]
    d9 = d_param[5]
    d8 = d-d7-d9
    d10 = d_param[6]
    d12 = d_param[7]
    d11 = d-d10-d12

    full_samples = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d]
    
    k_horizontal = calculate_k_horizontal('../../model_data_generation/models/cross-shape-diff-horizontal.mph', full_samples, q0)
    k_vertical = calculate_k_vertical('../../model_data_generation/models/cross-shape-diff-vertical.mph',full_samples, q0)

    error = 100 * np.sqrt((k_vertical - target_k_vertical) ** 2 + (k_horizontal - target_k_horizontal) ** 2)

    global last_Time, total_runtime, iteration_count

    iteration_count += 1
    current_time = time.time()
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))

    if last_Time is None:
        time_diff = "首次运行"
        runtime_ms = 0.0 # 首次运行，时间差为0
    else:
        runtime_ms = (current_time - last_Time) * 1000
        time_diff = f"{runtime_ms:.2f} 毫秒"
        total_runtime += runtime_ms

    print(current_time_str, "迭代次数：", iteration_count, "本次计算时长：",time_diff, error, k_vertical, k_horizontal, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12)
    utils.add_to_csv(np.array([error, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12]), np.array([k_vertical, k_horizontal]), 'log/get_target_kap_comsol.csv')

    last_Time = current_time #更新时间

    return error


if __name__ == '__main__':
    ga = GA(func = worker, n_dim = 8, size_pop = 300, max_iter=50, prob_mut=0.01, lb=0.01, ub=0.49, precision=1e-5)
    best_x, best_y = ga.run()
    print('best_x, best_y: ', best_x, best_y)
    print("平均计算时长（毫秒）：", round(total_runtime / iteration_count, 2), "总耗时（毫秒）：", total_runtime, "总迭代次数（次数）：", iteration_count)

    d = 1
    d1 = best_x[0]
    d3 = best_x[1]
    d2 = d-d1-d3
    d4 = best_x[2]
    d6 = best_x[3]
    d5 = d-d4-d6
    d7 = best_x[4]
    d9 = best_x[5]
    d8 = d-d7-d9
    d10 = best_x[6]
    d12 = best_x[7]
    d11 = d-d10-d12

    full_samples = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d]

    k_horizontal = calculate_k_horizontal('../model_data_generation/models/cross-shape-diff-horizontal.mph', full_samples, q0)
    k_vertical = calculate_k_vertical('../model_data_generation/models/cross-shape-diff-vertical.mph',full_samples, q0)

    print(full_samples, k_horizontal, k_vertical)