import re

def extract_and_analyze_time_intervals_trimmed(filename, output_filename, num_intervals=10, trim_count=10):
    """
    从文件中提取每次迭代的计算时长，去掉指定数量的最大值和最小值，
    计算平均耗时，将耗时分成指定数量的区间，统计每个区间的耗时数量，
    并将结果存储到另一个文件中。

    Args:
        filename (str): 包含数据的文件的路径。
        output_filename (str): 用于存储提取的计算时长、平均耗时和区间统计结果的文件路径。
        num_intervals (int): 要创建的区间的数量 (默认为 10)。
        trim_count (int): 要去除的最大值和最小值的数量 (默认为 10)。

    Returns:
        bool: True 如果成功提取、计算、分析并保存了数据，False 否则。
    """
    iteration_times = []
    try:
        with open(filename, 'r', encoding='utf-8') as infile:
            for line in infile:
                # 使用正则表达式匹配包含 "迭代次数" 和 "本次计算时长" 的行
                match = re.search(r"迭代次数：\s*(\d+)\s*本次计算时长：\s*([\d.]+)\s*毫秒", line)  # 匹配毫秒单位
                if match:
                    iteration_number = int(match.group(1))  # 提取迭代次数 (可选，如果需要)
                    time_ms = float(match.group(2))  # 提取以毫秒为单位的时长
                    iteration_times.append(time_ms)
                else:
                    # 匹配 "首次运行" 的情况
                    match = re.search(r"迭代次数：\s*(\d+)\s*本次计算时长：\s*首次运行\s*([\d.]+)", line)
                    if match:
                        iteration_number = int(match.group(1))
                        time_ms = float(match.group(2))
                        iteration_times.append(time_ms)

        # 去除最大值和最小值
        if len(iteration_times) > 2 * trim_count:
            sorted_times = sorted(iteration_times)
            trimmed_times = sorted_times[trim_count:-trim_count]
        else:
            trimmed_times = []  # 如果数据太少，则不进行裁剪

        # 计算平均耗时 (基于裁剪后的数据)
        if trimmed_times:
            average_time = sum(trimmed_times) / len(trimmed_times)
            min_time = min(trimmed_times)
            max_time = max(trimmed_times)

            # 创建区间
            interval_size = (max_time - min_time) / num_intervals
            intervals = [(min_time + i * interval_size, min_time + (i + 1) * interval_size) for i in range(num_intervals)]

            # 统计每个区间的耗时数量
            interval_counts = [0] * num_intervals
            for time in trimmed_times:
                for i, (start, end) in enumerate(intervals):
                    if start <= time <= end:
                        interval_counts[i] += 1
                        break  # 找到所属区间后，跳出内层循环

        else:
            average_time = 0
            min_time = 0
            max_time = 0
            intervals = []
            interval_counts = []

        # 将结果写入输出文件
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            outfile.write("原始数据:\n")
            for i, time in enumerate(iteration_times):
                outfile.write(f"迭代 {i+1}: {time:.2f} 毫秒\n")

            outfile.write("\n去除异常值后的数据:\n")
            if trimmed_times:
                for time in trimmed_times:
                    outfile.write(f"{time:.2f} 毫秒\n")
            else:
                outfile.write("数据量过少，无法去除异常值。\n")

            outfile.write(f"\n平均耗时: {average_time:.2f} 毫秒 (去除异常值后)\n")
            outfile.write(f"最短耗时: {min_time:.2f} 毫秒 (去除异常值后)\n")
            outfile.write(f"最长耗时: {max_time:.2f} 毫秒 (去除异常值后)\n")
            outfile.write("\n耗时区间统计 (去除异常值后):\n")
            for i, (start, end) in enumerate(intervals):
                outfile.write(f"区间 {i+1}: [{start:.2f}, {end:.2f}] - 数量: {interval_counts[i]}\n")

        return True  # 成功提取、计算、分析并保存

    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 未找到。")
        return False
    except Exception as e:
        print(f"发生错误：{e}")
        return False

# 示例用法
input_filename = '../log/log_get_target_kap_comsol'  # 替换为你的输入文件名
output_filename = 'log_get_target_kap_comsol_time_consumption.txt'  # 替换为你的输出文件名
num_intervals = 10  # 区间数量
trim_count = 10  # 去除的最大值和最小值的数量

if extract_and_analyze_time_intervals_trimmed(input_filename, output_filename, num_intervals, trim_count):
    print(f"数据已成功提取、计算、分析并保存到 '{output_filename}'。")
else:
    print("提取数据失败。")