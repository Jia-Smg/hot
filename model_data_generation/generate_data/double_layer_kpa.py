from scipy.stats import qmc
import pandas as pd
import numpy as np
import mph
import time

import sys
sys.path.append("..") 
import utils

def train():
    # 可供调整的参数
    # up_height, down_height, up_pdms1_width, up_pdms2_width, down_pdms1_width, down_pdms2_width

    # 约束
    # up_height + down_height = 0.3, max(height) = 0.3, min(height) = 0.05
    # 0.02 < up_pdms1_width, up_pdms2_width, down_pdms1_width, down_pdms2_width  < 0.47

    # 生成样本
    # technique of generating sample: optimal latin hypercube 
    sampler = qmc.LatinHypercube(d=5) # 参数数量
    sample = sampler.random(n=100) # 样本数量

    l_bound = [0.01, 0.01, 0.01, 0.01, 0.01]
    u_bound = [0.275, 0.49, 0.49, 0.49, 0.49]
    sample_scaled = qmc.scale(sample, l_bound, u_bound)

    for sample in sample_scaled:
        up_half_height = round(sample[0], 3)
        up_pdms1_width = round(sample[1], 3)
        up_pdms2_width = round(sample[2], 3)
        up_pdms1_length = 0.99
        up_pdms2_length = 0.99
        down_pdms1_width = round(sample[3], 3)
        down_pdms2_width = round(sample[4], 3)
        down_pdms1_length = 0.99
        down_pdms2_length = 0.99

        up_position = up_half_height
        up_height = up_position * 2
        down_position = round(-(0.56 - up_height)/2, 3)
        down_height = round(-down_position*2, 3)

        q_horizontal = 1000
        q_vertical = 1000

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "几何参数", 
              up_height, up_position,  down_height, down_position, up_height/down_height, 
              up_pdms1_width, up_pdms2_width, up_pdms1_length, up_pdms2_length,
              down_pdms1_width, down_pdms2_width, down_pdms1_length, down_pdms2_length,
              q_horizontal, q_vertical)

        mph.option('session', 'stand-alone')
        client = mph.start(cores=1)
        client.caching(False)
        model = client.load('./models/double-layer-module_k400.mph')

        k_horizonal, k_vertical = calculate_kpa(model, up_height, up_position, up_pdms1_width, up_pdms2_width, up_pdms1_length, up_pdms2_length,
        down_height, down_position, down_pdms1_width, down_pdms2_width, down_pdms1_length, down_pdms2_length,
        q_horizontal, q_vertical)

        sys.stdout.flush()

        # to free up memory
        model.clear()
        model.reset()

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), 'kpa: ', k_horizonal, k_vertical, '热导率比例', round(k_horizonal/k_vertical, 3))
        # save data
        utils.add_to_csv(np.array([up_height, down_height, 
                                   up_pdms1_width, up_pdms2_width, up_pdms1_length, up_pdms2_length,
                                   down_pdms1_width, down_pdms2_width, down_pdms1_length, down_pdms2_length]), 
                         np.array([k_horizonal, k_vertical, round(k_horizonal/k_vertical, 3)]), 'temp/double_layer_model_k400.csv')


# 计算横纵热导率
def calculate_kpa(model, up_height, up_position, up_pdms1_width, up_pdms2_width, up_pdms1_length, up_pdms2_length,
                  down_height, down_position, down_pdms1_width, down_pdms2_width, down_pdms1_length, down_pdms2_length,
                  q_horizontal, q_vertical):
    model = model.java

    try:
        # 上层-横
        # bk1
        model.component("comp1").geom("geom1").feature("blk1").set("base", "center");
        model.component("comp1").geom("geom1").feature("blk1").set("size", ["1", "1", str(up_height)]);
        model.component("comp1").geom("geom1").feature("blk1").set("pos", [".5", ".5", str(up_position)]);
        # bk2
        model.component("comp1").geom("geom1").feature("blk2").set("base", "center");
        model.component("comp1").geom("geom1").feature("blk2").set("size", [str(up_pdms1_length), str(up_pdms1_width), str(up_height)]);
        model.component("comp1").geom("geom1").feature("blk2").set("pos", [".5", ".25", str(up_position)]);
        # bk3
        model.component("comp1").geom("geom1").feature("blk3").set("base", "center");
        model.component("comp1").geom("geom1").feature("blk3").set("size", [str(up_pdms2_length), str(up_pdms2_width), str(up_height)]);
        model.component("comp1").geom("geom1").feature("blk3").set("pos", ["0.5", ".75", str(up_position)]);

        # 下层-纵
        # bk4
        model.component("comp1").geom("geom1").feature("blk4").set("base", "center");
        model.component("comp1").geom("geom1").feature("blk4").set("size", ["1", "1", str(down_height)]);
        model.component("comp1").geom("geom1").feature("blk4").set("pos", ["0.5", ".5", str(down_position)]);
        # bk5
        model.component("comp1").geom("geom1").feature("blk5").set("base", "center");
        model.component("comp1").geom("geom1").feature("blk5").set("size", [str(down_pdms1_width), str(down_pdms1_length), str(down_height)]);
        model.component("comp1").geom("geom1").feature("blk5").set("pos", [".25", ".5", str(down_position)]);
        # bk6
        model.component("comp1").geom("geom1").feature("blk6").set("base", "center");
        model.component("comp1").geom("geom1").feature("blk6").set("size", [str(down_pdms2_width), str(down_pdms2_length), str(down_height)]);
        model.component("comp1").geom("geom1").feature("blk6").set("pos", [".75", ".5", str(down_position)]);
        # 设置横向传热
        model.component("comp1").physics("ht").feature("hf1").active(True)
        model.component("comp1").physics("ht").feature("temp1").active(True)
        model.component("comp1").physics("ht").feature("hf2").active(False)
        model.component("comp1").physics("ht").feature("temp2").active(False)
        # 设置热通量（横）
        model.component("comp1").physics("ht").feature("hf1").set("q0", str(q_horizontal));
        # 获取温度并计算横向热导率
        model.study("std1").run();
        horizontal_max_temp = model.result().table("tbl1").getRealRow(0)[0]
        horizontal_min_temp = model.result().table("tbl1").getRealRow(0)[1]
        k_horizontal = q_horizontal / (horizontal_max_temp - horizontal_min_temp)

        # 更换为纵向传热
        model.component("comp1").physics("ht").feature("hf1").active(False)
        model.component("comp1").physics("ht").feature("temp1").active(False)
        model.component("comp1").physics("ht").feature("hf2").active(True)
        model.component("comp1").physics("ht").feature("temp2").active(True)
        # 设置热通量（纵）
        model.component("comp1").physics("ht").feature("hf2").set("q0", str(q_vertical));
        # 获取温度并计算纵向热导率
        model.study("std1").run();
        vertical_max_temp = model.result().table("tbl1").getRealRow(0)[2]
        vertical_min_temp = model.result().table("tbl1").getRealRow(0)[3]
        k_vertical = q_vertical / (vertical_max_temp - vertical_min_temp)


        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), '横向温度:', round(horizontal_max_temp,3), round(horizontal_min_temp,3), 
        '纵向温度:', round(vertical_max_temp,3), round(vertical_min_temp,3))

        return round(k_horizontal, 3), round(k_vertical, 3)
    except:
        print("上层", up_height, up_position, up_pdms1_width, up_pdms2_width, "下层", down_height, down_position, down_pdms1_width, down_pdms2_width)


if __name__ == '__main__':
    train()