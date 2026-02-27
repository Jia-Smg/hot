from scipy.stats import qmc
import pandas as pd
import numpy as np
import mph

import sys
sys.path.append("../..") 
import utils


def calculate_k_vertical(model_path, sample, q0):
    mph.option('session', 'stand-alone')
    client = mph.start(cores=1)
    client.caching(False)
    model = client.load(model_path)

    max_temperature, min_termpature = get_termperature(model, sample)
    print("max_temperature, min_termpature: " + str(max_temperature) + " " + str(min_termpature))
    sys.stdout.flush()

    # to free up memory
    model.clear()
    model.reset()

    d = sample[12]
    d5 = sample[4]
    k_vertical =  q0 * d * d5 / (max_temperature - min_termpature)

    return k_vertical


def calculate_k_horizontal(model_path, sample, q0):
    mph.option('session', 'stand-alone')
    client = mph.start(cores=1)
    client.caching(False)
    model = client.load(model_path)

    max_temperature, min_termpature = get_termperature(model, sample)
    print("max_temperature, min_termpature: " + str(max_temperature) + " " + str(min_termpature))
    sys.stdout.flush()

    # to free up memory
    model.clear()
    model.reset()

    d = sample[12]
    d2 = sample[1]
    k_horizontal =  q0 * d * d2/ (max_temperature - min_termpature)
    return k_horizontal


def get_termperature(model, sample):
    model = model.java

    model.param().set("d1", str(sample[0]) + "[m]")
    model.param().set("d2", str(sample[1]) + "[m]")
    model.param().set("d3", str(sample[2]) + "[m]")
    model.param().set("d4", str(sample[3]) + "[m]")
    model.param().set("d5", str(sample[4]) + "[m]")
    model.param().set("d6", str(sample[5]) + "[m]")
    model.param().set("d7", str(sample[6]) + "[m]")
    model.param().set("d8", str(sample[7]) + "[m]")
    model.param().set("d9", str(sample[8]) + "[m]")
    model.param().set("d10", str(sample[9]) + "[m]")
    model.param().set("d11", str(sample[10]) + "[m]")
    model.param().set("d12", str(sample[11]) + "[m]")
    model.param().set("d", "1[m]")

    model.study("std1").run()

    max_temperature = model.result().table("tbl1").getRealRow(0)[0]
    min_termpature = model.result().table("tbl1").getRealRow(0)[1]

    return max_temperature, min_termpature

q0 = 10000
d = 1

def train_kpa_data():
    # technique of generating sample: optimal latin hypercube 
    sampler = qmc.LatinHypercube(d=8)
    sample = sampler.random(n=200)

    l_bound = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    u_bound = [0.49, 0.49, 0.49, 0.49, 0.49, 0.49, 0.49, 0.49]
    sample_scaled = qmc.scale(sample, l_bound, u_bound)

    for sample in sample_scaled:
        # low kpa
        # d1 = sample[0]
        # d2 = sample[1]
        # d3 = d-d1-d2
        # d4 = sample[2]
        # d5 = sample[3]
        # d6 = d-d4-d5
        # d7 = sample[4]
        # d8 = sample[5]
        # d9 = d-d7-d8
        # d10 = sample[6]
        # d11 = sample[7]
        # d12 = d-d10-d11

        # high kpa（在生成较大的kpa时也能有好的表现）
        d1 = sample[0]
        d3 = sample[1]
        d2 = d-d1-d3
        d4 = sample[2]
        d6 = sample[3]
        d5 = d-d4-d6
        d7 = sample[4]
        d9 = sample[5]
        d8 = d-d7-d9
        d10 = sample[6]
        d12 = sample[7]
        d11 = d-d10-d12
        full_samples = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d]

        k_vertical = calculate_k_vertical('../models/cross-shape-diff-vertical.mph', full_samples, q0)
        k_horizontal = calculate_k_horizontal('../models/cross-shape-diff-horizontal.mph', full_samples, q0)

        # save data
        utils.add_to_csv(np.array([d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12]), np.array([k_vertical, k_horizontal]), 'temp/cross_shape/high_kpa_data_test.csv')


if __name__ == '__main__':
    train_kpa_data()