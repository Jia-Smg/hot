import pandas as pd
import numpy as np
from sko.GA import GA

k_vertical = 233
k_horizontal = 50


def get_error(row):
    error = np.sqrt((row[13] - k_vertical) ** 2 + (row[14] - k_horizontal) ** 2)
    return error


def get_k_vertical_error(row):
    error = abs(row[14] - k_vertical)
    return error


def get_k_horizontal_error(row):
    error = abs(row[13] - k_horizontal)
    return error


def select_k_couple():
    df = pd.read_csv('./generated_data.csv', header=None)
    errors = df.apply(get_error, axis=1)
    min_error_index = errors.idxmin()

    min_error_row = df.loc[min_error_index, :]

    print(min_error_row)


def select_best_k_vertical():
    df = pd.read_csv('./generated_data.csv', header=None)
    error = 0
    row_num = 0

    for index, row in df.iterrows():
        if index == 0:
            error = abs(row[13] - k_vertical)
        
        current_error = abs(row[13] - k_vertical)
        if current_error < error:
            error = current_error
            row_num = index
    
    print(error, df.iloc[row_num, :])
        
    return error, row_num


def select_best_k_horizontal():
    df = pd.read_csv('./generated_data.csv', header=None)
    errors = df.apply(get_k_horizontal_error, axis=1)
    min_error_index = errors.idxmin()

    min_error_row = df.loc[min_error_index, :]

    print(min_error_row)


def analyze_column():
    df = pd.read_csv('data/high_kpa_data.csv', header=None)
    # df = pd.read_csv('data.csv', header=None)
    col_range = df.iloc[:, 13].describe()

    print(col_range)

if __name__ == '__main__':
    analyze_column()
    # select_best_k_vertical()
    # select_best_k_horizontal()
    # select_k_couple()
