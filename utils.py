import pandas as pd
import numpy as np

def add_to_csv(arr_data1, arr_data2, path):
    content = np.append(arr_data1, arr_data2)
    content = content.reshape(1, -1)
    data = pd.DataFrame(content)
    data.to_csv(path, mode='a', header=False)
