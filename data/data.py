import pandas as pd
import numpy as np

DATA_DIRECTORY_PATH = "/Users/shahzaib/Documents/Thesis/datasets/" \
                      "ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_"
number_of_files = 1


def data_from_file(path):
    df = pd.read_csv(path)
    X = df.iloc[:, 1:2].to_numpy()
    y = df.iloc[:, 2:3].to_numpy()
    y = y.reshape(y.size)
    return X, y
