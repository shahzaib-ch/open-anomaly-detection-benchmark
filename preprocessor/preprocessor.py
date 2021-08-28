import pandas as pd


def get_data_from_file(path):
    df = pd.read_csv(path)
    X = df.iloc[:, 1:2].to_numpy()
    y = df.iloc[:, 2:3].to_numpy()
    y = y.reshape(y.size)
    return X, y
