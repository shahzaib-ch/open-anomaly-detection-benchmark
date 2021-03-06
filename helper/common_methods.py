import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def read_dictionary_from_file(file_path):
    """
    Returns dictionary after reading from file
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_dictionary_to_file(dictionary, file_path):
    """
    Saves dictionary to file
    """
    with open(file_path, 'wb+') as f:
        pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)


def round_xy_coordinates(x, y):
    """
    round float, example -0.5 to 0.5 to 0
    """
    x_rounded = int(round(x))
    y_rounded = int(round(y))
    return x_rounded, y_rounded


def list_of_all_files_in_folder_and_subfolders(path):
    # we shall store all the file names in this list
    file_list = []

    for root, dirs, files in os.walk(path):
        for file in files:
            # append the file name to the list
            if file == '.DS_Store':
                continue
            file_list.append(os.path.join(root, file))

    return file_list


def add_date_time_index_to_df(input_instances, index_start="2021-10-15 00:00:00"):
    date_time_index = pd.date_range(index_start, periods=input_instances.size, freq="S")
    input_instances = input_instances.set_index(date_time_index)
    return input_instances


def list_contains(dataset_path, not_supported_datasets):
    if dataset_path in not_supported_datasets:
        return True

    for not_supported_dataset_path in not_supported_datasets:
        if not_supported_dataset_path in dataset_path:
            return True

    return False


def convert_data_frame_to_float(df):
    for column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], downcast='float')

    return df


def standardize_data(data):
    """
    between 0 and 1
    """
    scalar = MinMaxScaler()
    reshaped_data = data.reshape(-1, 1)
    scalar.fit(reshaped_data)
    return scalar.transform(reshaped_data).reshape(1, -1)[0]


def get_2nd_value_from_list(scores_list):
    anomaly_scores = []
    for i in range(len(scores_list)):
        anomaly_scores.append(scores_list[i, 1])

    return np.asarray(anomaly_scores)
