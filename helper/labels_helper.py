import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset_collector import get_all_csv_files
from helper.common_methods import read_dictionary_from_file


def replace_in_array(numpy_array, element_to_replace, replacement):
    """
    Takes a numpy array and replace an element with replacement and returns numpy array
    :param numpy_array:
    :param element_to_replace:
    :param replacement:
    :return: numpy array
    """
    return np.where(numpy_array == element_to_replace, replacement, numpy_array)


def update_column_name_for_all_file_in_folder(folder_path, old_column_name, new_column_name):
    """
    Update all csv files header in a folder as provided
    :param folder_path:
    :param old_column_name:
    :param new_column_name:
    """
    files = get_all_csv_files(folder_path)
    print(files)
    for file_path in files:
        update_column_name_in_csv_file(file_path, old_column_name, new_column_name)


def update_column_name_in_csv_file(file_path, old_column_name, new_column_name):
    df = pd.read_csv(file_path)
    df.rename(columns={old_column_name: new_column_name}, inplace=True)
    df.to_csv(file_path, index=False)
    print("Updated: " + file_path)


def remove_first_column_in_csv_file(file_path):
    df = pd.read_csv(file_path)
    df = df.iloc[:, 1:]
    df.set_index("timestamps")
    df.to_csv(file_path, index=False)
    print("Updated: " + file_path)


def unpickle_result():
    """
    Converts results file to json in readable form
    """
    dictionary = read_dictionary_from_file("result/benchmark_result")
    with open("result/benchmark_result.json", 'w') as fp:
        json.dump(dictionary, fp, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def convert_ucr_txt_file_oadb_standard_csv(source_folder_path, destination_folder_path):
    path_list = Path(source_folder_path).rglob('*.txt')
    for path in path_list:
        # because path is object not string
        file_path = str(path)
        print(file_path)

        file_path_parts = file_path.split("_")
        file_path_parts.reverse()
        # added one to include last point also as anomaly
        anomaly_end = int(file_path_parts[0][:-4])
        anomaly_start = int(file_path_parts[1])
        # training_data_limit = file_path_parts[2]
        file_data = pd.read_csv(file_path, sep="\t", names=["value"])

        if file_data.size <= 1:
            row = file_data["value"][0]
            row = row.strip()
            row_list = row.split()
            # row_list = int(row_list)
            new_row_list = []
            for element in row_list:
                new_row_list.append(float(element))
            file_data = pd.DataFrame()
            file_data["value"] = new_row_list

        pre_anomaly = np.zeros(anomaly_start - 1)
        anomaly = np.ones((anomaly_end + 1) - anomaly_start)
        post_anomaly = np.zeros(file_data.size - anomaly_end)
        is_anomaly = np.concatenate((pre_anomaly, anomaly, post_anomaly))

        file_data["is_anomaly"] = is_anomaly

        file_path_parts = file_path.split("/")
        file_path_parts.reverse()
        file_name = file_path_parts[0][:-4] + ".csv"
        file_data.to_csv(destination_folder_path + file_name, index_label="timestamp")
