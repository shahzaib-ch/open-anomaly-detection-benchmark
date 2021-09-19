import numpy as np
import pandas as pd
import json

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
