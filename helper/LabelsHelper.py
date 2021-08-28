import numpy as np
import pandas as pd

from data.dataset_collector import get_all_csv_files


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
    files = get_all_csv_files(folder_path)
    for file_path in files:
        update_column_name_in_csv_file(file_path, old_column_name, new_column_name)


def update_column_name_in_csv_file(file_path, old_column_name, new_column_name):
    df = pd.read_csv(file_path)
    df.rename(columns={old_column_name:new_column_name}, inplace=True)
    df.to_csv(file_path)
    print("Updated: " + file_path)
