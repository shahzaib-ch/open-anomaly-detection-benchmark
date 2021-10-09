import numpy as np
from sklearn.metrics import accuracy_score

from visualizer.result_data_keys import ResultDataKey


def add_accuracy_to_df(result_data_frame):
    """
    Adds accuracy column in data frame of results
    """
    result_data_frame[ResultDataKey.accuracy] = result_data_frame.apply(lambda row: calculate_accuracy_score(row),
                                                                        axis=1)
    return result_data_frame


def calculate_accuracy_score(row):
    labels = np.concatenate((row[ResultDataKey.labels_train], row[ResultDataKey.labels_test]))
    labels_detected = row[ResultDataKey.labels_detected]
    return accuracy_score(labels, labels_detected)
