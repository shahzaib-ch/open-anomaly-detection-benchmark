import numpy as np
from sklearn.metrics import accuracy_score

from visualizer.visualizing_tools import get_result_data_as_data_frame


def calculate_accuracy_score(row):
    labels = np.concatenate((row["training data labels"], row["test data labels"]))
    labels_detected = row["detected labels"]
    if labels_detected.size != labels.size:
        raise ValueError("labels of dataset and detected labels are not same for file: " +
                         row["dataset file"] + " and detector: " + row["detector"])
    return accuracy_score(labels, labels_detected)


result_data_frame = get_result_data_as_data_frame()
heat_map_data = result_data_frame[["detector", "dataset file"]]
heat_map_data["accuracy"] = result_data_frame.apply(lambda row: calculate_accuracy_score(row), axis=1)
print(heat_map_data)
