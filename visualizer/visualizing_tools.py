import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from helper.common_methods import read_dictionary_from_file

"""
Result format is:
{
    detector name: {
        [
             dataset_file_path: {
                "dataset_name": dataset_name,
                "detector_name": detector_name,
                "data": {
                    "input_instances_train": input_instances_train,
                    "input_instances_test": input_instances_test,
                    "labels_train": labels_train,
                    "labels_test": labels_test,
                    "labels_detected": complete_detected_labels
                }
            },
            ...
        },
    ....
}
"""


def get_result_data_as_data_frame():
    result_data_array = []
    dictionary = read_dictionary_from_file("result/benchmark_result")
    for detector_name, file_results in dictionary.items():
        for file_result in file_results:
            for file_path, result_of_file in file_result.items():
                data = result_of_file["data"]
                input_instances_train = data["input_instances_train"]
                input_instances_test = data["input_instances_test"]
                labels_train = data["labels_train"]
                labels_test = data["labels_test"]
                labels_detected = data["labels_detected"]
                if labels_detected.size != np.concatenate((labels_train, labels_test)).size:
                    raise ValueError("labels of dataset and detected labels are not same for file: " +
                                     file_path + " and detector: " + detector_name)
                """visualize(input_instances_train, input_instances_test, labels_train, labels_test,
                          result_of_file_list[2], file_path, detector_name)"""
                result_data_frame_row = __convert_to_result_data_frame_row(input_instances_train, input_instances_test,
                                                                         labels_train, labels_test, labels_detected,
                                                                         file_path, detector_name)
                result_data_array.append(result_data_frame_row)
    return __make_result_data_frame(result_data_array)


def __transform_result_data_frame_for_sns_heat_map(result_data_frame):
    heat_map_data = result_data_frame.loc[:, ("detector", "dataset file")]
    heat_map_data["accuracy"] = result_data_frame.apply(lambda row: __calculate_accuracy_score(row), axis=1)
    heat_map_data = heat_map_data.pivot(index="dataset file", columns="detector")
    return heat_map_data


"""
def visualize(input_instances_train, input_instances_test, labels_train, labels_test, labels_detected, file_name,
              algo_name):
    labels_test = pd.DataFrame(labels_test).to_numpy().reshape(labels_test.size)
    labels_train = pd.DataFrame(labels_train).to_numpy().reshape(labels_train.size)
    labels = np.concatenate((labels_train, labels_test))

    x_axis = np.arange(labels.size)
    plt.plot(x_axis, labels, 'go')
    plt.plot(x_axis, labels_detected, 'ro')
    plt.show()

    print("Data file: ", file_name)
    print("Algorithm Name: ", algo_name)
    print("Acurracy: ", accuracy_score(labels, labels_detected))
    print("Matrix = tn, fp, fn, tp: ", confusion_matrix(labels, labels_detected).ravel())
    
"""


def __convert_to_result_data_frame_row(input_instances_train, input_instances_test, labels_train, labels_test,
                                     labels_detected, file_path, detector_name):
    return [detector_name, file_path, input_instances_train, input_instances_test, labels_train, labels_test,
            labels_detected]


def __make_result_data_frame(result_data_array):
    return pd.DataFrame(result_data_array, columns=["detector", "dataset file", "training data", "test data",
                                                    "training data labels", "test data labels",
                                                    "detected labels"])


def show_full_detailed_result_heat_map():
    result_data_frame = get_result_data_as_data_frame()
    heat_map_df = __transform_result_data_frame_for_sns_heat_map(result_data_frame)
    r = sns.heatmap(heat_map_df, cmap='BuPu')
    r.set_title("Heatmap of algorithm accuracy on Yahoo dataset")
    plt.show()


def __calculate_accuracy_score(row):
    labels = np.concatenate((row["training data labels"], row["test data labels"]))
    labels_detected = row["detected labels"]
    return accuracy_score(labels, labels_detected)
