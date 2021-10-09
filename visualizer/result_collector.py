import numpy as np
import pandas as pd

from helper.common_methods import read_dictionary_from_file
from visualizer.result_data_keys import ResultDataKey
from visualizer.result_metric_calculators import calculate_accuracy_score

"""
Result format is:
{
    detector name: {
        [
             dataset_file_path: {
                ResultDataKey.dataset_name: dataset_name,
                ResultDataKey.detector_name: detector_name,
                "data": {
                    "input_instances_train": input_instances_train,
                    "input_instances_test": input_instances_test,
                    ResultDataKey.labels_train: labels_train,
                    ResultDataKey.labels_test: labels_test,
                    ResultDataKey.labels_detected: complete_detected_labels
                }
            },
            ...
        },
    ....
}
"""


def get_result_data_as_data_frame():
    """
    Reads data from result file and converts it to data frame
    """
    result_data_array = []
    dictionary = read_dictionary_from_file("result/benchmark_result")
    for detector_name, file_results in dictionary.items():
        for file_result in file_results:
            for file_path, result_of_file in file_result.items():
                data = result_of_file[ResultDataKey.data]
                input_instances_train = data[ResultDataKey.input_instances_train]
                input_instances_test = data[ResultDataKey.input_instances_test]
                labels_train = data[ResultDataKey.labels_train]
                labels_test = data[ResultDataKey.labels_test]
                labels_detected = data[ResultDataKey.labels_detected]
                dataset_name = result_of_file[ResultDataKey.dataset_name]
                if labels_detected.size != np.concatenate((labels_train, labels_test)).size:
                    raise ValueError("labels of dataset and detected labels are not same for file: " +
                                     file_path + " and detector: " + detector_name)
                result_data_frame_row = __convert_to_result_data_frame_row(input_instances_train, input_instances_test,
                                                                           labels_train, labels_test, labels_detected,
                                                                           file_path, detector_name, dataset_name)
                result_data_array.append(result_data_frame_row)
    return __make_result_data_frame(result_data_array)


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
                                       labels_detected, file_path, detector_name, dataset_name):
    return [detector_name, dataset_name, file_path, input_instances_train, input_instances_test,
            labels_train, labels_test, labels_detected]


def __make_result_data_frame(result_data_array):
    return pd.DataFrame(result_data_array, columns=[ResultDataKey.detector_name, ResultDataKey.dataset_name,
                                                    ResultDataKey.file_path, ResultDataKey.input_instances_train,
                                                    ResultDataKey.input_instances_test, ResultDataKey.labels_train,
                                                    ResultDataKey.labels_test, ResultDataKey.labels_detected])

# Todo should add FP, TP, FN, FP visualization as well.
