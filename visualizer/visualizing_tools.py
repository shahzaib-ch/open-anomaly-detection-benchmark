import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from helper.common_methods import read_dictionary_from_file
from preprocessor.preprocessor import PreProcessor

"""
Result format is:
{
    detector name: {
        [
            dataset_file_path: [
                    dataset_name,
                    detector_name,
                    complete_detected_labels
            ]
        },
    ....
}
"""


def get_result_data_and_show():
    dictionary = read_dictionary_from_file("result/benchmark_result")
    for detector_name, file_results in dictionary.items():
        for file_result in file_results:
            for file_path, result_of_file_list in file_result.items():
                preprocessor = PreProcessor(file_path)
                input_instances_train, input_instances_test, labels_train, labels_test = \
                    preprocessor.get_input_instances_and_labels_split()
                visualize(input_instances_train, input_instances_test, labels_train, labels_test,
                          result_of_file_list[2], file_path, detector_name)


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
