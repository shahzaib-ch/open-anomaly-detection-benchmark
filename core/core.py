import os
import time

import numpy as np

from data.dataset_collector import DatasetCollector
from detector.detector_aggregator import ALGORITHMS_DICTIONARY
from helper.common_methods import save_dictionary_to_file, list_contains
from preprocessor.preprocessor import PreProcessor


def do_benchmarking():
    __clearing_data_from_last_run()

    dataset_collector = DatasetCollector()
    datasets = dataset_collector.get_all_csv_files_in_datasets_folder()
    print("Got these datasets: " + str(datasets.keys()))
    print("Got these algorithms: " + str(ALGORITHMS_DICTIONARY.keys()))
    print("Performing benchmarking....")
    for detector_name, detector_instance in ALGORITHMS_DICTIONARY.items():
        print("Evaluating detector: " + detector_name + " ....")

        for dataset_name, files_path_array in datasets.items():

            for dataset_file_path in files_path_array:
                print("Evaluating detector: " + detector_name + " on file " + dataset_file_path + "....")

                if list_contains(dataset_file_path, detector_instance.notSupportedDatasets()):
                    print("Detector: " + detector_name + " is not suitable for " + dataset_file_path + "....")
                    continue

                input_instances_train, input_instances_test, labels_train, labels_test = \
                    __pre_process_data_set(dataset_file_path)
                detected_labels, training_time, test_time = __run_detector_on_data(detector_instance,
                                                                                   input_instances_train,
                                                                                   input_instances_test, labels_train)
                complete_detected_labels = np.concatenate((labels_train, detected_labels))
                detector_result = __create_result_json(detector_name, dataset_name, dataset_file_path,
                                                       input_instances_train, input_instances_test, labels_train,
                                                       labels_test, complete_detected_labels, training_time, test_time)
                __save_detector_result(detector_name, dataset_name, dataset_file_path, detector_result)


def __pre_process_data_set(dataset_file_path):
    preprocessor = PreProcessor(dataset_file_path)
    return preprocessor.get_input_instances_and_labels_split()


def __run_detector_on_data(detector_instance, input_instances_train, input_instances_test, labels_train):
    """
    Returns list of detected anomalies, 0=normal and 1=anomaly
    :param detector_instance:
    :param input_instances_train:
    :param input_instances_test:
    :param labels_train:
    """
    # creating model
    features_count = len(input_instances_train[0])
    detector_instance.createInstance(features_count)

    start_time = time.monotonic()
    # training model
    detector_instance.train(input_instances_train, labels_train)

    training_time = time.monotonic() - start_time
    start_time = time.monotonic()

    # predicting/anomaly detection
    detected_labels = detector_instance.predict(input_instances_test)
    test_time = time.monotonic() - start_time
    return detected_labels, training_time, test_time


def __create_result_json(detector_name, dataset_name, dataset_file_path,
                         input_instances_train, input_instances_test, labels_train, labels_test,
                         complete_detected_labels, training_time, test_time):
    return {
        "data": {
            "dataset_name": dataset_name,
            "detector_name": detector_name,
            "dataset_file_path": dataset_file_path,
            "training_time": training_time,
            "test_time": test_time,
            "data": {
                "input_instances_train": input_instances_train,
                "input_instances_test": input_instances_test,
                "labels_train": labels_train.to_numpy(),
                "labels_test": labels_test.to_numpy(),
                "labels_detected": complete_detected_labels
            }
        }
    }


def __save_detector_result(detector_name, dataset_name, dataset_file_path, detector_result):
    dataset_file_path_parts = dataset_file_path.replace("data/datasets/", "").split("/")
    file_name = dataset_file_path_parts[len(dataset_file_path_parts) - 1][:-4]
    result_detector_dataset_folder_path = "/".join(dataset_file_path_parts[:-1])
    result_detector_dataset_folder_path = "result/" + detector_name + "/" + result_detector_dataset_folder_path
    result_file_path = result_detector_dataset_folder_path + "/" + file_name

    if not os.path.isdir(result_detector_dataset_folder_path):
        os.makedirs(result_detector_dataset_folder_path)

    save_dictionary_to_file(detector_result, result_file_path)


def __clearing_data_from_last_run():
    if os.path.isfile("result/benchmark_result"):
        os.remove("result/benchmark_result")
