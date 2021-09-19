import os

import numpy as np

from data.dataset_collector import DatasetCollector
from detector.detector_aggregator import ALGORITHMS_DICTIONARY
from helper.common_methods import read_dictionary_from_file, save_dictionary_to_file
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

                if dataset_file_path in detector_instance.notSupportedDatasets():
                    print("Detector: " + detector_name + " is not suitable for " + dataset_file_path + "....")
                    continue
                    
                input_instances_train, input_instances_test, labels_train, labels_test = \
                    __pre_process_data_set(dataset_file_path)
                detected_labels = __run_detector_on_data(detector_instance, input_instances_train, input_instances_test,
                                                         labels_train)
                complete_detected_labels = np.concatenate((labels_train, detected_labels))
                detector_result = __create_result_json(detector_name, dataset_name, dataset_file_path,
                                                       input_instances_train, input_instances_test, labels_train,
                                                       labels_test, complete_detected_labels)
                __save_detector_result(detector_name, detector_result)


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
    detector_instance.createInstance()

    # training model
    detector_instance.train(input_instances_train, labels_train)

    # predicting/anomaly detection
    return detector_instance.predict(input_instances_test)


def __create_result_json(detector_name, dataset_name, dataset_file_path,
                         input_instances_train, input_instances_test, labels_train, labels_test,
                         complete_detected_labels):
    return {
        dataset_file_path: {
            "dataset_name": dataset_name,
            "detector_name": detector_name,
            "data": {
                "input_instances_train": input_instances_train,
                "input_instances_test": input_instances_test,
                "labels_train": labels_train.to_numpy(),
                "labels_test": labels_test.to_numpy(),
                "labels_detected": complete_detected_labels
            }
        }
    }


def __save_detector_result(detector_name, detector_result):
    result_folder_path = "result"
    result_file_path = "result/benchmark_result"
    benchmark_result_dictionary = {}
    detector_results_from_file = []

    if not os.path.isdir(result_folder_path):
        os.mkdir(result_folder_path)

    if os.path.isfile(result_file_path):
        benchmark_result_dictionary = read_dictionary_from_file(result_file_path)

    if detector_name in benchmark_result_dictionary.keys():
        detector_results_from_file = benchmark_result_dictionary.get(detector_name)

    detector_results_from_file.append(detector_result)
    benchmark_result_dictionary[detector_name] = detector_results_from_file

    save_dictionary_to_file(benchmark_result_dictionary, result_file_path)


def __clearing_data_from_last_run():
    if os.path.isfile("result/benchmark_result"):
        os.remove("result/benchmark_result")
