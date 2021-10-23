import os
import time
from multiprocessing import Pool

from data.dataset_collector import DatasetCollector
from detector.detector_aggregator import ALGORITHMS_DICTIONARY
from helper.common_methods import save_dictionary_to_file, list_contains
from preprocessor.preprocessor import PreProcessor


def do_benchmarking(training_dataset_size, do_not_update_existing_result):
    __clearing_data_from_last_run()

    args = get_list_of_args_for_detector_run(training_dataset_size, do_not_update_existing_result)

    with Pool(5) as p:
        p.map_async(run_detector, args)
        p.close()
        p.join()


def get_list_of_args_for_detector_run(training_dataset_size, do_not_update_existing_result):
    args = []
    dataset_collector = DatasetCollector()
    datasets = dataset_collector.get_all_csv_files_in_datasets_folder()
    print("Got these datasets: " + str(datasets.keys()))
    print("Got these algorithms: " + str(ALGORITHMS_DICTIONARY.keys()))
    print("Performing benchmarking....")
    for detector_name, detector_instance in ALGORITHMS_DICTIONARY.items():
        print("Evaluating detector: " + detector_name + " ....")

        for dataset_name, files_path_array in datasets.items():

            for dataset_file_path in files_path_array:

                if list_contains(dataset_file_path, detector_instance.notSupportedDatasets()):
                    print("Detector: " + detector_name + " is not suitable for " + dataset_file_path + "....")
                    continue

                result_file_path = __create_result_file_path(detector_name, dataset_file_path)
                if os.path.isfile(result_file_path) and do_not_update_existing_result:
                    print("Detector: " + detector_name + " already has result for " + dataset_file_path + "....")
                    continue

                args.append([dataset_file_path, training_dataset_size,
                             detector_instance, result_file_path, dataset_name, detector_name])

    return args


def __pre_process_data_set(dataset_file_path, train_size):
    preprocessor = PreProcessor(dataset_file_path, train_size)
    return preprocessor.get_input_instances_and_labels_split()


def run_detector(args):
    dataset_file_path = args[0]
    training_dataset_size = args[1]
    detector_instance = args[2]
    result_file_path = args[3]
    dataset_name = args[4]
    detector_name = args[5]

    print("Evaluating detector: " + detector_name + " on file " + dataset_file_path + "....")

    input_instances_train, input_instances_test, labels_train, labels_test = \
        __pre_process_data_set(dataset_file_path, training_dataset_size)
    detected_labels, training_time, test_time = __run_detector_on_data(detector_instance,
                                                                       input_instances_train,
                                                                       input_instances_test, labels_train)
    if labels_test.size != detected_labels.size:
        ValueError("detected_labels and labels_test are not same size: " + result_file_path)

    detector_result = __create_result_json(result_file_path, dataset_name, dataset_file_path,
                                           input_instances_train, input_instances_test, labels_train,
                                           labels_test, detected_labels, training_time, test_time)
    __save_detector_result(result_file_path, detector_result)
    print("saved result: ", result_file_path)


def __run_detector_on_data(detector_instance, input_instances_train, input_instances_test, labels_train):
    """
    Returns list of detected anomalies, 0=normal and 1=anomaly
    :param detector_instance:
    :param input_instances_train:
    :param input_instances_test:
    :param labels_train:
    """
    # creating model
    features_count = input_instances_train.columns.size
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
                         detected_labels, training_time, test_time):
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
                "labels_detected": detected_labels
            }
        }
    }


def __create_result_file_path(detector_name, dataset_file_path):
    dataset_file_path_parts = dataset_file_path.replace("data/datasets/", "").split("/")
    file_name = dataset_file_path_parts[len(dataset_file_path_parts) - 1][:-4]
    result_detector_dataset_folder_path = "/".join(dataset_file_path_parts[:-1])
    result_detector_dataset_folder_path = "result/" + detector_name + "/" + result_detector_dataset_folder_path
    result_file_path = result_detector_dataset_folder_path + "/" + file_name

    if not os.path.isdir(result_detector_dataset_folder_path):
        os.makedirs(result_detector_dataset_folder_path)

    return result_file_path


def __save_detector_result(result_file_path, detector_result):
    save_dictionary_to_file(detector_result, result_file_path)


def __clearing_data_from_last_run():
    if os.path.isfile("result/benchmark_result"):
        os.remove("result/benchmark_result")
