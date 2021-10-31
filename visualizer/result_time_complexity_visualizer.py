import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

from helper.common_methods import list_of_all_files_in_folder_and_subfolders, read_dictionary_from_file
from visualizer.result_data_keys import ResultDataKey


class TimeComplexityResultVisualizer:
    def __init__(self):
        self.result_df = get_result_data_as_data_frame()

    def show_multivariate_training(self):
        df = self.result_df[self.result_df[ResultDataKey.dataset_name] == "odd"]

        df = df.loc[:, (ResultDataKey.detector_name, ResultDataKey.train_data_instances, ResultDataKey.training_time)]
        data_dict = {}
        group_by = df.groupby([ResultDataKey.detector_name],
                              as_index=False)
        for group in group_by.groups:
            group_df = group_by.get_group(group)
            data_dict[group] = {
                ResultDataKey.training_time: group_df[ResultDataKey.training_time].sort_values().to_numpy(),
                ResultDataKey.train_data_instances: group_df[
                    ResultDataKey.train_data_instances].sort_values().to_numpy(),
            }

        figure = plt.figure(len(plt.get_fignums()) + 1)
        for detector, data in data_dict.items():
            training_time = data[ResultDataKey.training_time]
            train_data_instances = data[ResultDataKey.train_data_instances]
            plt.plot(train_data_instances, training_time, label=detector)

        ax = figure.axes[0]
        ax.set_xlabel("Dataset size (number of data instances)")
        ax.set_ylabel("Time (seconds)")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

    def show_univariate_test(self):
        df = self.result_df[self.result_df[ResultDataKey.dataset_name] != "odd"]

        df = df.loc[:, (ResultDataKey.detector_name, ResultDataKey.test_data_instances, ResultDataKey.test_time)]
        data_dict = {}
        group_by = df.groupby([ResultDataKey.detector_name],
                              as_index=False)
        for group in group_by.groups:
            group_df = group_by.get_group(group)
            data_dict[group] = {
                ResultDataKey.test_time: group_df[ResultDataKey.test_time].sort_values().to_numpy(),
                ResultDataKey.test_data_instances: group_df[ResultDataKey.test_data_instances].sort_values().to_numpy(),
            }

        figure = plt.figure(len(plt.get_fignums()) + 1)
        for detector, data in data_dict.items():
            test_time = data[ResultDataKey.test_time]
            test_data_instances = data[ResultDataKey.test_data_instances]
            test_data_instances, test_time = smoothing(test_data_instances, test_time)
            plt.plot(test_data_instances, test_time, label=detector)

        ax = figure.axes[0]
        ax.set_xlabel("Dataset size (number of data instances)")
        ax.set_ylabel("Time (seconds)")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

    def show_univariate_total(self):
        df = self.result_df[self.result_df[ResultDataKey.dataset_name] != "odd"]

        df = df.loc[:, (ResultDataKey.detector_name, ResultDataKey.total_data_instances, ResultDataKey.total_time)]
        data_dict = {}
        group_by = df.groupby([ResultDataKey.detector_name],
                              as_index=False)
        for group in group_by.groups:
            group_df = group_by.get_group(group)
            data_dict[group] = {
                ResultDataKey.total_time: group_df[ResultDataKey.total_time].sort_values().to_numpy(),
                ResultDataKey.total_data_instances: group_df[
                    ResultDataKey.total_data_instances].sort_values().to_numpy(),
            }

        figure = plt.figure(len(plt.get_fignums()) + 1)
        for detector, data in data_dict.items():
            total_time = data[ResultDataKey.total_time]
            total_data_instances = data[ResultDataKey.total_data_instances]
            total_data_instances, total_time = smoothing(total_data_instances, total_time)
            plt.plot(total_data_instances, total_time, label=detector)

        ax = figure.axes[0]
        ax.set_xlabel("Dataset size (number of data instances)")
        ax.set_ylabel("Time (seconds)")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

    def show_multivariate_total(self):
        df = self.result_df[self.result_df[ResultDataKey.dataset_name] == "odd"]

        df = df.loc[:, (ResultDataKey.detector_name, ResultDataKey.total_data_instances, ResultDataKey.total_time)]
        data_dict = {}
        group_by = df.groupby([ResultDataKey.detector_name],
                              as_index=False)
        for group in group_by.groups:
            group_df = group_by.get_group(group)
            data_dict[group] = {
                ResultDataKey.total_time: group_df[ResultDataKey.total_time].sort_values().to_numpy(),
                ResultDataKey.total_data_instances: group_df[
                    ResultDataKey.total_data_instances].sort_values().to_numpy(),
            }

        figure = plt.figure(len(plt.get_fignums()) + 1)
        for detector, data in data_dict.items():
            total_time = data[ResultDataKey.total_time]
            total_data_instances = data[ResultDataKey.total_data_instances]
            total_data_instances, total_time = smoothing(total_data_instances, total_time)
            plt.plot(total_data_instances, total_time, label=detector)

        ax = figure.axes[0]
        ax.set_xlabel("Dataset size (number of data instances)")
        ax.set_ylabel("Time (seconds)")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()


def smoothing(x, y):
    lowess_frac = 0.15  # size of data (%) for estimation =~ smoothing window
    lowess_it = 0
    x_smooth = x
    y_smooth = lowess(y, x, is_sorted=False, frac=lowess_frac, it=lowess_it, return_sorted=False)
    return x_smooth, y_smooth


def get_result_data_as_data_frame():
    """
    Reads data from result files and converts it to data frame
    """
    result_data_array = []
    result_files = list_of_all_files_in_folder_and_subfolders("result/")
    for file in result_files:
        dictionary = read_dictionary_from_file(file)
        result_of_file = dictionary[ResultDataKey.data]
        file_path = result_of_file[ResultDataKey.dataset_file_path]
        detector_name = result_of_file[ResultDataKey.detector_name]
        dataset_name = result_of_file[ResultDataKey.dataset_name]
        training_time = result_of_file[ResultDataKey.training_time]
        test_time = result_of_file[ResultDataKey.test_time]
        total_time = training_time + test_time

        data = result_of_file[ResultDataKey.data]
        labels_train = data[ResultDataKey.labels_train]
        labels_test = data[ResultDataKey.labels_test]
        train_data_instances = len(labels_train)
        test_data_instances = len(labels_test)
        total_data_instances = len(labels_test)

        data_array = [detector_name, dataset_name, file_path, training_time, test_time, total_time,
                      train_data_instances, test_data_instances, total_data_instances]
        result_data_array.append(data_array)

    return pd.DataFrame(result_data_array,
                        columns=[ResultDataKey.detector_name, ResultDataKey.dataset_name,
                                 ResultDataKey.file_path, ResultDataKey.training_time,
                                 ResultDataKey.test_time, ResultDataKey.total_time,
                                 ResultDataKey.train_data_instances,
                                 ResultDataKey.test_data_instances,
                                 ResultDataKey.total_data_instances])
