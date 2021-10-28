import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from pandas.plotting import parallel_coordinates
from sklearn.metrics import precision_recall_curve

from helper.common_methods import round_xy_coordinates
from visualizer.heatmap_helper import heatmap, annotate_heatmap
from visualizer.result_collector import get_result_data_as_data_frame
from visualizer.result_data_keys import ResultDataKey
from visualizer.result_metric_calculators import add_accuracy_to_df, add_subfolder_name_to_df, add_f1_score_to_df, \
    add_detected_labels_to_df, add_average_precision_score_to_df, add_precision_score_to_df, add_recall_score_to_df

matplotlib.use("TkAgg")


def visualize_dataset_labels(file_path, detector_name, labels, labels_detected):
    figure = plt.figure(len(plt.get_fignums()) + 1)
    plt.plot(labels, label="Ground truth labels")
    plt.plot(labels_detected, label="Detected labels by detector")
    ax = figure.axes[0]
    ax.legend()
    title = detector_name + " performance on file: " + file_path
    ax.set_title(title)
    ax.set_ylabel("1 = anomaly, 0 = normal instance")
    plt.show()


def visualize_dataset(file_path, input_instances):
    figure = plt.figure(len(plt.get_fignums()) + 1)
    plt.plot(input_instances)
    ax = figure.axes[0]
    title = file_path + " data"
    ax.set_title(title)
    ax.set_ylabel("values")
    plt.show()


class AccuracyResultVisualizer:

    def __init__(self, accuracy_measure, anomaly_threshold, use_windows):
        self.accuracy_measure = accuracy_measure
        self.result_data_frame = get_result_data_as_data_frame()
        self.result_data_frame.dropna()
        self.result_data_frame = add_detected_labels_to_df(self.result_data_frame, anomaly_threshold, use_windows)

        if accuracy_measure == ResultDataKey.accuracy:
            self.result_data_frame = add_accuracy_to_df(self.result_data_frame)
        elif accuracy_measure == ResultDataKey.f1_score:
            self.result_data_frame = add_f1_score_to_df(self.result_data_frame)
        elif accuracy_measure == ResultDataKey.average_precision_score:
            self.result_data_frame = add_average_precision_score_to_df(self.result_data_frame)
        elif accuracy_measure == ResultDataKey.precision:
            self.result_data_frame = add_precision_score_to_df(self.result_data_frame)
        elif accuracy_measure == ResultDataKey.recall:
            self.result_data_frame = add_recall_score_to_df(self.result_data_frame)
        self.result_data_frame.dropna()
        self.result_data_frame = add_subfolder_name_to_df(self.result_data_frame)

    def show_full_detailed_result_heat_map(self):
        heat_map_df = self.result_data_frame.loc[:,
                      (ResultDataKey.detector_name, ResultDataKey.file_path, self.accuracy_measure)]
        heat_map_df = heat_map_df.pivot(index=ResultDataKey.file_path, columns=ResultDataKey.detector_name)
        ax = sns.heatmap(heat_map_df, cmap='BuPu')
        ax.set_title("Heatmap of each algorithm " + self.accuracy_measure.lower() + " score for each dataset")
        plt.show()

    def show_result_overview_heat_map(self):
        heat_map_df = self.result_data_frame.loc[:,
                      (ResultDataKey.detector_name, ResultDataKey.dataset_name, self.accuracy_measure)]
        heat_map_df = heat_map_df.groupby([ResultDataKey.detector_name, ResultDataKey.dataset_name],
                                          as_index=False).mean()
        heat_map_df = heat_map_df.pivot(index=ResultDataKey.detector_name, columns=ResultDataKey.dataset_name)
        im, cbar, ax = heatmap(heat_map_df, heat_map_df.index, heat_map_df.columns, cmap='BuPu',
                               cbarlabel=self.accuracy_measure,
                               picker=True)
        annotate_heatmap(im)

        def on_pick(event):
            # ... process selected item
            if isinstance(event.artist, Text):
                detector_name = event.artist.get_text()
                print("Selected detector: ", detector_name)
                self.__show_result_of_detector(detector_name)

            if isinstance(event.artist, AxesImage):
                dataset_index, detector_index = round_xy_coordinates(event.mouseevent.xdata, event.mouseevent.ydata)
                detector_name = heat_map_df.index.array[detector_index]
                dataset_name = heat_map_df.columns[dataset_index][1]
                print("Selected detector: ", detector_name, "---", "Selected dataset: ", dataset_name)
                self.__show_result_of_detector_against_dataset_sub_folders(detector_name, dataset_name)

        ax.figure.canvas.mpl_connect("pick_event", on_pick)
        ax.set_title("Heatmap of each detector " + self.accuracy_measure.lower() + " score for each data repository")
        plt.show()

    def __show_result_of_detector(self, detector_name):
        # to be implemented
        print(detector_name)

    def __show_result_of_detector_against_dataset_sub_folders(self, detector_name, dataset_name):
        result_data_frame = self.result_data_frame
        bar_df = result_data_frame[
            (result_data_frame.detector_name == detector_name) & (result_data_frame.dataset_name == dataset_name)]
        bar_df = bar_df.loc[:, (ResultDataKey.subfolder, self.accuracy_measure)]
        bar_df = bar_df.groupby([ResultDataKey.subfolder],
                                as_index=False).mean()
        subfolders = bar_df[ResultDataKey.subfolder].to_numpy()
        accuracy = bar_df[self.accuracy_measure].to_numpy()

        figure = plt.figure(len(plt.get_fignums()) + 1)
        plt.bar(subfolders, accuracy, picker=True)

        def on_pick(event):
            # ... process selected item
            if isinstance(event.artist, Rectangle):
                folder_index, _ = round_xy_coordinates(event.mouseevent.xdata, event.mouseevent.ydata)
                subfolder = subfolders[folder_index]
                print("Selected subfolder: ", subfolder)
                self.__show_result_of_detector_against_dataset_single_sub_folder(detector_name, dataset_name, subfolder)

        figure.canvas.mpl_connect("pick_event", on_pick)
        ax = figure.axes[0]
        title = self.accuracy_measure.lower() + " score of " + detector_name + " on " + dataset_name + "data " \
                                                                                                       "repository " \
                                                                                                       "subfolders "
        ax.set_title(title)
        ax.set_ylabel(self.accuracy_measure)
        ax.set_xlabel("Dataset repository subfolders")
        plt.xticks(rotation=45)
        plt.show()

    def __show_result_of_detector_against_dataset_single_sub_folder(self, detector_name, dataset_name, subfolder):
        result_data_frame = self.result_data_frame
        bar_df = result_data_frame[
            (result_data_frame.detector_name == detector_name) & (
                    result_data_frame.dataset_name == dataset_name) & (result_data_frame.subfolder == subfolder)]

        bar_df = bar_df.loc[:, (ResultDataKey.file_path, self.accuracy_measure)]

        file_paths = bar_df[ResultDataKey.file_path].to_numpy()
        accuracy = bar_df[self.accuracy_measure].to_numpy()
        figure = plt.figure(len(plt.get_fignums()) + 1)
        plt.bar(file_paths, accuracy, picker=True)

        def on_pick(event):
            # ... process selected item
            if isinstance(event.artist, Rectangle):
                file_path_index, _ = round_xy_coordinates(event.mouseevent.xdata, event.mouseevent.ydata)
                file_path = file_paths[file_path_index]
                print("Selected dataset file: ", file_path)
                data = self.__get_result_of_detector_against_single_dataset_file(detector_name, dataset_name, file_path)
                self.visualize_single_dataset_file_result(data)

        figure.canvas.mpl_connect("pick_event", on_pick)
        ax = figure.axes[0]
        title = self.accuracy_measure + " score of " + detector_name + " on " + \
                dataset_name + "data repository " \
                               "subfolder " + subfolder
        ax.set_title(title)
        ax.set_ylabel(self.accuracy_measure)
        ax.set_xlabel("Dataset files")
        plt.xticks(rotation=90)
        plt.show()

    def __get_result_of_detector_against_single_dataset_file(self, detector_name, dataset_name, file_path):
        result_data_frame = self.result_data_frame
        dataset_file_df = result_data_frame[
            (result_data_frame.detector_name == detector_name) & (
                    result_data_frame.dataset_name == dataset_name) & (
                    result_data_frame.file_path == file_path)]
        return dataset_file_df

    """
    def show_result_of_detector_against_dataset(self, detector_name, dataset_name):
        heat_map_df = add_accuracy_to_df(self.result_data_frame)
        heat_map_df = heat_map_df[
            (heat_map_df.detector_name == detector_name) & (heat_map_df.dataset_name == dataset_name)]
        heat_map_df = heat_map_df.loc[:, (ResultDataKey.file_path, self.accuracy_measure)]
        file_paths = heat_map_df[ResultDataKey.file_path].to_numpy()
        accuracy = heat_map_df[self.accuracy_measure].to_numpy()
        plt.figure(len(plt.get_fignums()) + 1)
        plt.bar(file_paths, accuracy)
        plt.show()
    """

    def visualize_single_dataset_file_result(self, data):
        file_path = data[ResultDataKey.file_path].values[0]
        dataset_name = data[ResultDataKey.dataset_name].values[0]
        detector_name = data[ResultDataKey.detector_name].values[0]
        accuracy = data[self.accuracy_measure].values[0]
        input_instances_train = data[ResultDataKey.input_instances_train].values[0]
        input_instances_test = data[ResultDataKey.input_instances_test].values[0]
        labels_train = data[ResultDataKey.labels_train].values[0]
        labels_test = data[ResultDataKey.labels_test].values[0]
        labels_detected = data[ResultDataKey.labels_detected].values[0]
        subfolder = data[ResultDataKey.subfolder].values[0]
        input_instances = np.concatenate((input_instances_train, input_instances_test))
        labels = np.concatenate((labels_train, labels_test))

        def visualize_dataset_clicked():
            visualize_dataset(file_path, input_instances)

        def visualize_dataset_labels_clicked():
            visualize_dataset_labels(file_path, detector_name, labels, labels_detected)

        def visualize_dataset_with_anomalies():
            self.visualize_dataset_with_anomalies(file_path, detector_name, input_instances_train,
                                                  input_instances_test, labels_train, labels_test, labels_detected)

        visualize_dataset_with_anomalies()
        """

        summary_window = DatasetResultSummaryWindow(
            visualize_dataset_clicked,
            visualize_dataset_labels_clicked,
            visualize_dataset_with_anomalies
        )
        summary_window.show_window()
        """

    def visualize_dataset_with_anomalies(self, file_path, detector_name, input_instances_train,
                                         input_instances_test, labels_train, labels_test, labels_detected):
        input_instances = np.concatenate((input_instances_train, input_instances_test))
        labels_detected_joined = np.concatenate((labels_train, labels_detected))
        labels_test_joined = np.concatenate((labels_train, labels_test))
        dataset_length = np.arange(len(input_instances))

        x_detected = []
        y_detected = []
        detected_color = 'mo'

        x_labelled = []
        y_labelled = []
        labelled_color = 'ro'

        x_detected_and_labelled = []
        y_detected_and_labelled = []
        detected_and_labelled_color = 'go'

        colors = []
        data_class = []
        for element in dataset_length:
            is_detected = labels_detected_joined[element] == 1
            is_labeled = labels_test_joined[element] == 1
            if element < len(input_instances_train) & is_labeled:
                color = 'r'
                x_labelled.append(element)
                y_labelled.append(input_instances[element])
                colors.append(color)
                data_class.append("Actual anomaly")
                continue

            color = 'b'
            if is_detected & is_labeled:
                color = 'g'
                x_detected_and_labelled.append(element)
                y_detected_and_labelled.append(input_instances[element])
                data_class.append("Actual anomaly and detected")
            elif is_detected:
                color = 'm'
                x_detected.append(element)
                y_detected.append(input_instances[element])
                data_class.append("Detected anomaly")
            elif is_labeled:
                color = 'r'
                x_labelled.append(element)
                y_labelled.append(input_instances[element])
                data_class.append("Actual anomaly")
            else:
                data_class.append("Normal instance")
            colors.append(color)
        figure = plt.figure(len(plt.get_fignums()) + 1)
        feature_count = len(input_instances_train[0])
        if feature_count <= 1:
            plt.plot(dataset_length, input_instances, label="Data instances")
            plt.plot(x_labelled, y_labelled, labelled_color, label="Actual anomalies")
            plt.plot(x_detected, y_detected, detected_color, label="Anomalies detected by algorithm")
            plt.plot(x_detected_and_labelled, y_detected_and_labelled, detected_and_labelled_color,
                     label="Actual anomalies detected by algorithm")
        else:
            # plt.scatter(dataset_length, input_instances, c=colors)
            df = pd.DataFrame(input_instances)
            df["is_anomaly"] = data_class
            parallel_coordinates(df, 'is_anomaly', colormap=plt.get_cmap("Set2"))

        ax = figure.axes[0]
        title = file_path + " data with algorithm: " + detector_name
        ax.set_title(title)
        ax.set_ylabel("values")
        plt.legend(loc='upper right')
        plt.show()

        figure = plt.figure(len(plt.get_fignums()) + 1)
        # also show precision recall curve
        precision, recall, thresholds = precision_recall_curve(labels_test, labels_detected)
        plt.plot(recall, precision)
        ax = figure.axes[0]
        title = file_path + " data with algorithm: " + detector_name
        ax.set_title(title)
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        plt.show()
