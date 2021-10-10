import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.text import Text

from helper.common_methods import round_xy_coordinates
from visualizer.heatmap_helper import heatmap, annotate_heatmap
from visualizer.result_collector import get_result_data_as_data_frame
from visualizer.result_data_keys import ResultDataKey
from visualizer.result_metric_calculators import add_accuracy_to_df, add_subfolder_name_to_df


class ResultVisualizer:

    def __init__(self):
        self.result_data_frame = get_result_data_as_data_frame()

    def show_full_detailed_result_heat_map(self):
        heat_map_df = add_accuracy_to_df(self.result_data_frame)
        heat_map_df = heat_map_df.loc[:, (ResultDataKey.detector_name, ResultDataKey.file_path, ResultDataKey.accuracy)]
        heat_map_df = heat_map_df.pivot(index=ResultDataKey.file_path, columns=ResultDataKey.detector_name)
        ax = sns.heatmap(heat_map_df, cmap='BuPu')
        ax.set_title("Heatmap of algorithm accuracy on Yahoo dataset")
        plt.show()

    def show_result_overview_heat_map(self):
        heat_map_df = add_accuracy_to_df(self.result_data_frame)
        heat_map_df = heat_map_df.loc[:,
                      (ResultDataKey.detector_name, ResultDataKey.dataset_name, ResultDataKey.accuracy)]
        heat_map_df = heat_map_df.groupby([ResultDataKey.detector_name, ResultDataKey.dataset_name],
                                          as_index=False).mean()
        heat_map_df = heat_map_df.pivot(index=ResultDataKey.detector_name, columns=ResultDataKey.dataset_name)
        im, cbar, ax = heatmap(heat_map_df, heat_map_df.index, heat_map_df.columns, picker=True)
        annotate_heatmap(im)

        def on_pick(event):
            # ... process selected item
            if isinstance(event.artist, Text):
                detector_name = event.artist.get_text()
                print("Selected detector: ", detector_name)
                self.show_result_of_detector(detector_name)

            if isinstance(event.artist, AxesImage):
                dataset_index, detector_index = round_xy_coordinates(event.mouseevent.xdata, event.mouseevent.ydata)
                detector_name = heat_map_df.index.array[detector_index]
                dataset_name = heat_map_df.columns[dataset_index][1]
                print("Selected detector: ", detector_name, "---", "Selected dataset: ", dataset_name)
                self.show_result_of_detector_against_dataset_sub_folders(detector_name, dataset_name)

        ax.figure.canvas.mpl_connect("pick_event", on_pick)
        plt.figure(1)
        plt.show()

    def show_result_of_detector(self, detector_name):
        # to be implemented
        print(detector_name)

    def show_result_of_detector_against_dataset_sub_folders(self, detector_name, dataset_name):
        result_data_frame = self.result_data_frame
        heat_map_df = result_data_frame[
            (result_data_frame.detector_name == detector_name) & (result_data_frame.dataset_name == dataset_name)]
        heat_map_df = add_subfolder_name_to_df(heat_map_df)
        heat_map_df = heat_map_df.loc[:, (ResultDataKey.subfolder, ResultDataKey.accuracy)]
        heat_map_df = heat_map_df.groupby([ResultDataKey.subfolder],
                                          as_index=False).mean()
        subfolders = heat_map_df[ResultDataKey.subfolder].to_numpy()
        accuracy = heat_map_df[ResultDataKey.accuracy].to_numpy()
        plt.figure(2)
        plt.bar(subfolders, accuracy)
        plt.show()

    def show_result_of_detector_against_dataset(self, detector_name, dataset_name):
        heat_map_df = add_accuracy_to_df(self.result_data_frame)
        heat_map_df = heat_map_df[
            (heat_map_df.detector_name == detector_name) & (heat_map_df.dataset_name == dataset_name)]
        heat_map_df = heat_map_df.loc[:, (ResultDataKey.file_path, ResultDataKey.accuracy)]
        file_paths = heat_map_df[ResultDataKey.file_path].to_numpy()
        accuracy = heat_map_df[ResultDataKey.accuracy].to_numpy()
        plt.figure(2)
        plt.bar(file_paths, accuracy)
        plt.show()

