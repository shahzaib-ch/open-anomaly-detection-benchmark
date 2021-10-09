import seaborn as sns
from matplotlib import pyplot as plt

from visualizer.result_collector import get_result_data_as_data_frame
from visualizer.result_data_keys import ResultDataKey
from visualizer.result_metric_calculators import add_accuracy_to_df


def show_full_detailed_result_heat_map():
    result_data_frame = get_result_data_as_data_frame()
    heat_map_df = add_accuracy_to_df(result_data_frame)
    heat_map_df = heat_map_df.loc[:, (ResultDataKey.detector_name, ResultDataKey.file_path, ResultDataKey.accuracy)]
    heat_map_df = heat_map_df.pivot(index=ResultDataKey.file_path, columns=ResultDataKey.detector_name)
    r = sns.heatmap(heat_map_df, cmap='BuPu')
    r.set_title("Heatmap of algorithm accuracy on Yahoo dataset")
    plt.show()


def show_result_overview_heat_map():
    result_data_frame = get_result_data_as_data_frame()
    heat_map_df = add_accuracy_to_df(result_data_frame)
    heat_map_df = heat_map_df.loc[:, (ResultDataKey.detector_name, ResultDataKey.dataset_name, ResultDataKey.accuracy)]
    heat_map_df = heat_map_df.groupby([ResultDataKey.detector_name, ResultDataKey.dataset_name], as_index=False).mean()
    heat_map_df = heat_map_df.pivot(index=ResultDataKey.detector_name, columns=ResultDataKey.dataset_name)
    r = sns.heatmap(heat_map_df, cmap='BuPu')
    r.set_title("Heatmap of algorithm accuracy on Yahoo dataset")
    plt.show()
