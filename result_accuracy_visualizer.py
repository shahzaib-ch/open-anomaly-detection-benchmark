from visualizer.result_data_keys import ResultDataKey
from visualizer.result_visualizer import AccuracyResultVisualizer

visualizer = AccuracyResultVisualizer(accuracy_measure=ResultDataKey.recall, anomaly_threshold=0.60, use_windows=True)
visualizer.show_result_overview_heat_map()
