from visualizer.result_data_keys import ResultDataKey
from visualizer.result_accuracy_visualizer import AccuracyResultVisualizer

visualizer = AccuracyResultVisualizer(accuracy_measure=ResultDataKey.precision, anomaly_threshold=0.99, use_windows=True)
visualizer.show_result_overview_heat_map()
