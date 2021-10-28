from visualizer.result_data_keys import ResultDataKey
from visualizer.result_visualizer import AccuracyResultVisualizer

visualizer = AccuracyResultVisualizer(accuracy_measure=ResultDataKey.precision, anomaly_threshold=0.90)
visualizer.show_result_overview_heat_map()
