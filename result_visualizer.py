from visualizer.result_data_keys import ResultDataKey
from visualizer.result_visualizer import ResultVisualizer

visualizer = ResultVisualizer(accuracy_measure=ResultDataKey.accuracy, anomaly_threshold=0.95)
visualizer.show_result_overview_heat_map()
