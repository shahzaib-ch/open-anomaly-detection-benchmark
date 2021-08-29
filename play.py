from helper.LabelsHelper import update_column_name_for_all_file_in_folder
from visualizer.visualizing_tools import get_result_data_and_show

import pandas as pd

update_column_name_for_all_file_in_folder("data/datasets/yahoo/A3Benchmark", "timestamps", "timestamp")
