import os


def get_all_csv_files(folder_path):
    """
    Return list of path to all csv files in the folder
    :param folder_path:
    :return: csv_files
    """
    files_in_folder = os.listdir(folder_path)
    csv_files = []
    for file_name in files_in_folder:
        if os.path.isdir(folder_path + "/" + file_name):
            csv_files.extend(get_all_csv_files(folder_path + "/" + file_name))
        else:
            csv_files.extend([folder_path + "/" + file_path for file_path in files_in_folder if file_path.endswith(".csv")])
    return csv_files


class DatasetCollector:
    DATA_DIRECTORY_PATH = "datasets/"

    # Returns dictionary with folder name as key and
    # list of csv files path in folder
    def get_all_csv_files_in_datasets_folder(self):
        datasets_folder_list = os.listdir(self.DATA_DIRECTORY_PATH)
        csv_files_dictionary = {}
        for folder_name in datasets_folder_list:
            folder_path = self.DATA_DIRECTORY_PATH + folder_name
            csv_files_dictionary[folder_name] = get_all_csv_files(folder_path)
        return csv_files_dictionary

    """def data_from_file(self, path):
    df = pd.read_csv(path)
    X = df.iloc[:, 1:2].to_numpy()
    y = df.iloc[:, 2:3].to_numpy()
    y = y.reshape(y.size)
    return X, y"""


dataset_collector = DatasetCollector()
print(dataset_collector.get_all_csv_files_in_datasets_folder())
