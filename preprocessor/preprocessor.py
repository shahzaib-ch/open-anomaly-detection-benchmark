import pandas as pd
from sklearn.model_selection import train_test_split


class PreProcessor:

    def __init__(self, file_path):
        self.file_path = file_path

    def __get_data_from_file(self):
        """
        Gets data from file and returns test, train split
        input_instances_train, input_instances_test, labels_train, labels_test
        """
        df = pd.read_csv(self.file_path)
        labels = df["is_anomaly"]
        input_instances = df.loc[:, df.columns != "is_anomaly"]
        return train_test_split(input_instances, labels, train_size=0.5, shuffle=False)

    def get_input_instances_and_labels_split(self):
        input_instances_train, input_instances_test, labels_train, labels_test = self.__get_data_from_file()
        return input_instances_train, input_instances_test, labels_train, labels_test
