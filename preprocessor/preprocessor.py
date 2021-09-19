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
        # cleaning data
        df = self.__clean_data(df)
        labels = df["is_anomaly"]
        input_instances = df.loc[:, df.columns != "is_anomaly"]
        input_instances = input_instances.set_index("timestamp")
        return input_instances, labels

    def get_input_instances_and_labels_split(self):
        input_instances, labels = self.__get_data_from_file()
        input_instances_train, input_instances_test, labels_train, labels_test = \
            train_test_split(input_instances, labels, train_size=0.5, shuffle=False)
        return input_instances_train, input_instances_test, labels_train, labels_test

    def __clean_data(self, df):
        data = df.dropna()
        return data
