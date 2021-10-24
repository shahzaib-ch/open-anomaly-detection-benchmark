import pandas as pd
from sklearn.model_selection import train_test_split


class PreProcessor:

    def __init__(self, file_path, train_size):
        self.file_path = file_path
        self.train_size = train_size

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
            train_test_split(input_instances, labels, train_size=self.train_size, shuffle=False)
        input_instances_train = input_instances_train.to_numpy()
        input_instances_test = input_instances_test.to_numpy()
        labels_train = labels_train.to_numpy()
        labels_test = labels_test.to_numpy()
        return input_instances_train, input_instances_test, labels_train, labels_test

    def __clean_data(self, df):
        data = df.dropna()
        return data
