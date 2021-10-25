from abc import ABC

from sklearn.ensemble import IsolationForest

from detector.base_detector import BaseDetector
from helper.labels_helper import replace_in_array


class IsolationForestDetector(BaseDetector, ABC):
    __not_supported_datasets = []

    def createInstance(self, features_count, contamination):
        self.model = IsolationForest(contamination=contamination)

    def train(self, input_instances, labels):
        self.model.fit(input_instances, labels)

    def predict(self, input_instances):
        labels = self.model.predict(input_instances)
        labels = replace_in_array(labels, 1, 0)
        labels = replace_in_array(labels, -1, 1)
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
