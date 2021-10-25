from abc import ABC

from sklearn.neighbors import LocalOutlierFactor

from detector.base_detector import BaseDetector
from helper.labels_helper import replace_in_array


class LocalOutlierFactorDetector(BaseDetector, ABC):
    __not_supported_datasets = []

    def createInstance(self, features_count, contamination):
        self.model = LocalOutlierFactor(contamination=contamination, novelty=True)

    def train(self, input_instances, labels):
        self.model.fit(input_instances, labels)

    def predict(self, input_instances):
        scores = self.model.decision_function(input_instances)
        scores = 1 - scores
        return scores

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
