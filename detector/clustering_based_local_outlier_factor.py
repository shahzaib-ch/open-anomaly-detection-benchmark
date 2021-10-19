from abc import ABC

from pyod.models.cblof import CBLOF

from detector.base_detector import BaseDetector


class ClusteringBasedLocalOutlierFactorDetector(BaseDetector, ABC):
    __not_supported_datasets = []

    def createInstance(self, features_count):
        self.model = CBLOF()

    def train(self, input_instances, labels):
        self.model.fit(input_instances)

    def predict(self, input_instances):
        labels = self.model.predict(input_instances)
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
