from abc import ABC

from pyod.models.abod import ABOD
from pyod.models.knn import KNN

from detector.base_detector import BaseDetector


class KNearestNeighborsDetector(BaseDetector, ABC):
    __not_supported_datasets = []

    def createInstance(self):
        self.model = KNN()

    def train(self, input_instances, labels):
        self.model.fit(input_instances)

    def predict(self, input_instances):
        labels = self.model.predict(input_instances)
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
