from abc import ABC

from pyod.models.ocsvm import OCSVM

from detector.base_detector import BaseDetector


class OneClassSupportVectorMachineDetector(BaseDetector, ABC):
    __not_supported_datasets = []

    def createInstance(self, features_count):
        self.model = OCSVM(cache_size=4000)

    def train(self, input_instances, labels):
        self.model.fit(input_instances)

    def predict(self, input_instances):
        labels = self.model.predict(input_instances)
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
