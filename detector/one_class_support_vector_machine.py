from abc import ABC

from pyod.models.ocsvm import OCSVM

from detector.base_detector import BaseDetector


class OneClassSupportVectorMachineDetector(BaseDetector, ABC):
    __not_supported_datasets = []

    def createInstance(self, features_count, contamination):
        self.model = OCSVM(cache_size=4000, contamination=contamination)

    def train(self, input_instances, labels):
        self.model.fit(input_instances)

    def predict(self, input_instances):
        scores = self.model.decision_function(input_instances)
        return scores

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
