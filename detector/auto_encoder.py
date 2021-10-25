from abc import ABC

from pyod.models.auto_encoder import AutoEncoder

from detector.base_detector import BaseDetector


class AutoEncoderDetector(BaseDetector, ABC):
    __not_supported_datasets = []

    def createInstance(self, features_count, contamination):
        hidden_neurons = [1, 1]
        if features_count > 1:
            hidden_neurons = [features_count * 2/3, features_count, features_count, features_count * 2/3]
        self.model = AutoEncoder(hidden_neurons=hidden_neurons)

    def train(self, input_instances, labels):
        self.model.fit(input_instances)

    def predict(self, input_instances):
        labels = self.model.predict(input_instances)
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
