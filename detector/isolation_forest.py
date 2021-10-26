from abc import ABC

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize

from detector.base_detector import BaseDetector


class IsolationForestDetector(BaseDetector, ABC):
    __not_supported_datasets = []

    def createInstance(self, features_count, contamination):
        self.model = IsolationForest(contamination=contamination, random_state=5)

    def train(self, input_instances, labels):
        self.model.fit(input_instances, labels)

    def predict(self, input_instances):
        scores = self.model.decision_function(input_instances)
        scores = normalize(scores.reshape(1, -1))[0]
        scores = 1 + scores
        return scores

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
