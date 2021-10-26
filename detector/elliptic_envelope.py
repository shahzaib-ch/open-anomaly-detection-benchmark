from abc import ABC

from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import normalize

from detector.base_detector import BaseDetector
from helper.labels_helper import replace_in_array


class EllipticEnvelopeDetector(BaseDetector, ABC):
    __not_supported_datasets = ['data/datasets/nab/artificialNoAnomaly/art_flatline.csv']

    def createInstance(self, features_count, contamination):
        self.model = EllipticEnvelope(support_fraction=1, contamination=contamination)

    def train(self, input_instances, labels):
        self.model.fit(input_instances, labels)

    def predict(self, input_instances):
        scores = self.model.decision_function(input_instances)
        scores = normalize(scores.reshape(1, -1))[0]
        scores = 1 + scores
        return scores

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
