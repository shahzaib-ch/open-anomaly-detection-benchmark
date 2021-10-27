from abc import ABC

from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import normalize

from detector.base_detector import BaseDetector
from helper.common_methods import standardize_data
from helper.labels_helper import replace_in_array


class EllipticEnvelopeDetector(BaseDetector, ABC):
    __not_supported_datasets = ['data/datasets/nab/artificialNoAnomaly/art_flatline.csv']

    def createInstance(self, features_count, contamination):
        self.model = EllipticEnvelope(support_fraction=1, contamination=contamination)

    def train(self, input_instances, labels):
        self.model.fit(input_instances, labels)

    def predict(self, input_instances):
        scores = self.model.score_samples(input_instances)
        scores = standardize_data(abs(scores))
        return scores

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
