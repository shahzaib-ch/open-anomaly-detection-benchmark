from abc import ABC

from sklearn.covariance import EllipticEnvelope

from detector.base_detector import BaseDetector
from helper.labels_helper import replace_in_array


class EllipticEnvelopeDetector(BaseDetector, ABC):
    __not_supported_datasets = ['data/datasets/nab/artificialNoAnomaly/art_flatline.csv']

    def createInstance(self):
        self.model = EllipticEnvelope(random_state=10, support_fraction=1)

    def train(self, input_instances, labels):
        self.model.fit(input_instances, labels)

    def predict(self, input_instances):
        labels = self.model.predict(input_instances)
        labels = replace_in_array(labels, 1, 0)
        labels = replace_in_array(labels, -1, 1)
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
