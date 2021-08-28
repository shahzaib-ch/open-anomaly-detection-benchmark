from abc import ABC

from sklearn.covariance import EllipticEnvelope

from detector.base_detector import BaseDetector
from helper.LabelsHelper import replace_in_array


class EllipticEnvelopeDetector(BaseDetector, ABC):
    def createInstance(self):
        self.model = EllipticEnvelope(random_state=10)

    def train(self, input_instances, labels):
        self.model.fit(input_instances, labels)

    def predict(self, input_instances):
        labels = self.model.predict(input_instances)
        labels = replace_in_array(labels, 1, 0)
        labels = replace_in_array(labels, -1, 1)
        return labels
