from abc import ABC, abstractmethod

from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope


class BaseDetector(ABC):
    model = None

    @abstractmethod
    def createInstance(self):
        """
        Creates mode and assigned to model variable
        """
        pass

    @abstractmethod
    def train(self, input_instances, labels):
        """
        Trains model with training data
        :param input_instances: training data
        :param labels: labels for training data
        """
        pass

    @abstractmethod
    def predict(self, input_instances):
        """
        Takes test data and does anomaly detection,
        returns a list of labels with 0 as normal data instance and 1 as anomaly
        :param input_instances: test data
        """
        pass


ALGORITHMS_DICTIONARY = {
    "Kmeans": KMeans(n_clusters=2, random_state=20),
    "Elliptic Envelope": EllipticEnvelope(random_state=10)
}