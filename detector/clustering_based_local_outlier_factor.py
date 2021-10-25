from abc import ABC

from pyod.models.cblof import CBLOF

from detector.base_detector import BaseDetector


class ClusteringBasedLocalOutlierFactorDetector(BaseDetector, ABC):
    __not_supported_datasets = ["data/datasets/nab/artificialNoAnomaly/art_daily_perfect_square_wave.csv",
                                "data/datasets/nab/artificialNoAnomaly/art_noisy.csv",
                                "data/datasets/nab/artificialNoAnomaly/art_flatline.csv",
                                "data/datasets/yahoo/A4Benchmark/A4Benchmark-TS42.csv"]

    def createInstance(self, features_count, contamination):
        self.model = CBLOF()

    def train(self, input_instances, labels):
        self.model.fit(input_instances)

    def predict(self, input_instances):
        labels = self.model.predict(input_instances)
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
