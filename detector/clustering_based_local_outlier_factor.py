from abc import ABC

from pyod.models.cblof import CBLOF

from detector.base_detector import BaseDetector
from helper.common_methods import get_2nd_value_from_list


class ClusteringBasedLocalOutlierFactorDetector(BaseDetector, ABC):
    __not_supported_datasets = ["data/datasets/nab/artificialNoAnomaly/art_daily_perfect_square_wave.csv",
                                "data/datasets/nab/artificialNoAnomaly/art_noisy.csv",
                                "data/datasets/nab/artificialNoAnomaly/art_flatline.csv",
                                "data/datasets/yahoo/A4Benchmark/A4Benchmark-TS42.csv"]

    def createInstance(self, features_count, contamination):
        self.model = CBLOF(contamination=contamination)

    def train(self, input_instances, labels):
        self.model.fit(input_instances)

    def predict(self, input_instances):
        scores = self.model.predict_proba(input_instances)
        scores = get_2nd_value_from_list(scores)
        return scores

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
