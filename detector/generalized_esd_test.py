from abc import ABC

import numpy as np
from adtk.detector import GeneralizedESDTestAD

from detector.base_detector import BaseDetector
from helper.common_methods import add_date_time_index_to_df


class GeneralizedESDTestDetector(BaseDetector, ABC):
    __not_supported_datasets = ["odd"]

    def createInstance(self, features_count, contamination):
        self.model = GeneralizedESDTestAD()

    def train(self, input_instances, labels):
        input_instances = add_date_time_index_to_df(input_instances)

        self.model.fit(input_instances)

    def predict(self, input_instances):
        input_instances = add_date_time_index_to_df(input_instances)
        labels = self.model.detect(input_instances)
        labels = np.where(labels.value, 1, 0)
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
