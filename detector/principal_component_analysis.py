from abc import ABC

import numpy as np
import pandas as pd
from adtk.detector import PcaAD

from detector.base_detector import BaseDetector
from helper.common_methods import add_date_time_index_to_df


class PrincipalComponentAnalysisDetector(BaseDetector, ABC):
    __not_supported_datasets = ["odd"]

    def createInstance(self, features_count, contamination):
        self.model = PcaAD()

    def train(self, input_instances, labels):
        input_instances = pd.DataFrame(input_instances)
        input_instances = add_date_time_index_to_df(input_instances)

        self.model.fit(input_instances)

    def predict(self, input_instances):
        input_instances = pd.DataFrame(input_instances)
        input_instances = add_date_time_index_to_df(input_instances)
        labels = self.model.detect(input_instances)
        labels = labels.to_numpy().reshape(len(labels))
        labels = np.where(labels, 1, 0)
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
