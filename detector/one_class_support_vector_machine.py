from abc import ABC

import numpy as np
from adtk.detector import GeneralizedESDTestAD, PcaAD
from pyod.models.ocsvm import OCSVM

from detector.base_detector import BaseDetector
from helper.common_methods import add_date_time_index_to_df


class OneClassSupportVectorMachineDetector(BaseDetector, ABC):
    __not_supported_datasets = []

    def createInstance(self):
        self.model = OCSVM()

    def train(self, input_instances, labels):
        self.model.fit(input_instances)

    def predict(self, input_instances):
        labels = self.model.predict(input_instances)
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
