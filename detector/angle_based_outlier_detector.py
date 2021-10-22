from abc import ABC

import pandas as pd
from pyod.models.abod import ABOD

from detector.base_detector import BaseDetector
from helper.common_methods import convert_data_frame_to_float


class AngleBasedOutlierDetector(BaseDetector, ABC):
    __not_supported_datasets = []

    def createInstance(self, features_count):
        self.model = ABOD()

    def train(self, input_instances, labels):
        if isinstance(input_instances, pd.DataFrame):
            input_instances = convert_data_frame_to_float(input_instances)
        self.model.fit(input_instances)

    def predict(self, input_instances):
        if isinstance(input_instances, pd.DataFrame):
            input_instances = convert_data_frame_to_float(input_instances)
        labels = self.model.predict(input_instances)
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
