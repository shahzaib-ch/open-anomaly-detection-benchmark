from abc import ABC

import numpy as np

from detector.base_detector import BaseDetector
from detector.nab.context_ose.context_ose_detector import ContextOSEDetectorNab


class ContextualAnomalyDetector(BaseDetector, ABC):
    __not_supported_datasets = ["odd"]
    __input_instances_train = []
    __probationary_period = 0

    def createInstance(self):
        """
        we will create instance in predict method due to NAB implementation
        """

    def train(self, input_instances, labels):
        self.__input_instances_train = input_instances
        self.__probationary_period = len(input_instances)

    def predict(self, input_instances):
        input_instances = np.concatenate((self.__input_instances_train, input_instances))
        self.model = ContextOSEDetectorNab(input_instances, self.__probationary_period)
        self.model.initialize()
        labels = self.model.run()
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
