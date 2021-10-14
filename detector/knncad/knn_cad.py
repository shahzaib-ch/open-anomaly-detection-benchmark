from abc import ABC

import numpy as np

from detector.base_detector import BaseDetector
from detector.knncad.knncad_detector_nab import KnncadDetectorNab


class KnncadDetector(BaseDetector, ABC):
    __not_supported_datasets = []
    __input_instances_train = []
    __probationary_period = 0

    def createInstance(self):
        self.model = KnncadDetectorNab()

    def train(self, input_instances, labels):
        self.__input_instances_train = input_instances
        self.__probationary_period = len(input_instances)
        self.model.probationaryPeriod = self.__probationary_period

    def predict(self, input_instances):
        input_instances = np.concatenate((self.__input_instances_train, input_instances))
        labels = []
        for row in input_instances:
            label = self.model.handleRecord(row)[0]
            labels.append(label)
            print(label)
        # labels = replace_in_array(labels, 1, 0)
        # labels = replace_in_array(labels, -1, 1)
        return labels

    def notSupportedDatasets(self):
        return self.__not_supported_datasets
