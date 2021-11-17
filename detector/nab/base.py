# ----------------------------------------------------------------------
# Copyright (C) 2014-2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import abc


class AnomalyDetector(object, metaclass=abc.ABCMeta):
    """
    Base class for all anomaly detectors. When inheriting from this class please
    take note of which methods MUST be overridden, as documented below.
    """

    def __init__(self,
                 dataSet,
                 probationaryPeriod):

        self.dataSet = dataSet
        self.probationaryPeriod = probationaryPeriod

        self.inputMin = self.dataSet.min()
        self.inputMax = self.dataSet.max()

    def initialize(self):
        """Do anything to initialize your detector in before calling run.

        Pooling across cores forces a pickling operation when moving objects from
        the main core to the pool and this may not always be possible. This function
        allows you to create objects within the pool itself to avoid this issue.
        """
        pass

    def getAdditionalHeaders(self):
        """
        Returns a list of strings. Subclasses can add in additional columns per
        record.

        This method MAY be overridden to provide the names for those
        columns.
        """
        return []

    @abc.abstractmethod
    def handleRecord(self, inputData):
        """
        Returns a list [anomalyScore, *]. It is required that the first
        element of the list is the anomalyScore. The other elements may
        be anything, but should correspond to the names returned by
        getAdditionalHeaders().

        This method MUST be overridden by subclasses
        """
        raise NotImplementedError

    def run(self):
        """
        Main function that is called to collect anomaly scores for a given file.
        """

        detectorValues = []
        for i, row in enumerate(self.dataSet):

            detectorValue = self.handleRecord(row)[0]

            # Make sure anomalyScore is between 0 and 1
            if not 0 <= detectorValue <= 1:
                raise ValueError(
                    f"anomalyScore must be a number between 0 and 1. "
                    f"Please verify if '{self.handleRecord.__qualname__}' method is "
                    f"returning a value between 0 and 1")

            detectorValues.append(detectorValue)

            """"# Progress report
            if (i % 1000) == 0:
                print(".", end=' ')
                sys.stdout.flush()
            """

        return detectorValues
