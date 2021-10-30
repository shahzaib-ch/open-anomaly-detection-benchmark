import numpy as np


class ResultWindowLabeler:

    def __init__(self, labels_test, labels_detected):
        self.labels_test = labels_test
        self.labels_detected = labels_detected

    def mark_whole_window_if_any_point_detected(self):
        """
        Marks all actual consecutive anomalies as detected if any of point is detected by algorithm
        :return: new labels numpy array,
        position of detected anomaly point in set of consecutive actual anomaly points (percentage values in list)
        """
        anomaly_windows = self.find_anomaly_detected_windows()
        new_labels = np.copy(self.labels_detected)

        for window in anomaly_windows:
            start_index = int(window[0])
            end_index = int(window[1]) + 1
            new_labels[start_index:end_index] = 1

        return new_labels

    def find_anomaly_detected_windows(self):
        labels_t = self.labels_test
        labels_d = self.labels_detected
        anomalies_actual_and_detected = np.where(np.logical_and(labels_t, labels_d), -1, labels_t)

        anomaly_windows = []
        for index, element in enumerate(anomalies_actual_and_detected):
            if element != -1:
                continue
            front_count = self.find_front_consecutive_ones_count(index, anomalies_actual_and_detected)
            back_count = self.find_back_consecutive_ones_count(index, anomalies_actual_and_detected)
            anomaly_window_size = front_count + back_count + 1
            position = back_count + 1
            position_of_detected_anomaly = (position / anomaly_window_size) * 100

            start_index = index - back_count
            end_index = index + front_count
            anomaly_windows.append([start_index, end_index, position_of_detected_anomaly])

        return np.asarray(anomaly_windows)

    def find_front_consecutive_ones_count(self, index, anomalies_actual_and_detected):
        count = 0
        index = index + 1
        while index < len(anomalies_actual_and_detected):
            if anomalies_actual_and_detected[index] != 1:
                break
            count = count + 1
            index = index + 1
        return count

    def find_back_consecutive_ones_count(self, index, anomalies_actual_and_detected):
        count = 0
        index = index - 1
        while_end = len(anomalies_actual_and_detected) - index
        while index < while_end:
            if anomalies_actual_and_detected[index] != 1:
                break
            count = count + 1
            index = index - 1
        return count
