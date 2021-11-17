import numpy as np
from numpy import NaN
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, precision_score, recall_score, \
    confusion_matrix

from helper.ResultWindowLabeler import ResultWindowLabeler
from visualizer.result_data_keys import ResultDataKey


def add_detected_labels_to_df(result_data_frame, anomaly_threshold, use_windows):
    """
    Adds accuracy column in data frame of results
    """
    print("making window lables")
    result_data_frame[ResultDataKey.labels_detected] = result_data_frame.apply(
        lambda row: __calculate_detected_labels(row, anomaly_threshold, use_windows), axis=1)
    return result_data_frame


def __calculate_detected_labels(row, anomaly_threshold, use_windows):
    scores = row[ResultDataKey.anomaly_scores_by_algorithm]
    detected_labels = np.where(scores >= anomaly_threshold, 1, 0)
    if use_windows:
        labeler = ResultWindowLabeler(row[ResultDataKey.labels_test], detected_labels)
        return labeler.mark_whole_window_if_any_point_detected()
    return detected_labels


def add_accuracy_to_df(result_data_frame):
    """
    Adds accuracy column in data frame of results
    """
    result_data_frame[ResultDataKey.accuracy] = result_data_frame.apply(lambda row: __calculate_accuracy_score(row),
                                                                        axis=1)
    return result_data_frame


def __calculate_accuracy_score(row):
    print("........")
    labels = row[ResultDataKey.labels_test]
    labels_detected = row[ResultDataKey.labels_detected]
    return accuracy_score(labels, labels_detected) * 100


def add_f1_score_to_df(result_data_frame):
    """
    Adds accuracy column in data frame of results
    """
    result_data_frame[ResultDataKey.f1_score] = result_data_frame.apply(lambda row: __calculate_f1_score(row),
                                                                        axis=1)
    return result_data_frame


def __calculate_f1_score(row):
    print("........")
    labels = row[ResultDataKey.labels_test]
    labels_detected = row[ResultDataKey.labels_detected]
    tn, fp, fn, tp = confusion_matrix(labels, labels_detected, labels=[0, 1]).ravel()
    if tn == 0 and fp == 0 and fn == 0:
        return 100.0
    if (tp + fp) == 0 or (tp + fn) == 0:
        return NaN
    return f1_score(labels, labels_detected, zero_division=1, labels=[0, 1]) * 100


def add_average_precision_score_to_df(result_data_frame):
    """
    Adds accuracy column in data frame of results
    """
    result_data_frame[ResultDataKey.average_precision_score] = result_data_frame.apply(
        lambda row: __calculate_average_precision_score_labels(row),
        axis=1)
    return result_data_frame


def __calculate_average_precision_score_labels(row):
    print("........")
    scores = row[ResultDataKey.anomaly_scores_by_algorithm]
    labels = row[ResultDataKey.labels_test]
    try:
        precision = average_precision_score(labels, scores) * 100
    except:
        precision = NaN
    return precision


def add_recall_score_to_df(result_data_frame):
    """
    Adds accuracy column in data frame of results
    """
    result_data_frame[ResultDataKey.recall] = result_data_frame.apply(
        lambda row: __calculate_recall_score_labels(row),
        axis=1)
    return result_data_frame


def __calculate_recall_score_labels(row):
    print("........")
    scores = row[ResultDataKey.labels_detected]
    labels = row[ResultDataKey.labels_test]
    tn, fp, fn, tp = confusion_matrix(labels, scores, labels=[0, 1]).ravel()
    if tn == 0 and fp == 0 and fn == 0:
        return 100.0
    if (tp + fn) == 0:
        return NaN
    precision = recall_score(labels, scores, zero_division=1, labels=[0, 1]) * 100
    return precision


def add_precision_score_to_df(result_data_frame):
    """
    Adds accuracy column in data frame of results
    """
    result_data_frame[ResultDataKey.precision] = result_data_frame.apply(
        lambda row: __calculate_precision_score_labels(row),
        axis=1)
    return result_data_frame


def __calculate_precision_score_labels(row):
    print("........")
    scores = row[ResultDataKey.labels_detected]
    labels = row[ResultDataKey.labels_test]
    tn, fp, fn, tp = confusion_matrix(labels, scores, labels=[0, 1]).ravel()
    if tn == 0 and fp == 0 and fn == 0:
        return 100.0
    if (tp + fp) == 0:
        return NaN
    precision = precision_score(labels, scores, zero_division=1, labels=[0, 1]) * 100
    return precision


def add_subfolder_name_to_df(result_data_frame):
    """
    Adds subfolder column in data frame of results
    """
    result_data_frame[ResultDataKey.subfolder] = result_data_frame.apply(
        lambda row: __get_subfolder_name_from_path(row),
        axis=1)
    return result_data_frame


def __get_subfolder_name_from_path(row):
    path = row[ResultDataKey.file_path]
    dataset_name = row[ResultDataKey.dataset_name]
    dataset_name_index = path.rfind(dataset_name)
    slash_index = path.rfind("/")
    new_path = path[dataset_name_index:slash_index]
    slash_index = new_path.find("/")
    return new_path[slash_index + 1:]
