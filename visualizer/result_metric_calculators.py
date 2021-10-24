from sklearn.metrics import accuracy_score, f1_score

from visualizer.result_data_keys import ResultDataKey


def add_accuracy_to_df(result_data_frame):
    """
    Adds accuracy column in data frame of results
    """
    result_data_frame[ResultDataKey.accuracy] = result_data_frame.apply(lambda row: __calculate_accuracy_score(row),
                                                                        axis=1)
    return result_data_frame


def add_f1_score_to_df(result_data_frame):
    """
    Adds accuracy column in data frame of results
    """
    result_data_frame[ResultDataKey.f1_score] = result_data_frame.apply(lambda row: __calculate_f1_score(row),
                                                                        axis=1)
    return result_data_frame


def __calculate_f1_score(row):
    labels = row[ResultDataKey.labels_test]
    labels_detected = row[ResultDataKey.labels_detected]
    return f1_score(labels, labels_detected)


def __calculate_accuracy_score(row):
    labels = row[ResultDataKey.labels_test]
    labels_detected = row[ResultDataKey.labels_detected]
    return accuracy_score(labels, labels_detected)


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
