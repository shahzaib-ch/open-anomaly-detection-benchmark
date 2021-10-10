import pickle


def read_dictionary_from_file(file_path):
    """
    Returns dictionary after reading from file
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_dictionary_to_file(dictionary, file_path):
    """
    Saves dictionary to file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)


def round_xy_coordinates(x, y):
    """
    round float, example -0.5 to 0.5 to 0
    """
    x_rounded = int(round(x))
    y_rounded = int(round(y))
    return x_rounded, y_rounded
