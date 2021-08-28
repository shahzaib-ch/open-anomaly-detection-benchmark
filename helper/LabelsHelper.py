import numpy as np


def replace(numpy_array, element_to_replace, replacement):
    """
    Takes a numpy array and replace an element with replacement and returns numpy array
    :param numpy_array:
    :param element_to_replace:
    :param replacement:
    :return: numpy array
    """
    return np.where(numpy_array == element_to_replace, replacement, numpy_array)
