import pandas as pd
from sklearn.metrics import confusion_matrix

from helper.labels_helper import unpickle_result

labels = "/Users/shahzaib/PycharmProjects/benchmark/result/One-class SVM/yahoo/A2Benchmark/synthetic_56"
unpickle_result(labels, "totot")