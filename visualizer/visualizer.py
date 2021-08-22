import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def visualize(x_test, y_test, y_predicted, file_name, algo_name):
    y_predicted = np.where(y_predicted == 0, 1, 0)
    # red predicted, green actual anomalies
    x_axis = np.arange(x_test.size)
    plt.plot(x_axis, x_test, color='blue')
    plt.plot(x_axis[y_predicted == 0], x_test[y_predicted == 0], 'ro')
    plt.show()
    plt.plot(x_axis, x_test, color='blue')
    plt.plot(x_axis[y_test == 0], x_test[y_test == 0], 'go')
    plt.show()

    print("Data file: ", file_name)
    print("Algorithm Name: ", algo_name)
    print("Acurracy: ", accuracy_score(y_test, y_predicted))
    print("Matrix = tn, fp, fn, tp: ", confusion_matrix(y_test, y_predicted).ravel())
