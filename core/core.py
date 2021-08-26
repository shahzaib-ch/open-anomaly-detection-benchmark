import numpy as np
from sklearn.model_selection import train_test_split

from algorithms.algorithms import ALGORITHMS_DICTIONARY
from data.dataset_collector import number_of_files, DATA_DIRECTORY_PATH, data_from_file
from visualizer.visualizer import visualize


def run():
    for num in range(number_of_files):
        file_path = DATA_DIRECTORY_PATH + str(num + 1) + ".csv"
        X, y = data_from_file(file_path)
        # 1 is normal, 0 is outlier. Reversing this to match our dataset labels
        y = np.where(y == 0, 1, 0)
        y = np.where(y == -1, 0, 1)
        for algo_name, algo in ALGORITHMS_DICTIONARY.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, shuffle=False)
            algo.fit(X_train, y_train)
            y_predicted = algo.predict(X_test)
            visualize(X_test, y_test, y_predicted, file_path, algo_name)
