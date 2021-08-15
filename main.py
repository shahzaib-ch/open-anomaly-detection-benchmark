import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope


def do_the_job():
    path = "/Users/shahzaib/Documents/Thesis/datasets/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_2.csv"
    df = pd.read_csv(path)

    X = df.iloc[:, 1:2].to_numpy()
    y = df.iloc[:, 2:3].to_numpy()
    y = y.reshape(y.size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, shuffle=False)

    algo = KMeans(n_clusters=2, random_state=20)
    algo.fit(X_train, y_train)

    y_pred = algo.predict(X_test)

    # 1 is normal, 0 is outlier. Revresing this to match our dataset lables
    y_pred = np.where(y_pred == 0, 1, 0)

    plt.plot(y_pred, color='green')

    plt.plot(y_test, color='blue')

    plt.plot(X_test, color='red')

    X_test_percent_change = pd.DataFrame(X_test).pct_change()
    plt.plot(X_test_percent_change, color='red')

    x_axis = np.arange(X_test.size)
    plt.plot(x_axis, X_test, color='blue')
    plt.plot(x_axis[y_pred == 1], X_test[y_pred == 1], 'ro')
    plt.plot(x_axis[y_test == 1], X_test[y_test == 1], 'go')

    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

    # (tn, fp, fn, tp)
    matrix.ravel()

    print("Acurracy: ", accuracy_score(y_test, y_pred))

    number_of_files = 1
    path = "/Users/shahzaib/Documents/Thesis/datasets/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_"
    algos = {
        "Kmeans": KMeans(n_clusters=2, random_state=20),
        "Elliptic Envelope": EllipticEnvelope(random_state=10)
    }

    def data_from_file(path):
        df = pd.read_csv(path)
        X = df.iloc[:, 1:2].to_numpy()
        y = df.iloc[:, 2:3].to_numpy()
        y = y.reshape(y.size)
        return X, y

    def visualize(X_train, X_test, y_train, y_test, y_pred, file_name, algo_name):
        y_pred = np.where(y_pred == 0, 1, 0)
        # red predicted, green actual anomalies
        x_axis = np.arange(X_test.size)
        plt.plot(x_axis, X_test, color='blue')
        plt.plot(x_axis[y_pred == 0], X_test[y_pred == 0], 'ro')
        plt.show()
        plt.plot(x_axis, X_test, color='blue')
        plt.plot(x_axis[y_test == 0], X_test[y_test == 0], 'go')
        plt.show()

        print("Data file: ", file_name)
        print("Algorithm Name: ", algo_name)
        print("Acurracy: ", accuracy_score(y_test, y_pred))
        print("Matrix = tn, fp, fn, tp: ", confusion_matrix(y_test, y_pred).ravel())

    for num in range(number_of_files):
        file_path = path + str(num + 1) + ".csv"
        X, y = data_from_file(file_path)
        # 1 is normal, 0 is outlier. Revresing this to match our dataset lables
        y = np.where(y == 0, 1, 0)
        y = np.where(y == -1, 0, 1)
        for algo_name, algo in algos.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, shuffle=False)
            algo.fit(X_train, y_train)
            y_pred = algo.predict(X_test)
            visualize(X_train, X_test, y_train, y_test, y_pred, file_path, algo_name)


if __name__ == '__main__':
    do_the_job()
