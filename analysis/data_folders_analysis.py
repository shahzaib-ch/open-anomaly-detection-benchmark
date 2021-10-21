from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting._matplotlib import scatter_matrix


def visualize_overview():
    base_path = "/Users/shahzaib/PycharmProjects/benchmark/data/datasets/"

    labels = ['nab', 'odd', 'ucr', 'yahoo']

    avg_instance = []
    datasets_count = []

    odd_features = []

    for folder in labels:
        path = base_path + folder
        number_of_datasets, avg_instances_per_file, features = data_from_one_data_folder(path)
        avg_instance.append(avg_instances_per_file)
        datasets_count.append(number_of_datasets)

        if folder == "odd":
            odd_features = features

    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(3)
    ax[0].bar(x, datasets_count, width)

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].bar_label(ax[0].containers[0], padding=5)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    add_annotation(ax[0], '(a) Number of datasets per folder')

    ax[1].bar(x, avg_instance, width)

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].bar_label(ax[1].containers[0], padding=5)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    add_annotation(ax[1], '(b) Average number of data instances per dataset')

    ax[2].bar(np.arange(len(odd_features)), odd_features)

    ax[2].tick_params(bottom=False, labelbottom=False)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    add_annotation(ax[2], '(c) Features of datasets in odd folder')

    fig.tight_layout()

    plt.show()


def data_from_one_data_folder(path):
    datasets = Path(path).rglob('*.csv')
    number_of_datasets = 0
    features = []
    instances_in_datasets = []
    for dataset in datasets:
        number_of_datasets = number_of_datasets + 1
        df = pd.read_csv(str(dataset))
        instances_count = df.size
        features.append(df.columns.size - 2)
        instances_in_datasets.append(instances_count)

    avg_instances_per_file = int(np.average(instances_in_datasets))

    return number_of_datasets, avg_instances_per_file, features


def add_annotation(ax, text):
    ax.set_xlabel(text, labelpad=10)


def visualize_4_datasets():
    d1 = "/Users/shahzaib/PycharmProjects/benchmark/data/datasets/odd/annthyroid.csv"
    d2 = "/Users/shahzaib/PycharmProjects/benchmark/data/datasets/nab/realTraffic/occupancy_6005.csv"
    d3 = "/Users/shahzaib/PycharmProjects/benchmark/data/datasets/ucr/054_UCR_Anomaly_DISTORTEDWalkingAceleration5_2700_5920_5979.csv"
    d4 = "/Users/shahzaib/PycharmProjects/benchmark/data/datasets/yahoo/A1Benchmark/real_14.csv"

    d1 = pd.read_csv(d1)
    d2 = pd.read_csv(d2)
    d3 = pd.read_csv(d3)
    d4 = pd.read_csv(d4)

    d1 = d1.iloc[:, 1:-2]
    d2 = d2.iloc[:, 1:-1]
    d3 = d3.iloc[:, 1:-1]
    d4 = d4.iloc[:, 1:-1]

    fig, ax = plt.subplots(4)

    add_annotation(ax[0], '(a) ODDS multidimensional dataset')
    add_annotation(ax[1], '(b) NAB dataset')
    add_annotation(ax[2], '(c) UCR dataset')
    add_annotation(ax[3], '(d) Yahoo dataset')

    ax[0].plot(d1.to_numpy())
    ax[1].plot(d2.to_numpy())
    ax[2].plot(d3.to_numpy())
    ax[3].plot(d4.to_numpy())

    fig.subplots_adjust(hspace=0)

    fig.tight_layout()

    plt.show()


def visualize_multi_dimensional_file():
    d1 = "/Users/shahzaib/PycharmProjects/benchmark/data/datasets/odd/annthyroid.csv"
    d1 = pd.read_csv(d1)
    d1 = d1.iloc[:, 1:-2]
    scatter_matrix(d1, alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()


# visualize_multi_dimensional_file()


visualize_4_datasets()
