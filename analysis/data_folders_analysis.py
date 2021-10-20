from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_overview():
    base_path = "/Users/shahzaib/PycharmProjects/benchmark/data/datasets/"

    labels = ['nab', 'odd', 'ucr', 'yahoo']

    avg_instance = []
    datasets_count = []

    for folder in labels:
        path = base_path + folder
        number_of_datasets, avg_instances_per_file = data_from_one_data_folder(path)
        avg_instance.append(avg_instances_per_file)
        datasets_count.append(number_of_datasets)

    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(2)
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

    fig.tight_layout()

    plt.show()


def data_from_one_data_folder(path):
    datasets = Path(path).rglob('*.csv')
    number_of_datasets = 0
    instances_in_datasets = []
    for dataset in datasets:
        number_of_datasets = number_of_datasets + 1
        instances_count = pd.read_csv(str(dataset)).size
        instances_in_datasets.append(instances_count)

    avg_instances_per_file = int(np.average(instances_in_datasets))

    return number_of_datasets, avg_instances_per_file


def add_annotation(ax, text):
    ax.set_xlabel(text, labelpad=10)


visualize_overview()
