import tkinter as tk
from tkinter import TOP
from tkinter.ttk import *


class DatasetResultSummaryWindow:

    def __init__(self, view_dataset_command, view_dataset_labels, visualize_dataset_with_anomalies):
        self.view_dataset_command = view_dataset_command
        self.view_dataset_labels_command = view_dataset_labels
        self.view_dataset_with_anomalies_command = visualize_dataset_with_anomalies

    def show_window(self):
        root = tk.Tk()

        root.title('Open Anomaly Detection Benchmark')
        root.geometry('1000x600+50+50')

        frame = Frame(root)
        frame.pack()
        # place a label on the root window
        message = tk.Label(frame, text="Different ways to visualize detector performance on a dataset file")
        message.pack(side=TOP, pady=30)

        view_dataset_button = tk.Button(frame, text="View dataset file", command=self.view_dataset_command)
        view_dataset_button.pack(side=TOP, pady=5, ipady=3)

        view_dataset_labels = tk.Button(frame,
                                        text="View actual labels with detected labels",
                                        command=self.view_dataset_labels_command)
        view_dataset_labels.pack(side=TOP, pady=5, ipady=3)

        view_dataset_with_anomalies = tk.Button(frame,
                                                text="View dataset with anomalies",
                                                command=self.view_dataset_with_anomalies_command)
        view_dataset_with_anomalies.pack(side=TOP, pady=5, ipady=3)

        # packing all ui elements
        for c in root.children:
            root.children[c].pack()

        # keep the window displaying
        root.mainloop()
