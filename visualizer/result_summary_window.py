import tkinter as tk
from tkinter.ttk import *

root = tk.Tk()

root.title('Anomaly benchmark')
root.geometry('1000x600+50+50')

# place a label on the root window
message = tk.Label(root, text="Select detectors and datasets for viewing results for them")

all_algorithms_check_button = Checkbutton(root, text="All algorithms")

all_datasets_check_button = Checkbutton(root, text="All datasets")

show_result_button = Button(root, text="Visualize results")

# packing all ui elements
for c in root.children:
    root.children[c].pack()

# keep the window displaying
root.mainloop()
