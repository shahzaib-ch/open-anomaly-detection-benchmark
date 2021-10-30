import pandas as pd
from sklearn.metrics import confusion_matrix

labels = [0, 1, 0, 1, 1, 0, 0, 0]
scores = [0, 0, 0, 1, 1, 0, 1, 0]

for el in labels:
    if el == 0:
        continue

    print(el)

