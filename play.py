import pandas as pd
from sklearn.metrics import confusion_matrix

labels = [0, 1, 0, 1, 1, 0, 0, 0]
scores = [0, 0, 0, 1, 1, 0, 1, 0]

sf = pd.DataFrame(scores, index="daa")
sf["hgsd"] = labels


print(sf.isna())
