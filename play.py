from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize, MinMaxScaler
import numpy as np

scalar = MinMaxScaler()
l = np.asarray([-4.783, -5.983743, 0, -0.73483, -2.98374, 2, 3, 5, 4.6]).reshape(1, -1)
print(l)
m = normalize(l)
print(m)
