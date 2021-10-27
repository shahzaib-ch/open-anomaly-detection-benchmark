import numpy as np
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()
l = np.asarray([-4.783, -5.983743, 0, -0.73483, -2.98374, 2, 3, 5, 4.6]).reshape(-1, 1)
scalar.fit(l)
m = scalar.transform(l)
print(m)
