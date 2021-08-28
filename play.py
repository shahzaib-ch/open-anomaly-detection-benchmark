import numpy as np
import pandas as pd

from helper.LabelsHelper import replace

a = np.array([[1, 0, 1, -1], [1, 0, 1, -1]])

df = pd.DataFrame(a, columns=["one", "two", "three", "four"])
print([df.columns != "one"])


