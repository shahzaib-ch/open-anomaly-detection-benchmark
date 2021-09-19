import pandas
from matplotlib import pyplot as plt

df = pandas.read_csv("/Users/shahzaib/PycharmProjects/benchmark/data/datasets/nab/artificialNoAnomaly/art_flatline.csv")
plt.plot(df["value"])
plt.show()

