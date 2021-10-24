import pickle

import matplotlib.pyplot as plt

with open(
        'result/Auto Encoder/odd/mnist',
        'rb') as f:
    data = pickle.load(f)
    l = data["data"]["data"]
    p = l["input_instances_train"]
    plt.plot(p)
    plt.show()
