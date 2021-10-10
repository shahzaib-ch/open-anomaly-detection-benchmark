import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.imshow(np.random.rand(10, 10), picker=True)


def onclick(event):
    print(event)


cid = fig.canvas.mpl_connect('pick_event', onclick)

plt.show()
