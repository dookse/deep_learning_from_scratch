import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.maximum(0, x)


x = np.arange(-7.0, 7.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1, 6)
plt.show()
