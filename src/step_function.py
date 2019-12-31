import matplotlib.pylab as plt
import numpy as np


def step_function(x):
    if type(x) == np.ndarray:
        y = x > 0
        return y.astype(np.int)
    else:
        return int(x > 0)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
