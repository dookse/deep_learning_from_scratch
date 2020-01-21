import numpy as np


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    # return -np.sum(t * np.log(y)) / batch_size
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def run():
    t = [2]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print(*(zip(t, y)))
    e1 = cross_entropy_error(np.array(y), np.array(t))
    print(e1)
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    e2 = cross_entropy_error(np.array(y), np.array(t))
    print(e2)


if __name__ == '__main__':
    run()
