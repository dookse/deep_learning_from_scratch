import numpy as np

from cross_entropy import cross_entropy_error
from numerical_gradient import numerical_gradient
from softmax import softmax


class SimpleNet:
    def __init__(self) -> None:
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


def main():
    net = SimpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)

    print('argmax = {}'.format(np.argmax(p)))
    t = np.array([0, 0, 1])
    loss = net.loss(x, t)
    print(loss)

    dw = numerical_gradient(lambda w: net.loss(x, t), net.W)
    print(dw)


if __name__ == '__main__':
    main()
