import numpy as np

from cross_entropy import cross_entropy_error
from numerical_gradient import numerical_gradient
from sigmoid import sigmoid
from softmax import softmax


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01) -> None:
        self.params = {
            'w1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'w2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }

    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        loss_w = lambda w: self.loss(x, t)

        grads = {
            'w1': numerical_gradient(loss_w, w1),
            'b1': numerical_gradient(loss_w, b1),
            'w2': numerical_gradient(loss_w, w2),
            'b2': numerical_gradient(loss_w, b2)
        }
        return grads
