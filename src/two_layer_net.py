from collections import OrderedDict

from layers import *
from numerical_gradient import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01) -> None:
        self.params = {
            'w1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'w2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }
        self.layers = OrderedDict({
            'affine1': Affine(self.params['w1'], self.params['b1']),
            'relu1': Relu(),
            'affine2': Affine(self.params['w2'], self.params['b2']),
        })
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        def loss_w(w):
            return self.loss(x, t)

        grads = {'w1': numerical_gradient(loss_w, self.params['w1']),
                 'b1': numerical_gradient(loss_w, self.params['b1']),
                 'w2': numerical_gradient(loss_w, self.params['w2']),
                 'b2': numerical_gradient(loss_w, self.params['b2'])}

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = reversed(list(self.layers.values()))
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {'w1': self.layers['affine1'].dw,
                 'b1': self.layers['affine1'].db,
                 'w2': self.layers['affine2'].dw,
                 'b2': self.layers['affine2'].db}
        return grads
