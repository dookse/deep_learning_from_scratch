import numpy as np

from numerical_gradient import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def main():
    init_x = np.array([-3.0, 4.0])
    r = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
    print(r)
    print(float(r[0]))


if __name__ == '__main__':
    main()
