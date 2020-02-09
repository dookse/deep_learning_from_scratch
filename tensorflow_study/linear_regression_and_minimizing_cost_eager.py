import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.random.set_seed(0)  # for reproducibility

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])

# W = tf.Variable([5.0])
W = tf.Variable(tf.random.normal((1,), -100., 100.))

print('{:^5} | {:^10} | {:^10}'.format('step', 'cost', 'W'))

for step in range(300):
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)

    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(
            step, cost.numpy(), W.numpy()[0]))

plt.rcParams["figure.figsize"] = (8, 6)

W_values = np.linspace(-3, 5, num=15)
cost_values = []


def cost_func(W, X, Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))


for feed_W in W_values:
    curr_cost = cost_func(feed_W, X, Y)
    cost_values.append(curr_cost)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

plt.rcParams["figure.figsize"] = (8, 6)
plt.plot(W_values, cost_values, "b")
plt.ylabel('Cost(W)')
plt.xlabel('W')
plt.show()

print(5.0 * W)
print(2.5 * W)
