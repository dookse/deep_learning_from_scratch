import matplotlib.pyplot as plt
import tensorflow as tf

x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]
learning_rate = 0.002

plt.plot(x_data, y_data, 'o')
plt.ylim(0, 8)

W = tf.Variable(2.0)
b = tf.Variable(0.5)

for i in range(101):
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))
        plt.plot(x_data, hypothesis.numpy(), 'r-')
        plt.plot(x_data, y_data, 'o')
        plt.ylim(0, 8)
        plt.show()

print()

# predict
print(W * 5 + b)
print(W * 2.5 + b)
