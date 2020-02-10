import tensorflow as tf

tf.random.set_seed(0)  # for reproducibility

X = [[1., 0., 3., 0., 5.],
     [0., 2., 0., 4., 0.]]
y = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random.uniform([1, 2], -10.0, 10.0))
b = tf.Variable(tf.random.uniform([1], -1.0, 1.0))

learning_rate = tf.Variable(0.001)

for i in range(1000 + 1):
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(W, X) + b
        cost = tf.reduce_mean(tf.square(hypothesis - y))

    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print('{:5} {:10.4f} {}'.format(i, cost.numpy(), *W.numpy()))
