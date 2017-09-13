import tensorflow as tf

# Model definition
# model = ax^2 + bx + c
x = tf.placeholder(tf.float32)
a = tf.Variable([0.1], dtype=tf.float32)
b = tf.Variable([0.1], dtype=tf.float32)
c = tf.Variable([0.1], dtype=tf.float32)

model = a * tf.square(x) + b * x + c

# Loss and training functions definition
y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(model - y))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

# Training data
x_train = [0, 1, 2, 3, 4]
y_train = [0, 3, 10, 21, 36]

# Training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(5000):
    sess.run(train, {x: x_train, y: y_train})

    # evaluate training accuracy
    curr_a, curr_b, curr_c, curr_loss = sess.run(
        [a, b, c, loss], {x: x_train, y: y_train})
    print("a: %s, b: %s, c: %s, loss: %s" %
            (curr_a[0], curr_b[0], curr_c[0], curr_loss))

