import tensorflow as tf
import numpy as np

# ReLU激活函数
'''
在代码中生成权重，定义了一张图
初始化W
生成输入的x m*784
生成图运算
'''
b = tf.Variable(tf.zeros((100,)))
w = tf.variable(tf.random_uniform((784, 100), -1, 1))
x = tf.placeholder(tf.float32, (100, 784))
h = tf.nn.relu(tf.matul(x, w) + b)

'''
执行上面的图。
我们可以通过session将这张图部署到某个执行环境（CPU、GPU）上去。
session就是到某个软硬件执行环境的绑定
'''
sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(h, {x: np.random.random(100, 784)})

prediction = tf.nn.softmax(...)  # Output of neural network
label = tf.placeholder(tf.float32, [100, 10])
cross_entropy = -tf.reduce_sum(label * tf.log(prediction), axis=1)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
'''
训练模型
sess.run(train_step, feeds)

'''

for i in range(1000):
    batch_x, batch_label = data.next_batch()
    sess.run(train_step, feed_dict={x: batch_x, label: batch_label})
