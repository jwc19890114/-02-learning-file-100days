'''
使用TensorFlow对逻辑回归进行复现，
数据集为coursera ex2data1.txt，通过成绩预测学生是否会被录取
'''
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ex2data1.txt', header=None)
train_data = df.values
# 前两列为输入X，第三列为输出
train_X = train_data[:, :-1]
train_y = train_data[:, -1:]
feature_num = len(train_X[0])
sample_num = len(train_X)


# print("Size of train_X: {}x{}".format(sample_num, feature_num))
# print("Size of train_y: {}x{}".format(len(train_y), len(train_y[0])))

def logistic_regression():
    global y, b
    # 模型设计
    # 使用 TensorFlow 定义两个变量用来存放我们的训练用数据。placeholder为占位符
    X = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    # 需要训练的参数W和b
    W = tf.Variable(tf.zeros([feature_num, 1]))
    b = tf.Variable([-.9])
    # 使用tensorflow表达损失函数
    '''
    表达损失函数是分三步进行的：先分别将求和内的两部分表示出来，
    再将它们加和并和外面的常数m进行运算，最后对这个向量进行求和，
    便得到了损失函数的值。
    '''
    db = tf.matmul(X, tf.reshape(W, [-1, 1])) + b
    hyp = tf.sigmoid(db) #损失函数，sigmoid
    cost0 = y * tf.log(hyp)
    cost1 = (1 - y) * tf.log(1 - hyp)
    cost = (cost0 + cost1) / -sample_num
    loss = tf.reduce_sum(cost)
    '''
    定义优化的方法，0,001是学习率
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)
    '''
    train=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    '''
    训练模型
    定义variable初始化
    gpu分配显存
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    '''
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    feed_dict = {X: train_X, y: train_y}
    for step in range(100000):
        sess.run(train, feed_dict)
        if step % 100 == 0:
            print(step, sess.run(W).flatten(), sess.run(b).flatten())


if __name__ == '__main__':
    '''
    运行结果
    999900 [0.12887694 0.12310407] [-15.472656]
    99900 [0.04858239 0.04162483] [-5.248103]
    '''
    # logistic_regression(train_X, train_y)

    w = [0.04858239, 0.04162483]
    b = -5.248103
    x1 = train_data[:, 0]
    x2 = train_data[:, 1]
    y = train_data[:, -1:]

    '''
    其中，我们用红色的x代表没有被录取，用绿色的o代表被录取。
    其次我们将训练得出的决策边界XW+b=0表示到图表上：
    '''

    for x1p, x2p, yp in zip(x1, x2, y):
        if yp == 0:
            plt.scatter(x1p, x2p, marker='x', c='r')
        else:
            plt.scatter(x1p, x2p, marker='o', c='g')

    x = np.linspace(20, 100, 10)
    y = []
    for i in x:
        y.append((i * -w[1] - b) / w[0])

    plt.plot(x, y)
    plt.show()
