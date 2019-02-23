'''
使用Tensorflow完成线性回归
## 使用Tensorflow训练一个线性模型你和数据，[blog地址](http://www.machinelearninguru.com/deep_learning/tensorflow/machine_learning_basics/linear_regresstion/linear_regression.html)
## 介绍
线性回归的作用（定义就是为因变量和自变量的相关性构建模型，**预测**）。  
线性回归算法的主要优势是简单粗暴（simplicity）。  
本文中构建并可视化线性回归模型
## 处理流程
为了训练模型，Tensorflow循环遍历数据，找到最佳的线来拟合数据。  
通过设计适当的优化问题来评估X和Y的线性关系，需要适当的损失函数。  
数据集来源于CS20SI：基于Tensorflow的深度学习研究。
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt

# ## 获取数据
# 原答案用的是xlrd，我在这里使用pandas改写了
'''
DATA_FILE = "./fire_theft.xls"
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
num_samples = sheet.nrows - 1
print(data)
'''
data = pd.read_excel(r'./fire_theft.xlsx')
# 注意iloc获取数值的写法[:,:2]，第一个“:”表示获取所有行数；第二个“:2”表示获取前2列
data = data.iloc[:, :2].values
# print(data)
tf.app.flags.DEFINE_integer(
    'num_epochs', 50, 'The number of epochs for training the model. Default=50')
# Store all elements in FLAG structure!
FLAGS = tf.app.flags.FLAGS

# ## 初始化参数
W = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')


# ## 定义重要函数
def inputs():
    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')
    return X, Y


def inference(X):
    '''
    计算线性结果，输入X为Input
    :return: 
    '''
    return X * W + b


def loss(X, Y):
    '''
    通过对比预测值和实际值来计算损失函数，
    :param X: 输入
    :param Y: 标签
    :return: 损失值
    '''
    Y_predicted = inference(X)
    loss = tf.squared_difference(Y, Y_predicted)
    return loss


def train(loss):
    '''
    使用梯度下降进行训练
    :param loss: 
    :return: opt
    '''
    lr = 0.0001
    opt = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    return opt


### 对数据进行循环遍历
with tf.Session() as sess:
    # 初始化变量W和b
    sess.run(tf.global_variables_initializer())
    # 获取输入张量
    X, Y = inputs()

    train_loss = loss(X, Y)
    train_op = train(train_loss)

    # for epoch in range(FLAGS.num_epochs):
    for epoch in range(40):
        for x, y in data:
            train_op = train(train_loss)
            loss_value, _ = sess.run([train_loss, train_op], feed_dict={X: x, Y: y})
        print('epoch %d, loss=%f' % (epoch + 1, loss_value))

        wcoeff, bias = sess.run([W, b])
        print(wcoeff, bias)

### 结果可视化
Input_values = data[:, 0]
Labels = data[:, 1]
Prediction_values = data[:, 0] * wcoeff + bias
plt.plot(Input_values, Labels, 'ro', label='main')
plt.plot(Input_values, Prediction_values, label='Predicted')

# Saving the result.
plt.legend()
plt.savefig('plot40.png')
plt.close()
