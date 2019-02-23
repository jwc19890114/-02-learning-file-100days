"""
**源代码来自莫烦python(https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/4-1-tensorboard1/)**  
**今日重点**
- 读懂教程中代码，手动重写一遍，并生成可视化网络图
- 修改《机器学习100天-Day12Tensorflow新手教程5（RNN）》中的代码，同样生成可视化网络图
Tensorboard是一个神经网络可视化工具，通过使用本地服务器在浏览器上查看神经网络训练日志，生成相应的可是画图，帮助炼丹师优化神经网络。  
油管上有单迪伦·马内在2017年做的汇报，很惊艳。主要包括了以下主要功能  
- 可视化网络
- 可视化训练过程
- 多模型效果可视化对比

先看一下教程提供的原始代码（不包括tensorboard构造），就是一个两层（包括输出）的线性回归网络。
~~~python
from __future__ import print_function
import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
~~~
要让神经网络中的每一个元素在可视化界面中显示，就需要对图和参数进行命名，需要修改两个部分  
- 定义图：with tf.name_scope()（ 里面写名字，下面用缩进） 
- 定义参数：在每一个参数后面增加一个name属性，如***xs = tf.placeholder(tf.float32, [None, 1])*** -> ***xs = tf.placeholder(tf.float32, [None, 1], name='x_input')***

#### 隐藏层
~~~python
def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        # add one more layer and return the output of this layer
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
    return outputs
~~~
#### 输入层
~~~python
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='X_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
~~~

#### 损失函数和训练
~~~python
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
~~~
#### 最后是保存数据
~~~python
writer = tf.summary.FileWriter("logs/", sess.graph)
~~~
完成之后，在pycharm的Terminal中输入‘tensorboard --logdir=/Users/01/Desktop/机器学习作业/sklearn+tensorflow/logs’，然后再chrome中输入‘http://localhost:6006’即可查看整个神经网络可视化结果，注意，因为没有数据的输入，现在仅能查看神经网络的结构。

### 修改自己的神经网络
这个神经网络是之前生成的一个RNN网络，我删去训练部分，看一下，其中还有一些备注。
~~~python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

### tensorflow警告记录，可以避免在运行文件时出现红色警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# Parser
'''
argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。argparse模块的作用是用于解析命令行参数。
我们很多时候，需要用到解析命令行参数的程序。
当运行参数不正确需要调用时会打印出描述信息，本例中为“Creating Classifier”
'''
parser = argparse.ArgumentParser(description='Creating Classifier')

'''
### 优化标记
- 学习速率:0.001
- seed:111
'''

tf.app.flags.DEFINE_float('learning_rate', default=0.001, help='initial learning rate')
tf.app.flags.DEFINE_integer('seed', default=111, help='seed')

'''
### 训练标记
- batch_size（批尺寸）:128
- num_epoch（全数据集迭代次数）:10
- batch_per_log:10
#### Batch_Size（批尺寸）是机器学习中一个重要参数。
Batch决定的是下降的方向。如果数据集比较小，可以采用全数据集（full batch learning）的形式。
这样做至少有 2 个好处：
- 全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。
- 由于不同权重的梯度值差别巨大，因此选取一个全局的学习率很困难。Full Batch Learning 可以使用 Rprop 只基于梯度符号并且针对性单独更新各权值。
但是面对更大的数据集，载入数据和梯度修正值都会是问题
所以要在合理的范围内增大batch_size，能够更有效利用内存、提高数据完整性
#### num_epoch（全数据集迭代次数）
'''
tf.app.flags.DEFINE_integer('batch_size', default=128, help='Batch size for training')
tf.app.flags.DEFINE_integer('num_epoch', default=10, help='Number of training iterations')
tf.app.flags.DEFINE_integer('batch_per_log', default=10, help='Print the log at what number of batches?')

'''
### 模型标记
隐藏层中神经元为128个
'''
tf.app.flags.DEFINE_integer('hidden_size', default=128, help='Number of neurons for RNN hidden layer')

'''
tf.app.flags可以认为是对模块argparse的简单封装，它实现了python-gflags的一个功能子集。
请注意，此模块目前封装在一起，主要用于编写演示应用程序，并且在技术上不是公共API的一部分，将来有发生更改的可能性。
建议您使用argparse或您喜欢的任何其他代码库实现自己的标志解析。
这里类似args可以调用之前存储的所有元素信息
- args.seed
- args = tf.app.flags.FLAGS
- args.learning_rate
- args.batch_size
- args.hidden_size
- args.seed
- args.batch_per_log
- args.num_epoch
在FLAG结构中存储所有元素
'''
args=tf.flags.FLAGS

# Reset the graph set the random numbers to be the same using "seed"
tf.reset_default_graph()
tf.set_random_seed(args.seed)
np.random.seed(args.seed)

# Divide 28x28 images to rows of data to feed to RNN as sequantial information
step_size = 28
input_size = 28
output_size = 10

# Input tensors
X = tf.placeholder(tf.float32, [None, step_size, input_size])
y = tf.placeholder(tf.int32, [None])

'''
### 实现RNN
rnn_cell:是Tensorflow中实现RNN的基本单元。每个rnn_cell都有一个call方法。  
每调用一次RNNCell的call方法，就相当于在时间上“推进了一步”
但是对于单个RNNCell调用call函数进行运算时，只是在序列时间上前进了一步。比如使用x1、h0得到h1，通过x2、h1得到h2等。
eg：如果序列长度为10，就要调用10次call函数，比较麻烦。
TensorFlow提供了一个tf.nn.dynamic_rnn函数，使用该函数就相当于调用了n次call函数。即通过{h0,x1, x2, …., xn}直接得{h1,h2…,hn}。
输入数据的格式为(batch_size, time_steps, input_size)  
- time_steps表示序列本身的长度，如在Char RNN中，长度为10的句子对应的time_steps就等于10。
- input_size就表示输入数据单个序列单个时间维度上固有的长度。
- X为上面初始化的占位符
- 已经定义好了一个RNNCell，调用该RNNCell的call函数time_steps次
最后得到的结果中有两个output和state
- output就是time_steps步里所有的输出。形状为(batch_size, time_steps, cell.output_size)。
- state是最后一步的隐状态，形状为(batch_size, cell.state_size)
'''
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=args.hidden_size)
# print(cell.state_size) # =>128
output, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

'''
### 前向传播和损失计算
tf.layers.dense: 构建稠密全连接层。输入参数是神经元数目和激活函数。
tf.nn.sparse_softmax_cross_entropy_with_logits: 在内部算得softmax函数值后，继续计算交叉熵
'''
logits = tf.layers.dense(state, output_size)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)

'''
### 优化和预测
tf.nn.in_top_k: 用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量
tf.nn.in_top_k(prediction, target, K):prediction就是表示预测的结果，大小是预测样本的数量乘以输出的维度，类型是tf.float32等。
target就是实际样本类别的标签，大小就是样本数量的个数。K表示每个样本的预测结果的前K个最大的数里面是否含有target中的值。
一般取1，即是说预测中元素内前一个元素是否相同，如果K值为2，就是前2个。
'''
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)
prediction = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

'''
### 数据处理
'''
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets(".\MNIST_data/")
# X_test = mnist.test.images # X_test shape: [num_test, 28*28]
# X_test = X_test.reshape([-1, step_size, input_size])
# y_test = mnist.test.labels

# 初始化变量
init = tf.global_variables_initializer()

# 生成追踪list。
loss_train_list = []
acc_train_list = []

### 训练模型
with tf.Session() as sess:
    sess.run(init)
~~~
下面列出修改的部分
~~~python
# Input tensors
with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, [None, step_size, input_size],name='X_input')
    y = tf.placeholder(tf.int32, [None],name='y_input')

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=args.hidden_size)
# print(cell.state_size) # =>128
output, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

'''
### 前向传播和损失计算
tf.layers.dense: 构建稠密全连接层。输入参数是神经元数目和激活函数。
tf.nn.sparse_softmax_cross_entropy_with_logits: 在内部算得softmax函数值后，继续计算交叉熵
'''
logits = tf.layers.dense(state, output_size)

with tf.name_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("loss", loss)

'''
### 优化和预测
tf.nn.in_top_k: 用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量
tf.nn.in_top_k(prediction, target, K):prediction就是表示预测的结果，大小是预测样本的数量乘以输出的维度，类型是tf.float32等。
target就是实际样本类别的标签，大小就是样本数量的个数。K表示每个样本的预测结果的前K个最大的数里面是否含有target中的值。
一般取1，即是说预测中元素内前一个元素是否相同，如果K值为2，就是前2个。
'''
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

with tf.name_scope('accuracy'):
    prediction = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
summ = tf.summary.merge_all()

# 初始化变量
init = tf.global_variables_initializer()

# 生成追踪list。
loss_train_list = []
acc_train_list = []

### 训练模型
sess=tf.Session()
sess.run(init)
writer = tf.summary.FileWriter('./logs/rnn',sess.graph)
~~~

"""
from __future__ import print_function
import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        # add one more layer and return the output of this layer
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
    return outputs


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='X_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

writer = tf.summary.FileWriter("./logs/layer1", sess.graph)



# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# import os
#
# ### tensorflow警告记录，可以避免在运行文件时出现红色警告
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
#
# # Parser
# '''
# argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。argparse模块的作用是用于解析命令行参数。
# 我们很多时候，需要用到解析命令行参数的程序。
# 当运行参数不正确需要调用时会打印出描述信息，本例中为“Creating Classifier”
# '''
# parser = argparse.ArgumentParser(description='Creating Classifier')
#
# '''
# ### 优化标记
# - 学习速率:0.001
# - seed:111
# '''
#
# tf.app.flags.DEFINE_float('learning_rate', default=0.001, help='initial learning rate')
# tf.app.flags.DEFINE_integer('seed', default=111, help='seed')
#
# '''
# ### 训练标记
# - batch_size（批尺寸）:128
# - num_epoch（全数据集迭代次数）:10
# - batch_per_log:10
# #### Batch_Size（批尺寸）是机器学习中一个重要参数。
# Batch决定的是下降的方向。如果数据集比较小，可以采用全数据集（full batch learning）的形式。
# 这样做至少有 2 个好处：
# - 全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。
# - 由于不同权重的梯度值差别巨大，因此选取一个全局的学习率很困难。Full Batch Learning 可以使用 Rprop 只基于梯度符号并且针对性单独更新各权值。
# 但是面对更大的数据集，载入数据和梯度修正值都会是问题
# 所以要在合理的范围内增大batch_size，能够更有效利用内存、提高数据完整性
# #### num_epoch（全数据集迭代次数）
# '''
# tf.app.flags.DEFINE_integer('batch_size', default=128, help='Batch size for training')
# tf.app.flags.DEFINE_integer('num_epoch', default=10, help='Number of training iterations')
# tf.app.flags.DEFINE_integer('batch_per_log', default=10, help='Print the log at what number of batches?')
#
# '''
# ### 模型标记
# 隐藏层中神经元为128个
# '''
# tf.app.flags.DEFINE_integer('hidden_size', default=128, help='Number of neurons for RNN hidden layer')
#
# '''
# tf.app.flags可以认为是对模块argparse的简单封装，它实现了python-gflags的一个功能子集。
# 请注意，此模块目前封装在一起，主要用于编写演示应用程序，并且在技术上不是公共API的一部分，将来有发生更改的可能性。
# 建议您使用argparse或您喜欢的任何其他代码库实现自己的标志解析。
# 这里类似args可以调用之前存储的所有元素信息
# - args.seed
# - args = tf.app.flags.FLAGS
# - args.learning_rate
# - args.batch_size
# - args.hidden_size
# - args.seed
# - args.batch_per_log
# - args.num_epoch
# 在FLAG结构中存储所有元素
# '''
# args=tf.flags.FLAGS
#
# # Reset the graph set the random numbers to be the same using "seed"
# tf.reset_default_graph()
# tf.set_random_seed(args.seed)
# np.random.seed(args.seed)
#
# # Divide 28x28 images to rows of data to feed to RNN as sequantial information
# step_size = 28
# input_size = 28
# output_size = 10
#
# # Input tensors
# with tf.name_scope('inputs'):
#     X = tf.placeholder(tf.float32, [None, step_size, input_size],name='X_input')
#     y = tf.placeholder(tf.int32, [None],name='y_input')
#
# '''
# ### 实现RNN
# rnn_cell:是Tensorflow中实现RNN的基本单元。每个rnn_cell都有一个call方法。
# 每调用一次RNNCell的call方法，就相当于在时间上“推进了一步”
# 但是对于单个RNNCell调用call函数进行运算时，只是在序列时间上前进了一步。比如使用x1、h0得到h1，通过x2、h1得到h2等。
# eg：如果序列长度为10，就要调用10次call函数，比较麻烦。
# TensorFlow提供了一个tf.nn.dynamic_rnn函数，使用该函数就相当于调用了n次call函数。即通过{h0,x1, x2, …., xn}直接得{h1,h2…,hn}。
# 输入数据的格式为(batch_size, time_steps, input_size)
# - time_steps表示序列本身的长度，如在Char RNN中，长度为10的句子对应的time_steps就等于10。
# - input_size就表示输入数据单个序列单个时间维度上固有的长度。
# - X为上面初始化的占位符
# - 已经定义好了一个RNNCell，调用该RNNCell的call函数time_steps次
# 最后得到的结果中有两个output和state
# - output就是time_steps步里所有的输出。形状为(batch_size, time_steps, cell.output_size)。
# - state是最后一步的隐状态，形状为(batch_size, cell.state_size)
# '''
# cell = tf.nn.rnn_cell.BasicRNNCell(num_units=args.hidden_size)
# # print(cell.state_size) # =>128
# output, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
#
# '''
# ### 前向传播和损失计算
# tf.layers.dense: 构建稠密全连接层。输入参数是神经元数目和激活函数。
# tf.nn.sparse_softmax_cross_entropy_with_logits: 在内部算得softmax函数值后，继续计算交叉熵
# '''
# logits = tf.layers.dense(state, output_size)
#
# with tf.name_scope('loss'):
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#     loss = tf.reduce_mean(cross_entropy)
#     tf.summary.scalar("loss", loss)
#
# '''
# ### 优化和预测
# tf.nn.in_top_k: 用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量
# tf.nn.in_top_k(prediction, target, K):prediction就是表示预测的结果，大小是预测样本的数量乘以输出的维度，类型是tf.float32等。
# target就是实际样本类别的标签，大小就是样本数量的个数。K表示每个样本的预测结果的前K个最大的数里面是否含有target中的值。
# 一般取1，即是说预测中元素内前一个元素是否相同，如果K值为2，就是前2个。
# '''
# with tf.name_scope('train'):
#     optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)
#
# with tf.name_scope('accuracy'):
#     prediction = tf.nn.in_top_k(logits, y, 1)
#     accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
#     tf.summary.scalar("accuracy", accuracy)
# summ = tf.summary.merge_all()
#
#
# '''
# ### 数据处理
# '''
# # from tensorflow.examples.tutorials.mnist import input_data
# # mnist = input_data.read_data_sets(".\MNIST_data/")
# # X_test = mnist.test.images # X_test shape: [num_test, 28*28]
# # X_test = X_test.reshape([-1, step_size, input_size])
# # y_test = mnist.test.labels
#
# # 初始化变量
# init = tf.global_variables_initializer()
#
# # 生成追踪list。
# loss_train_list = []
# acc_train_list = []
#
# ### 训练模型
# sess=tf.Session()
# sess.run(init)
# writer = tf.summary.FileWriter('./logs/rnn',sess.graph)