'''
**说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。**
进入第二部分深度学习  

第九章启动并运行Tensorflow
Tensorflow是一款用于数值计算的强大的开源软件库，特别适用于大规模机器学习的微调。   
它的基本原理很简单：首先在 Python 中定义要执行的计算图（例如图 9-1），然后 TensorFlow 使用该图并使用优化的 C++ 代码高效运行该图。
- 提供了一个非常简单的 Python API，名为 TF.Learn2（tensorflow.con trib.learn），与 Scikit-Learn 兼容。正如你将会看到的，你可以用几行代码来训练不同类型的神经网络。之前是一个名为 Scikit Flow（或 Skow）的独立项目。
- 提供了另一个简单的称为 TF-slim（tensorflow.contrib.slim）的 API 来简化构建，训练和求出神经网络。
- 其他几个高级 API 已经在 TensorFlow 之上独立构建，如 Keras 或 Pretty Tensor。
- 它的主要 Python API 提供了更多的灵活性（以更高复杂度为代价）来创建各种计算，包括任何你能想到的神经网络结构。
- 它提供了几个高级优化节点来搜索最小化损失函数的参数。由于 TensorFlow 自动处理计算您定义的函数的梯度，因此这些非常易于使用。这称为自动分解（或autodi）。
- 它还附带一个名为 TensorBoard 的强大可视化工具，可让您浏览计算图表，查看学习曲线等。
这一章主要是介绍TensorFlow基础知识，从安装到创建，运行，保存和可视化简单的计算图
我发现之前看的tf相关教程还是太零散了，当时应该直接跳到这里……

### 1.安装
我的anaconda依旧不行，看来回去的时候要重做系统了。  
常规的话使用pip install tensorflow语句即可完成安装。在GPU版本中需要安装英伟达对应的coda和相关包。这部分我打算放在colab上运行。

### 2.创建第一个图谱
#### 生成一个计算图谱，tf会负责在处理器上运行并保留变量值
~~~python
import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2
~~~
#### 运行会话
~~~python
#教程中做了两次优化
#1.使用sess.run()，但是有些麻烦
#2.使用with打开会话后，变量初始化并运行：x.initialzier.run()，依旧麻烦
#3.使用全局global_variables_initializer()，就不用手动处理每一个变量
init=tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result=f.eval()
    print(result)
~~~
TensorFlow程序通常分为两部分：第一部分构建计算图谱（这称为构造阶段），第二部分运行它（这是执行阶段）。 建设阶段通常构建一个表示 ML 模型的计算图谱,然后对其进行训练,计算。 执行阶段通常运行循环，重复地求出训练步骤（例如，每个小批次），逐渐改进模型参数

### 3.管理图谱
在程序中创建的每一个节点都会添加至默认图形中。

### 4.Tensorflow实现线性回归
这里使用的是第二章的房地产数据，完成求theta数值的公式
~~~python
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

housing=fetch_california_housing()
# print(housing)
m,n=housing.data.shape
housing_data_plus_bias=np.c_[np.ones((m,1)),housing.data]

X=tf.constant(housing_data_plus_bias,dtype=tf.float32,name='X')
y=tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name='y')
#对X进行转置
XT=tf.transpose(X)
#matmul矩阵相乘
theta=tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)

with tf.Session() as sess:
    #eval用于获取返回值
    theta_value=theta.eval()
print(theta_value)

# [[-3.7185181e+01]
#  [ 4.3633747e-01]
#  [ 9.3952334e-03]
#  [-1.0711310e-01]
#  [ 6.4479220e-01]
#  [-4.0338000e-06]
#  [-3.7813708e-03]
#  [-4.2348403e-01]
#  [-4.3721911e-01]]
~~~
教程中与sklearn的系那行回归模型结果进行了对比
~~~python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))
print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])
~~~



**说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。**
进入第二部分深度学习  

第九章启动并运行Tensorflow
Tensorflow是一款用于数值计算的强大的开源软件库，特别适用于大规模机器学习的微调。   
它的基本原理很简单：首先在 Python 中定义要执行的计算图（例如图 9-1），然后 TensorFlow 使用该图并使用优化的 C++ 代码高效运行该图。
- 提供了一个非常简单的 Python API，名为 TF.Learn2（tensorflow.con trib.learn），与 Scikit-Learn 兼容。正如你将会看到的，你可以用几行代码来训练不同类型的神经网络。之前是一个名为 Scikit Flow（或 Skow）的独立项目。
- 提供了另一个简单的称为 TF-slim（tensorflow.contrib.slim）的 API 来简化构建，训练和求出神经网络。
- 其他几个高级 API 已经在 TensorFlow 之上独立构建，如 Keras 或 Pretty Tensor。
- 它的主要 Python API 提供了更多的灵活性（以更高复杂度为代价）来创建各种计算，包括任何你能想到的神经网络结构。
- 它提供了几个高级优化节点来搜索最小化损失函数的参数。由于 TensorFlow 自动处理计算您定义的函数的梯度，因此这些非常易于使用。这称为自动分解（或autodi）。
- 它还附带一个名为 TensorBoard 的强大可视化工具，可让您浏览计算图表，查看学习曲线等。
这一章主要是介绍TensorFlow基础知识，从安装到创建，运行，保存和可视化简单的计算图
我发现之前看的tf相关教程还是太零散了，当时应该直接跳到这里……

### 5.实现梯度下降
使用批量梯度下降。 
- 使用 TensorFlow 的自动扩展功能来使 TensorFlow 自动计算梯度
- 使用几个 TensorFlow 的优化器
当使用梯度下降时，请记住，首先要对输入特征向量进行归一化，否则训练可能要慢得多。 
**注意**在代码中应该引入一个归一化的方法，“from sklearn.preprocessing import scale”，教程中似乎忘了。  

~~~python 
from sklearn.preprocessing import scale

housing = fetch_california_housing()
m, n = housing.data.shape
#使用np.c_按照colunm来组合array，其实就是生成m个元素为1的list，然后和housing.data数据合并，在每个housing.data数组前加1
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
#对数据进行归一化处理
scaled_housing_data_plus_bias = scale(housing_data_plus_bias)
#迭代次数和学习频率
n_epochs = 1000
learning_rate = 0.01
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

#random_uniform()函数在图形中创建一个节点，它将生成包含随机值的张量，给定其形状和值作用域，就像 NumPy 的rand()函数一样。
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2 / m * tf.matmul(tf.transpose(X), error)
#assign()函数创建一个为变量分配新值的节点。 在这种情况下，它实现了批次梯度下降步骤 
training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()

Epoch 0 MSE = 8.333049
Epoch 100 MSE = 4.96547
Epoch 200 MSE = 4.890386
Epoch 300 MSE = 4.8652663
Epoch 400 MSE = 4.847968
Epoch 500 MSE = 4.8355227
Epoch 600 MSE = 4.8265452
Epoch 700 MSE = 4.8200703
Epoch 800 MSE = 4.8153954
Epoch 900 MSE = 4.8120193
~~~
上面是手动撸的一个梯度下降，需要从代价函数（MSE）中利用数学公式推导梯度。  
在线性回归的情况下，这是相当容易的，但是如果用深层神经网络来做这个事情，会很繁琐。 您可以使用符号求导来为您自动找到偏导数的方程式，但结果代码不一定非常有效。

Tensorflow提供了自动梯度计算的方法（反向传播），同时使用优化器，修改后的代码如下，结果一致。
~~~python
from sklearn.preprocessing import scale

housing = fetch_california_housing()
m, n = housing.data.shape
#使用np.c_按照colunm来组合array，其实就是生成m个元素为1的list，然后和housing.data数据合并，在每个housing.data数组前加1
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
#对数据进行归一化处理
scaled_housing_data_plus_bias = scale(housing_data_plus_bias)
#迭代次数和学习频率
n_epochs = 1000
learning_rate = 0.01
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

#random_uniform()函数在图形中创建一个节点，它将生成包含随机值的张量，给定其形状和值作用域，就像 NumPy 的rand()函数一样。
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
#原始手动下降
# gradients = 2 / m * tf.matmul(tf.transpose(X), error)
#assign()函数创建一个为变量分配新值的节点。 在这种情况下，它实现了批次梯度下降步骤
# training_op = tf.assign(theta, theta - learning_rate * gradients)
#使用优化器
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(mse)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
~~~

### 6.将数据提供给训练算法
在这里其实就是实现小批量梯度下降mini batch。  
在每次迭代时使用下一个下批量替换X和y，使用占位符placeholder节点。这部分在之前的教程中提到过，略过，看代码即可。  
代码需要注意的是对之前代码的修改，如何实现小批量梯度下降的。
~~~python
# 使用StandardScaler代替scale
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m, n = housing.data.shape
print("数据集:{}行,{}列".format(m, n))
# 使用np.c_按照colunm来组合array，其实就是生成m个元素为1的list，然后和housing.data数据合并，在每个housing.data数组前加1
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# 对数据进行归一化处理，这部分有了调整
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
# 迭代次数和学习频率
n_epochs = 1000
learning_rate = 0.01

# 原始的X，y初始化
# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# 使用占位符
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# random_uniform()函数在图形中创建一个节点，它将生成包含随机值的张量，给定其形状和值作用域，就像 NumPy 的rand()函数一样。
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# 原始手动下降
# gradients = 2 / m * tf.matmul(tf.transpose(X), error)
# assign()函数创建一个为变量分配新值的节点。 在这种情况下，它实现了批次梯度下降步骤
# training_op = tf.assign(theta, theta - learning_rate * gradients)
# 使用优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))


def fetch_batch(epoch, batch_index, batch_size):
    know = np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    print("我是know:", know)
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices]  # not shown
    y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
    return X_batch, y_batch


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()
print(best_theta)

[[ 2.0700505 ]
 [ 0.84353966]
 [ 0.12656353]
 [-0.25964832]
 [ 0.3324695 ]
 [ 0.00659902]
 [-0.01405354]
 [-0.8211542 ]
 [-0.7868624 ]]
~~~
'''

# import tensorflow as tf

# x = tf.Variable(3, name='x')
# y = tf.Variable(4, name='y')
# f = x * x * y + y + 2

# 教程中做了两次优化
# 1.使用sess.run()，但是有些麻烦
# 2.使用with打开会话后，变量初始化并运行：x.initialzier.run()，依旧麻烦
# 3.使用全局global_variables_initializer()，就不用手动处理每一个变量

# init=tf.global_variables_initializer()
# with tf.Session() as sess:
#     init.run()
#     result=f.eval()
#     print(result)

# ============================================================

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

# housing=fetch_california_housing()
# # print(housing)
# m,n=housing.data.shape
# housing_data_plus_bias=np.c_[np.ones((m,1)),housing.data]
#
# X=tf.constant(housing_data_plus_bias,dtype=tf.float32,name='X')
# y=tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name='y')
# #对X进行转置
# XT=tf.transpose(X)
# #matmul矩阵相乘
# theta=tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)
#
# with tf.Session() as sess:
#     #eval用于获取返回值
#     theta_value=theta.eval()
# print(theta_value)
# from sklearn.preprocessing import scale

# 使用StandardScaler代替scale
from sklearn.preprocessing import StandardScaler

# housing = fetch_california_housing()
# m, n = housing.data.shape
# print("数据集:{}行,{}列".format(m, n))
# # 使用np.c_按照colunm来组合array，其实就是生成m个元素为1的list，然后和housing.data数据合并，在每个housing.data数组前加1
# housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# # 对数据进行归一化处理，这部分有了调整
# scaler = StandardScaler()
# scaled_housing_data = scaler.fit_transform(housing.data)
# scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
# # 迭代次数和学习频率
# n_epochs = 1000
# learning_rate = 0.01
#
# # 原始的X，y初始化
# # X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# # y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# # 使用占位符
# X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
# y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# # random_uniform()函数在图形中创建一个节点，它将生成包含随机值的张量，给定其形状和值作用域，就像 NumPy 的rand()函数一样。
# theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
# y_pred = tf.matmul(X, theta, name="predictions")
# error = y_pred - y
# mse = tf.reduce_mean(tf.square(error), name="mse")
# # 原始手动下降
# # gradients = 2 / m * tf.matmul(tf.transpose(X), error)
# # assign()函数创建一个为变量分配新值的节点。 在这种情况下，它实现了批次梯度下降步骤
# # training_op = tf.assign(theta, theta - learning_rate * gradients)
# # 使用优化器
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(mse)
# init = tf.global_variables_initializer()
#
# n_epochs = 10
# batch_size = 100
# n_batches = int(np.ceil(m / batch_size))
#
#
# def fetch_batch(epoch, batch_index, batch_size):
#     know = np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
#     print("我是know:", know)
#     indices = np.random.randint(m, size=batch_size)  # not shown
#     X_batch = scaled_housing_data_plus_bias[indices]  # not shown
#     y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
#     return X_batch, y_batch
#
#
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(n_epochs):
#         for batch_index in range(n_batches):
#             X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#     best_theta = theta.eval()
# print(best_theta)

'''
**说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。**
进入第二部分深度学习  

第九章启动并运行Tensorflow
Tensorflow是一款用于数值计算的强大的开源软件库，特别适用于大规模机器学习的微调。   
它的基本原理很简单：首先在 Python 中定义要执行的计算图（例如图 9-1），然后 TensorFlow 使用该图并使用优化的 C++ 代码高效运行该图。
- 提供了一个非常简单的 Python API，名为 TF.Learn2（tensorflow.con trib.learn），与 Scikit-Learn 兼容。正如你将会看到的，你可以用几行代码来训练不同类型的神经网络。之前是一个名为 Scikit Flow（或 Skow）的独立项目。
- 提供了另一个简单的称为 TF-slim（tensorflow.contrib.slim）的 API 来简化构建，训练和求出神经网络。
- 其他几个高级 API 已经在 TensorFlow 之上独立构建，如 Keras 或 Pretty Tensor。
- 它的主要 Python API 提供了更多的灵活性（以更高复杂度为代价）来创建各种计算，包括任何你能想到的神经网络结构。
- 它提供了几个高级优化节点来搜索最小化损失函数的参数。由于 TensorFlow 自动处理计算您定义的函数的梯度，因此这些非常易于使用。这称为自动分解（或autodi）。
- 它还附带一个名为 TensorBoard 的强大可视化工具，可让您浏览计算图表，查看学习曲线等。
这一章主要是介绍TensorFlow基础知识，从安装到创建，运行，保存和可视化简单的计算图
我发现之前看的tf相关教程还是太零散了，当时应该直接跳到这里……

### 7.模型保存和读取
完成模型训练后应该把参数保存到磁盘，所以你可以随时随地回到它，在另一个程序中使用它，与其他模型比较。   
此外，您可能希望在训练期间定期保存检查点，以便如果您的计算机在训练过程中崩溃，您可以从上次检查点继续进行，而不是从头开始。

TensorFlow 可以轻松保存和恢复模型。 只需在构造阶段结束（创建所有变量节点之后）创建一个保存节点; 那么在执行阶段，只要你想保存模型，只要调用它的save()方法“save_path = saver.save(sess, "需要保存的地址")”。  
恢复模型，在构建阶段结束时创建一个保存器，就像之前一样，但是在执行阶段的开始，而不是使用init节点初始化变量，可以调用restore()方法 的保存器对象。
~~~python
housing = fetch_california_housing()
m, n = housing.data.shape
print("数据集:{}行,{}列".format(m, n))
# 使用np.c_按照colunm来组合array，其实就是生成m个元素为1的list，然后和housing.data数据合并，在每个housing.data数组前加1
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# 对数据进行归一化处理，这部分有了调整
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
# 迭代次数和学习频率
n_epochs = 1000
learning_rate = 0.01

# 原始的X，y初始化
# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# 使用占位符
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# random_uniform()函数在图形中创建一个节点，它将生成包含随机值的张量，给定其形状和值作用域，就像 NumPy 的rand()函数一样。
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# 原始手动下降
# gradients = 2 / m * tf.matmul(tf.transpose(X), error)
# assign()函数创建一个为变量分配新值的节点。 在这种情况下，它实现了批次梯度下降步骤
# training_op = tf.assign(theta, theta - learning_rate * gradients)
# 使用优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
# 这里初始化一个saver
saver = tf.train.Saver()

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))


def fetch_batch(epoch, batch_index, batch_size):
    know = np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    # print("我是know:", know)
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices]  # not shown
    y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
    return X_batch, y_batch


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()
    #这个路径要注意，前面加‘.’表明是保存在项目文件下的tmp文件夹内
    save_path = saver.save(sess, "./tmp/tensorflow01.ckpt")
print(best_theta)
~~~

### 8.使用TensorBoard展现图形和训练曲线
**这是一个新的内容，请注意**
到目前位置，仍然依靠print()函数可视化训练过程中的进度。  
有一个更好的方法：进入 TensorBoard。提供一些训练统计信息，它将在您的网络浏览器中显示这些统计信息的良好交互式可视化（例如学习曲线）。 您还可以提供图形的定义，它将为您提供一个很好的界面来浏览它。 这对于识别图中的错误，找到瓶颈等是非常有用的。
- 第一步是调整程序，以便将图形定义和一些训练统计信息（例如，training_error（MSE））写入 TensorBoard 将读取的日志目录。 您每次运行程序时都需要使用不同的日志目录，否则 TensorBoard 将会合并来自不同运行的统计信息，这将会混乱可视化。 最简单的解决方案是在日志目录名称中包含时间戳。 在程序开始处添加以下代码
~~~python
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
~~~
接下来，在构建阶段结束时添加以下代码
~~~python
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
~~~
整体代码
~~~python
from datetime import datetime

housing = fetch_california_housing()
m, n = housing.data.shape
print("数据集:{}行,{}列".format(m, n))
# 使用np.c_按照colunm来组合array，其实就是生成m个元素为1的list，然后和housing.data数据合并，在每个housing.data数组前加1
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# 对数据进行归一化处理，这部分有了调整
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# 迭代次数和学习频率
n_epochs = 1000
learning_rate = 0.01

# 原始的X，y初始化
# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# 使用占位符
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# random_uniform()函数在图形中创建一个节点，它将生成包含随机值的张量，给定其形状和值作用域，就像 NumPy 的rand()函数一样。
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# 原始手动下降
# gradients = 2 / m * tf.matmul(tf.transpose(X), error)
# assign()函数创建一个为变量分配新值的节点。 在这种情况下，它实现了批次梯度下降步骤
# training_op = tf.assign(theta, theta - learning_rate * gradients)
# 使用优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
# 这里初始化一个saver
saver = tf.train.Saver()
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))


def fetch_batch(epoch, batch_index, batch_size):
    know = np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    # print("我是know:", know)
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices]  # not shown
    y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
    return X_batch, y_batch


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):  # not shown
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
file_writer.close()
print(best_theta)
~~~
'''
from datetime import datetime

housing = fetch_california_housing()
m, n = housing.data.shape
print("数据集:{}行,{}列".format(m, n))
# 使用np.c_按照colunm来组合array，其实就是生成m个元素为1的list，然后和housing.data数据合并，在每个housing.data数组前加1
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# 对数据进行归一化处理，这部分有了调整
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# 迭代次数和学习频率
n_epochs = 1000
learning_rate = 0.01

# 原始的X，y初始化
# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# 使用占位符
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# random_uniform()函数在图形中创建一个节点，它将生成包含随机值的张量，给定其形状和值作用域，就像 NumPy 的rand()函数一样。
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# 原始手动下降
# gradients = 2 / m * tf.matmul(tf.transpose(X), error)
# assign()函数创建一个为变量分配新值的节点。 在这种情况下，它实现了批次梯度下降步骤
# training_op = tf.assign(theta, theta - learning_rate * gradients)
# 使用优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
# 这里初始化一个saver
saver = tf.train.Saver()
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))


def fetch_batch(epoch, batch_index, batch_size):
    know = np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    # print("我是know:", know)
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices]  # not shown
    y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
    return X_batch, y_batch


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):  # not shown
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
file_writer.close()
print(best_theta)