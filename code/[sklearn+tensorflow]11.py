'''
**说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。**
进入第二部分深度学习  

第十一章训练深层神经网络
在第十章以及之前tf练习中，训练的深度神经网络都只是简单的demo，如果增大数据量或是面对更多的特征，遇到的问题就会棘手起来。  
- 梯度消失（梯度爆炸），这会影响深度神经网络，并使较低层难以训练
- 训练效率
- 包含数百万参数的模型将会有严重的过拟合训练集的风险
本章中，教程从解释梯度消失问题开始，并探讨解决这个问题的一些最流行的解决方案。   
接下来讨论各种优化器，与普通梯度下降相比，它们可以加速大型模型的训练。   
介绍大型神经网络正则化技术。

~~~python
# 常用的引入和初始化设定
import numpy as np
import os
import tensorflow as tf

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "deep"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
~~~

### 1.梯度消失/爆炸问题
#### 什么是梯度消失/爆炸
传播误差的梯度，一旦该算法已经计算了网络中每个参数的损失函数的梯度，它就使用这些梯度来用梯度下降步骤来更新每个参数。  
- 梯度消失：梯度往往变得越来越小，随着算法进展到较低层。 结果，梯度下降更新使得低层连接权重实际上保持不变，并且训练永远不会收敛到良好的解决方案。 这被称为梯度消失问题。   
- 梯度爆炸：梯度可能变得越来越大，许多层得到了非常大的权重更新，算法发散。这是梯度爆炸的问题，在循环神经网络中最为常见。   
深度神经网络受梯度不稳定之苦; 不同的层次可能以非常不同的速度学习。
目前流行的激活函数sigmoid等，可以看到当输入值的范数变大时，函数饱和在 0 或 1，导数非常接近 0。  
因此，当反向传播开始时，它几乎没有梯度通过网络传播回来，而且由于反向传播通过顶层向下传递，所以存在的小梯度不断地被稀释，因此较低层确实没有任何东西可用。  
这样就导致了每一层之间的信息无法很好传递。
~~~python
def logit(z):
    return 1/(1+np.exp(-z))

z=np.linspace(-5,5,200)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [1, 1], 'k--')
plt.plot([0, 0], [-0.2, 1.2], 'k-')
plt.plot([-5, 5], [-3/4, 7/4], 'g--')
plt.plot(z, logit(z), "b-", linewidth=2)
props = dict(facecolor='black', shrink=0.1)
plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha="center")
plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center")
plt.grid(True)
plt.title("Sigmoid activation function", fontsize=14)
plt.axis([-5, 5, -0.2, 1.2])
save_fig("sigmoid_saturation_plot")
plt.show()
~~~
#### 解决办法
Glorot 和 Bengio 提出需要信号在两个方向上正确地流动：在进行预测时是正向的，在反向传播梯度时是反向的。 我们不希望信号消失，也不希望它爆炸并饱和。  
为了使信号正确流动，作者认为需要**每层输出的方差等于其输入的方差**。实际上不可能保证两者都是一样的，除非这个层具有相同数量的输入和输出连接，他们提出了折衷办法：随机初始化连接权重必须如**公式**所描述的那样。其中n_inputs和n_outputs是权重正在被初始化的层（也称为扇入和扇出）的输入和输出连接的数量。 这种初始化策略通常被称为Xavier初始化。
公式如下
在满足之前要求的**每层输出的方差等于其输入的方差**的情况下时n_inputs = n_outputs，公式可以简化为

现在发展出了不同的激活函数，logistic、He、Relu等。Tensorflow修改了部分方法名称。  
- activation_fn变成激活（类似地，_fn后缀从诸如normalizer_fn之类的其他参数中移除）
- weights_initializer变成kernel_initializer
- 默认激活现在是None，而不是tf.nn.relu
- 不支持正则化的参数

### 2.非饱和激活函数
这里使用的使Relu作为激活函数，优点是对于正值不会饱和，而且计算速率更快。 
但是存在的问题是**Relu死区**。
#### Relu死区
在训练过程中，一些神经元死亡，意味着它们停止输出 0 以外的任何东西。  
在某些情况下，你可能会发现你网络的一半神经元已经死亡，特别是如果你使用大学习率。  
在训练期间，如果神经元的权重得到更新，使得神经元输入的加权和为负，则它将开始输出 0 。当这种情况发生时，由于当输入为负时，ReLU函数的梯度为0，神经元不可能恢复生机。

#### 使用leakyRelu
为了解决Relu死区问题，可以使用Relu函数的变体leaky RelU。  
这个函数定义为LeakyReLUα(z)= max(αz，z)  
超参数α定义了函数“leaks”的程度：它是z < 0时函数的斜率，通常设置为 0.01。这个小斜坡确保 leaky ReLU 永不死亡；他们可能会长期昏迷，但他们有机会最终醒来。  
也有文献指出当alpha=0.2时，即是设定一个超大的超参数的性能更优（在大型的数据集中表现良好，但是在小数据集中会出现过拟合的现象）

~~~python
def logit(z):
    return 1/(1+np.exp(-z))

z=np.linspace(-5,5,200)
#重置tf的图
reset_graph()
n_inputs=28*28
n_hidden1=300

X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")

he_init=tf.variance_scaling_initializer()
hidden1=tf.layers.dense(X,n_hidden1,activation=tf.nn.relu,kernel_initializer=he_init,name="hidden1")
#超参数α定义了函数“leaks”的程度：它是z < 0时函数的斜率，通常设置为 0.01
def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha*z, z)

plt.plot(z,leaky_relu(z,0.05),"b-",linewidth=2)
plt.plot([-5,5],[0,0],'k-')
plt.plot([0,0],[-0.5,4.2],'k-')
plt.grid(True)
props=dict(facecolor='black', shrink=0.1)
plt.annotate('Leak', xytext=(-3.5, 0.5), xy=(-5, -0.2), arrowprops=props, fontsize=14, ha="center")
plt.title("Leaky ReLU activation function", fontsize=14)
plt.axis([-5, 5, -0.5, 4.2])

save_fig("leaky_relu_plot")
plt.show()
~~~
'''
from functools import partial

import numpy as np
import os
import tensorflow as tf


# 为了使程序输出更加稳定
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "deep"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def logit(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-5, 5, 200)


# plt.plot([-5, 5], [0, 0], 'k-')
# plt.plot([-5, 5], [1, 1], 'k--')
# plt.plot([0, 0], [-0.2, 1.2], 'k-')
# plt.plot([-5, 5], [-3/4, 7/4], 'g--')
# plt.plot(z, logit(z), "b-", linewidth=2)
# props = dict(facecolor='black', shrink=0.1)
# plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha="center")
# plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
# plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center")
# plt.grid(True)
# plt.title("Sigmoid activation function", fontsize=14)
# plt.axis([-5, 5, -0.2, 1.2])
# save_fig("sigmoid_saturation_plot")
# plt.show()

# =======================================================================================================================

# reset_graph()
# n_inputs=28*28
# n_hidden1=300
# X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
#
# he_init=tf.variance_scaling_initializer()
# hidden1=tf.layers.dense(X,n_hidden1,activation=tf.nn.relu,kernel_initializer=he_init,name="hidden1")
def leaky_relu(z, alpha=0.01):
    # 注意，这里应该是tf而不是教程中的np，Tensorflow通常接受numpy数组（它们的值是静态已知的，因此可以转换为常量），但反之则不然（只有在运行会话时才知道张量值，除了急切的评估）
    return tf.maximum(alpha * z, z)


#
# plt.plot(z,leaky_relu(z,0.05),"b-",linewidth=2)
# plt.plot([-5,5],[0,0],'k-')
# plt.plot([0,0],[-0.5,4.2],'k-')
# plt.grid(True)
# props=dict(facecolor='black', shrink=0.1)
# plt.annotate('Leak', xytext=(-3.5, 0.5), xy=(-5, -0.2), arrowprops=props, fontsize=14, ha="center")
# plt.title("Leaky ReLU activation function", fontsize=14)
# plt.axis([-5, 5, -0.5, 4.2])
#
# save_fig("leaky_relu_plot")
# plt.show()

# =======================================================================================================================

'''
**说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。**
进入第二部分深度学习  

第十一章训练深层神经网络
在第十章以及之前tf练习中，训练的深度神经网络都只是简单的demo，如果增大数据量或是面对更多的特征，遇到的问题就会棘手起来。  
- 梯度消失（梯度爆炸），这会影响深度神经网络，并使较低层难以训练
- 训练效率
- 包含数百万参数的模型将会有严重的过拟合训练集的风险
本章中，教程从解释梯度消失问题开始，并探讨解决这个问题的一些最流行的解决方案。   
接下来讨论各种优化器，与普通梯度下降相比，它们可以加速大型模型的训练。   
介绍大型神经网络正则化技术。

### 3.使用LeakyRelu作为激活函数，训练神经网络
在这里使用Leaky Relu训练一个神经网络，处理MNIST数据集。
~~~python
def logit(z):
    return 1 / (1 + np.exp(-z))
z = np.linspace(-5, 5, 200)

def leaky_relu(z, alpha=0.01):
    #注意，这里应该是tf而不是教程中的np，Tensorflow通常接受numpy数组（它们的值是静态已知的，因此可以转换为常量），但反之则不然（只有在运行会话时才知道张量值，除了急切的评估）
    return tf.maximum(alpha * z, z)

reset_graph()
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=leaky_relu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
X_train=X_train.astype(np.float32).reshape(-1,28*28)/255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

n_epochs=40
batch_size=50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train,y_train,batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch%5==0:
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

    save_path = saver.save(sess, "./tf_logs/run-20190126025648/tensorflowmodel01.ckpt")

0 Batch accuracy: 0.86 Validation accuracy: 0.9044
5 Batch accuracy: 0.94 Validation accuracy: 0.9494
10 Batch accuracy: 0.92 Validation accuracy: 0.9656
15 Batch accuracy: 0.94 Validation accuracy: 0.971
20 Batch accuracy: 1.0 Validation accuracy: 0.9762
25 Batch accuracy: 1.0 Validation accuracy: 0.9772
30 Batch accuracy: 0.98 Validation accuracy: 0.9782
35 Batch accuracy: 1.0 Validation accuracy: 0.9788
~~~

#### 使用Elu
Djork-Arné Clevert 等人在 2015 年的一篇论文中提出了一种称为指数线性单元（exponential linear unit，ELU）的新的激活函数。  
Elu表现优于所有的 ReLU 变体：训练时间减少，神经网络在测试集上表现的更好。  
公式
公式看起来很像 ReLU 函数，但有一些区别：
- 在z < 0时取负值，这使得该单元的平均输出接近于 0。这有助于减轻梯度消失问题，如前所述。 超参数α定义为当z是一个大的负数时，ELU 函数接近的值。它通常设置为 1，但是如果你愿意，你可以像调整其他超参数一样调整它。
- 对z < 0有一个非零的梯度，避免了神经元死亡的问题。
- 函数在任何地方都是平滑的，包括z = 0左右，这有助于加速梯度下降，因为它不会弹回z = 0的左侧和右侧。

ELU 激活函数的主要**缺点**是计算速度慢于 ReLU 及其变体（由于使用指数函数）。  
那么你应该使用哪个激活函数来处理深层神经网络的隐藏层？  
一般 ELU > leaky ReLU（及其变体）> ReLU > tanh > sigmoid。  
- 如果关心运行时性能，可以使用leaky ReLU。 
- 如果不想调整另一个超参数，可以使用默认的α值（leaky ReLU 为 0.01，ELU 为 1）。 
- 如果有充足的时间和计算能力，可以使用交叉验证来评估其他激活函数，特别是如果您的神经网络过拟合，则为RReLU; 
- 如果拥有庞大的训练数据集，可以使用 PReLU。

TensorFlow 提供了一个可以用来建立神经网络的elu()函数。 调用fully_connected()函数时，只需设置activation_fn参数即可（activation=tf.nn.elu）。  
在训练期间，由使用SELU激活函数的一堆密集层组成的神经网络将自我归一化：每层的输出将倾向于在训练期间保持相同的均值和方差，这解决了消失/爆炸的梯度问题。因此，这种激活功能对于这样的神经网络来说非常显着地优于其他激活功能，因此应该尝试一下。

~~~python
def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)

reset_graph()
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=selu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=selu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 40
batch_size = 50

(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
X_train=X_train.astype(np.float32).reshape(-1,28*28)/255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

means = X_train.mean(axis=0, keepdims=True)
stds = X_train.std(axis=0, keepdims=True) + 1e-10
X_val_scaled = (X_valid - means) / stds

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch_scaled = (X_batch - means) / stds
            sess.run(training_op, feed_dict={X: X_batch_scaled, y: y_batch})
        if epoch % 5 == 0:
            acc_batch = accuracy.eval(feed_dict={X: X_batch_scaled, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_val_scaled, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

    save_path = saver.save(sess, "./tf_logs/run-20190126025649/tensorflowmodel02selu.ckpt")

0 Batch accuracy: 0.88 Validation accuracy: 0.9232
5 Batch accuracy: 0.98 Validation accuracy: 0.9574
10 Batch accuracy: 1.0 Validation accuracy: 0.9662
15 Batch accuracy: 0.96 Validation accuracy: 0.9684
20 Batch accuracy: 1.0 Validation accuracy: 0.9692
25 Batch accuracy: 1.0 Validation accuracy: 0.969
30 Batch accuracy: 1.0 Validation accuracy: 0.9694
35 Batch accuracy: 1.0 Validation accuracy: 0.9702
~~~

'''


# reset_graph()
# n_inputs = 28 * 28
# n_hidden1 = 300
# n_hidden2 = 100
# n_outputs = 10
#
# X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
# y = tf.placeholder(tf.int32, shape=(None), name="y")
#
# with tf.name_scope("dnn"):
#     hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu, name="hidden1")
#     hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=leaky_relu, name="hidden2")
#     logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
#
# with tf.name_scope("loss"):
#     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#     loss = tf.reduce_mean(xentropy, name="loss")
#
# learning_rate = 0.01
# with tf.name_scope("train"):
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     training_op = optimizer.minimize(loss)
#
# with tf.name_scope("eval"):
#     correct = tf.nn.in_top_k(logits, y, 1)
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# (X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
# X_train=X_train.astype(np.float32).reshape(-1,28*28)/255.0
# X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)
# X_valid, X_train = X_train[:5000], X_train[5000:]
# y_valid, y_train = y_train[:5000], y_train[5000:]
#
# def shuffle_batch(X, y, batch_size):
#     rnd_idx = np.random.permutation(len(X))
#     n_batches = len(X) // batch_size
#     for batch_idx in np.array_split(rnd_idx, n_batches):
#         X_batch, y_batch = X[batch_idx], y[batch_idx]
#         yield X_batch, y_batch
#
# n_epochs=40
# batch_size=50
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for X_batch, y_batch in shuffle_batch(X_train,y_train,batch_size):
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         if epoch%5==0:
#             acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#             acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
#             print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)
#
#     save_path = saver.save(sess, "./tf_logs/run-20190126025648/tensorflowmodel01.ckpt")

# =======================================================================================================================
def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)


# def selu(z, scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
#     return scale * elu(z, alpha)


reset_graph()
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
# X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
#
# hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, name="hidden1")
#
# np.random.seed(42)
# Z = np.random.normal(size=(500, 100))
# for layer in range(100):
#     W = np.random.normal(size=(100, 100), scale=np.sqrt(1 / 100))
#     Z = selu(np.dot(Z, W))
#     means = np.mean(Z, axis=1)
#     stds = np.std(Z, axis=1)
#     if layer % 10 == 0:
#         print("Layer {}: {:.2f} < mean < {:.2f}, {:.2f} < std deviation < {:.2f}".format(
#             layer, means.min(), means.max(), stds.min(), stds.max()))
# def selu(z,
#          scale=1.0507009873554804934193349852946,
#          alpha=1.6732632423543772848170429916717):
#     return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))
#
# X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
# y = tf.placeholder(tf.int32, shape=(None), name="y")
#
# with tf.name_scope("dnn"):
#     hidden1 = tf.layers.dense(X, n_hidden1, activation=selu, name="hidden1")
#     hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=selu, name="hidden2")
#     logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
#
# with tf.name_scope("loss"):
#     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#     loss = tf.reduce_mean(xentropy, name="loss")
#
# learning_rate = 0.01
#
# with tf.name_scope("train"):
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     training_op = optimizer.minimize(loss)
#
# with tf.name_scope("eval"):
#     correct = tf.nn.in_top_k(logits, y, 1)
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# n_epochs = 40
# batch_size = 50
#
# (X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
# X_train=X_train.astype(np.float32).reshape(-1,28*28)/255.0
# X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)
# X_valid, X_train = X_train[:5000], X_train[5000:]
# y_valid, y_train = y_train[:5000], y_train[5000:]
#
# def shuffle_batch(X, y, batch_size):
#     rnd_idx = np.random.permutation(len(X))
#     n_batches = len(X) // batch_size
#     for batch_idx in np.array_split(rnd_idx, n_batches):
#         X_batch, y_batch = X[batch_idx], y[batch_idx]
#         yield X_batch, y_batch

# means = X_train.mean(axis=0, keepdims=True)
# stds = X_train.std(axis=0, keepdims=True) + 1e-10
# X_val_scaled = (X_valid - means) / stds
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#             X_batch_scaled = (X_batch - means) / stds
#             sess.run(training_op, feed_dict={X: X_batch_scaled, y: y_batch})
#         if epoch % 5 == 0:
#             acc_batch = accuracy.eval(feed_dict={X: X_batch_scaled, y: y_batch})
#             acc_valid = accuracy.eval(feed_dict={X: X_val_scaled, y: y_valid})
#             print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)
#
#     save_path = saver.save(sess, "./tf_logs/run-20190126025649/tensorflowmodel02selu.ckpt")

'''
**说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。**
进入第二部分深度学习  

第十一章训练深层神经网络
在第十章以及之前tf练习中，训练的深度神经网络都只是简单的demo，如果增大数据量或是面对更多的特征，遇到的问题就会棘手起来。  
- 梯度消失（梯度爆炸），这会影响深度神经网络，并使较低层难以训练
- 训练效率
- 包含数百万参数的模型将会有严重的过拟合训练集的风险
本章中，教程从解释梯度消失问题开始，并探讨解决这个问题的一些最流行的解决方案。   
接下来讨论各种优化器，与普通梯度下降相比，它们可以加速大型模型的训练。   
介绍大型神经网络正则化技术。

### 5.批量标准化
尽管使用 He初始化和 ELU（或任何 ReLU 变体）可以显著减少训练开始阶段的梯度消失/爆炸问题，但不保证在训练期间问题不会回来，这样就提出了批量标准化策略，通过对每一层的输入值进行zero-centering和规范化，然后每层使用两个新参数（一个用于尺度变换，另一个用于偏移）对结果进行尺度变换和偏移。  
这个操作可以让模型学习到每层输入值的最佳尺度,均值。为了对输入进行归零和归一化，算法需要估计输入的均值和标准差。通过评估当前小批量输入的均值和标准差（因此命名为“批量标准化”）来实现。  
tensorflow中使用tf.layers.batch_normalization()完成批量标准化。  
以下代码为一个完整的经过标准化的神经网络
~~~python
reset_graph()

import tensorflow as tf


(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
X_train=X_train.astype(np.float32).reshape(-1,28*28)/255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#可以看到每一层以及输出层都做了标准化
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
training = tf.placeholder_with_default(False, shape=(), name="training")

hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
bn1_act = tf.nn.elu(bn1)

hidden2 = tf.layers.dense(X, n_hidden2, name="hidden2")
bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
bn2_act = tf.nn.elu(bn2)

logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
logits = tf.layers.batch_normalization(logits_before_bn, training=training, momentum=0.9)
~~~


现在构建一个处理MNIST数据集的神经网络，每一层使用ELU作为激活函数
~~~python
reset_graph()
batch_norm_momentum=0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y=tf.placeholder(tf.int32, shape=(None), name="y")
training=tf.placeholder_with_default(False, shape=(), name="training")

with tf.name_scope("dnn"):
    he_init=tf.variance_scaling_initializer()
    # 为了防止重复设定参数，这里使用了Python的partial方法
    # 对权重的初始化
    my_batch_norm_layer=partial(
        tf.layers.batch_normalization,
        training=training,
        momentum=batch_norm_momentum
    )
    my_dense_layer=partial(
        tf.layers.dense,
        kernel_initializer=he_init
    )

    hidden1=my_dense_layer(X,n_hidden1,name="hidden1")
    bn1=tf.nn.elu(my_batch_norm_layer(hidden1))
    hidden2=my_dense_layer(bn1,n_hidden2,name="hidden2")
    bn2=tf.nn.elu(my_batch_norm_layer(hidden2))
    logits_before_bn=my_dense_layer(bn2,n_outputs,name="outputs")
    logits=my_batch_norm_layer(logits_before_bn)

with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

init=tf.global_variables_initializer()
saver=tf.train.Saver()

n_epochs=20
batch_size=200

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch , y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run([training_op,extra_update_ops],
                     feed_dict={training:True,X:X_batch,y:y_batch}
                     )
        accuracy_val=accuracy.eval(feed_dict={X:X_valid,y:y_valid})
        print(epoch, accuracy_val)
    save_path=saver.save(sess,"./tf_logs/run-2019012801001/tensorflowmodel01normalization.ckpt")
~~~

'''

# reset_graph()
#
# import tensorflow as tf
#
#
# (X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
# X_train=X_train.astype(np.float32).reshape(-1,28*28)/255.0
# X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)
# X_valid, X_train = X_train[:5000], X_train[5000:]
# y_valid, y_train = y_train[:5000], y_train[5000:]
#
# n_inputs = 28 * 28
# n_hidden1 = 300
# n_hidden2 = 100
# n_outputs = 10

# 可以看到每一层以及输出层都做了标准化
# X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
# training = tf.placeholder_with_default(False, shape=(), name="training")
#
# hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
# bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
# bn1_act = tf.nn.elu(bn1)
#
# hidden2 = tf.layers.dense(X, n_hidden2, name="hidden2")
# bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
# bn2_act = tf.nn.elu(bn2)
#
# logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
# logits = tf.layers.batch_normalization(logits_before_bn, training=training, momentum=0.9)
#
#
# #现在构建一个处理MNIST数据集的神经网络，每一层使用ELU作为激活函数
# reset_graph()
# batch_norm_momentum=0.9
#
# X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
# y=tf.placeholder(tf.int32, shape=(None), name="y")
# training=tf.placeholder_with_default(False, shape=(), name="training")
#
# with tf.name_scope("dnn"):
#     he_init=tf.variance_scaling_initializer()
#     # 为了防止重复设定参数，这里使用了Python的partial方法
#     # 对权重的初始化
#     my_batch_norm_layer=partial(
#         tf.layers.batch_normalization,
#         training=training,
#         momentum=batch_norm_momentum
#     )
#     my_dense_layer=partial(
#         tf.layers.dense,
#         kernel_initializer=he_init
#     )
#
#     hidden1=my_dense_layer(X,n_hidden1,name="hidden1")
#     bn1=tf.nn.elu(my_batch_norm_layer(hidden1))
#     hidden2=my_dense_layer(bn1,n_hidden2,name="hidden2")
#     bn2=tf.nn.elu(my_batch_norm_layer(hidden2))
#     logits_before_bn=my_dense_layer(bn2,n_outputs,name="outputs")
#     logits=my_batch_norm_layer(logits_before_bn)
#
# with tf.name_scope("loss"):
#     xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
#     loss=tf.reduce_mean(xentropy,name="loss")
# learning_rate = 0.01
# with tf.name_scope("train"):
#     optimizer=tf.train.GradientDescentOptimizer(learning_rate)
#     training_op = optimizer.minimize(loss)
# with tf.name_scope("eval"):
#     correct=tf.nn.in_top_k(logits,y,1)
#     accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
#
# init=tf.global_variables_initializer()
# saver=tf.train.Saver()
#
# n_epochs=20
# batch_size=200
#
# def shuffle_batch(X, y, batch_size):
#     rnd_idx = np.random.permutation(len(X))
#     n_batches = len(X) // batch_size
#     for batch_idx in np.array_split(rnd_idx, n_batches):
#         X_batch, y_batch = X[batch_idx], y[batch_idx]
#         yield X_batch, y_batch
#
# extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for X_batch , y_batch in shuffle_batch(X_train, y_train, batch_size):
#             sess.run([training_op,extra_update_ops],
#                      feed_dict={training:True,X:X_batch,y:y_batch}
#                      )
#         accuracy_val=accuracy.eval(feed_dict={X:X_valid,y:y_valid})
#         print(epoch, accuracy_val)
#     save_path=saver.save(sess,"./tf_logs/run-2019012801001/tensorflowmodel01normalization.ckpt")

'''
**说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。**
进入第二部分深度学习  

第十一章训练深层神经网络
在第十章以及之前tf练习中，训练的深度神经网络都只是简单的demo，如果增大数据量或是面对更多的特征，遇到的问题就会棘手起来。  
- 梯度消失（梯度爆炸），这会影响深度神经网络，并使较低层难以训练
- 训练效率
- 包含数百万参数的模型将会有严重的过拟合训练集的风险
本章中，教程从解释梯度消失问题开始，并探讨解决这个问题的一些最流行的解决方案。   
接下来讨论各种优化器，与普通梯度下降相比，它们可以加速大型模型的训练。   
介绍大型神经网络正则化技术。

### 6.梯度裁剪
减少梯度爆炸问题的另一种常用技术就是在**反向传播**过程中裁剪梯度，使其不超过某个阈值。  
该类优化的实现是在minimize()函数之前，通过调用compute_gradients()、clip_by_value()、apply_gradients()函数来进行裁剪梯度的操作
**步骤**
1.调用优化器compute_gradients()
2.创建裁剪梯度操作clip_by_calue()
3.创建操作使优化器的apply_gradients()方法应用
教程里构建了一个跟上一节类似的神经网络用来处理MNIST数据集，但是增加了神经网络层数
~~~python
reset_graph()

import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_hidden5 = 50
n_outputs = 10
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
    logits = tf.layers.dense(hidden5, n_outputs, activation=tf.nn.relu, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

#进行裁剪梯度，在这里要获取梯度，使用clip_by_value()函数进行裁剪，然后提交
threshold = 1.0
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# 1.调用优化器compute_gradients()
grads_and_vars = optimizer.compute_gradients(loss)

# 2.创建裁剪梯度操作clip_by_calue()
capped_gvs = [
    (tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]

# 3.创建操作使优化器的apply_gradients()方法应用
training_op = optimizer.apply_gradients(capped_gvs)

with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32),name="accuracy")

init=tf.global_variables_initializer()
saver=tf.train.Saver()

n_epochs=20
batch_size=200

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./tf_logs/run-2019013101001/tensorflowmodel01clip.ckpt")
    
0 Validation accuracy: 0.266
1 Validation accuracy: 0.5248
2 Validation accuracy: 0.7464
3 Validation accuracy: 0.8094
4 Validation accuracy: 0.8618
5 Validation accuracy: 0.8868
6 Validation accuracy: 0.9006
7 Validation accuracy: 0.9104
8 Validation accuracy: 0.9138
9 Validation accuracy: 0.9186
10 Validation accuracy: 0.9258
11 Validation accuracy: 0.932
12 Validation accuracy: 0.9296
13 Validation accuracy: 0.9388
14 Validation accuracy: 0.9428
15 Validation accuracy: 0.9448
16 Validation accuracy: 0.9446
17 Validation accuracy: 0.9468
18 Validation accuracy: 0.9474
19 Validation accuracy: 0.9498
~~~
可以发现，在裁剪梯度部分，主要处理的是对损失函数的计算，查了一下tf.clip_by_value方法的作用。  
tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。  


### 7.复用预训练层
教材里不提倡从零开始训练DNN，可以寻找现有的神经网络来解决类似的任务，然后复用该网络较低层数据，这就是**迁移学习**。  
这样做的好处是可以加快训练速率。  
如果新任务的输入图像与原始任务中使用的输入图像的大小不一致，则必须添加预处理步骤以将其大小调整为原始模型的预期大小。一般来说如果输入具有类似的低级层次的特征，则迁移学习将很好地工作。

### 8.复用Tensorflow模型
如果原始模型使用了Tensorflow进行训练，则可以较好地进行迁移。
- 载入图结构（Graph's structure）。使用import_meta_graph()函数完成，该函数能够将图操作载入默认图中，返回一个saver，用户可以使用并重载模型。需要注意的是要载入.meta文件。
~~~python
reset_graph()
saver=tf.train.import_meta_graph("./tf_logs/run-2019013101001/tensorflowmodel01clip.ckpt.meta")
~~~
- 获取训练所需的所有操作。如果不知道图结构，可以列出所有操作
~~~python
for op in tf.get_default_graph().get_operations():
    print(op.name)
~~~
结果会呈现一大串操作名称，也可以调用tensorboard来可视化。
可以看到整体结构
- 如果知道需要使用哪些操作，可以使用get_default_graph()下的get_tensor_by_name()和get_operation_by_name()获取
~~~python
X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")
accuracy=tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")
training_op=tf.get_default_graph().get_operation_by_name("GradientDescent")
~~~

- 现在可以开始一个会话，重载模型并继续训练
~~~python
reset_graph()

import tensorflow as tf

mnist=tf.keras.datasets.mnist.load_data()
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_hidden5 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
    logits = tf.layers.dense(hidden5, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01
threshold = 1.0

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
              for grad, var in grads_and_vars]
training_op = optimizer.apply_gradients(capped_gvs)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

with tf.Session() as sess:
    saver.restore(sess, "./tf_logs/run-2019013101001/tensorflowmodel01clip.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./tf_logs/run-2019013101002/tensorflowmodel01reuse.ckpt")

0 Validation accuracy: 0.9614
1 Validation accuracy: 0.9622
2 Validation accuracy: 0.963
3 Validation accuracy: 0.9628
4 Validation accuracy: 0.9648
5 Validation accuracy: 0.9638
6 Validation accuracy: 0.9664
7 Validation accuracy: 0.967
8 Validation accuracy: 0.967
9 Validation accuracy: 0.9672
10 Validation accuracy: 0.969
11 Validation accuracy: 0.9694
12 Validation accuracy: 0.9642
13 Validation accuracy: 0.9676
14 Validation accuracy: 0.9708
15 Validation accuracy: 0.9694
16 Validation accuracy: 0.972
17 Validation accuracy: 0.9702
18 Validation accuracy: 0.9722
19 Validation accuracy: 0.9708
~~~
会发现正确率比之前的要高。


'''
# reset_graph()
#
# import tensorflow as tf
#
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
# X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)
# X_valid, X_train = X_train[:5000], X_train[5000:]
# y_valid, y_train = y_train[:5000], y_train[5000:]
#
# n_inputs = 28 * 28
# n_hidden1 = 300
# n_hidden2 = 50
# n_hidden3 = 50
# n_hidden4 = 50
# n_hidden5 = 50
# n_outputs = 10
# X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
# y = tf.placeholder(tf.int32, shape=(None), name="y")
#
# with tf.name_scope("dnn"):
#     hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
#     hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
#     hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
#     hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
#     hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
#     logits = tf.layers.dense(hidden5, n_outputs, activation=tf.nn.relu, name="outputs")
#
# with tf.name_scope("loss"):
#     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#     loss = tf.reduce_mean(xentropy, name="loss")
#
# learning_rate = 0.01
#
# # 进行裁剪梯度，在这里要获取梯度，使用clip_by_value()函数进行裁剪，然后提交
# threshold = 1.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# # 1.调用优化器compute_gradients()
# grads_and_vars = optimizer.compute_gradients(loss)
#
# # 2.创建裁剪梯度操作clip_by_calue()
# capped_gvs = [
#     (tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]
#
# # 3.创建操作使优化器的apply_gradients()方法应用
# training_op = optimizer.apply_gradients(capped_gvs)
#
# with tf.name_scope("eval"):
#     correct = tf.nn.in_top_k(logits, y, 1)
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# n_epochs = 20
# batch_size = 200
#
#
# def shuffle_batch(X, y, batch_size):
#     rnd_idx = np.random.permutation(len(X))
#     n_batches = len(X) // batch_size
#     for batch_idx in np.array_split(rnd_idx, n_batches):
#         X_batch, y_batch = X[batch_idx], y[batch_idx]
#         yield X_batch, y_batch
#
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
#         print(epoch, "Validation accuracy:", accuracy_val)
#
#     save_path = saver.save(sess, "./tf_logs/run-2019013101001/tensorflowmodel01clip.ckpt")

# 0 Validation accuracy: 0.266
# 1 Validation accuracy: 0.5248
# 2 Validation accuracy: 0.7464
# 3 Validation accuracy: 0.8094
# 4 Validation accuracy: 0.8618
# 5 Validation accuracy: 0.8868
# 6 Validation accuracy: 0.9006
# 7 Validation accuracy: 0.9104
# 8 Validation accuracy: 0.9138
# 9 Validation accuracy: 0.9186
# 10 Validation accuracy: 0.9258
# 11 Validation accuracy: 0.932
# 12 Validation accuracy: 0.9296
# 13 Validation accuracy: 0.9388
# 14 Validation accuracy: 0.9428
# 15 Validation accuracy: 0.9448
# 16 Validation accuracy: 0.9446
# 17 Validation accuracy: 0.9468
# 18 Validation accuracy: 0.9474
# 19 Validation accuracy: 0.9498

# =======================================================================================================================



# reset_graph()
# saver = tf.train.import_meta_graph("./tf_logs/run-2019013101001/tensorflowmodel01clip.ckpt.meta")

# for op in tf.get_default_graph().get_operations():
#     print(op.name)
# X
# y
# hidden1/kernel/Initializer/random_uniform/shape
# hidden1/kernel/Initializer/random_uniform/min
# hidden1/kernel/Initializer/random_uniform/max
# hidden1/kernel/Initializer/random_uniform/RandomUniform
# hidden1/kernel/Initializer/random_uniform/sub
# hidden1/kernel/Initializer/random_uniform/mul
# hidden1/kernel/Initializer/random_uniform
# hidden1/kernel
# hidden1/kernel/Assign
# hidden1/kernel/read
# hidden1/bias/Initializer/zeros
# hidden1/bias
# hidden1/bias/Assign
# hidden1/bias/read
# dnn/hidden1/MatMul
# dnn/hidden1/BiasAdd
# dnn/hidden1/Relu
# hidden2/kernel/Initializer/random_uniform/shape
# hidden2/kernel/Initializer/random_uniform/min
# hidden2/kernel/Initializer/random_uniform/max
# hidden2/kernel/Initializer/random_uniform/RandomUniform
# hidden2/kernel/Initializer/random_uniform/sub
# hidden2/kernel/Initializer/random_uniform/mul
# hidden2/kernel/Initializer/random_uniform
# hidden2/kernel
# hidden2/kernel/Assign
# hidden2/kernel/read
# hidden2/bias/Initializer/zeros
# hidden2/bias
# hidden2/bias/Assign
# hidden2/bias/read
# dnn/hidden2/MatMul
# dnn/hidden2/BiasAdd
# dnn/hidden2/Relu
# hidden3/kernel/Initializer/random_uniform/shape
# hidden3/kernel/Initializer/random_uniform/min
# hidden3/kernel/Initializer/random_uniform/max
# hidden3/kernel/Initializer/random_uniform/RandomUniform
# hidden3/kernel/Initializer/random_uniform/sub
# hidden3/kernel/Initializer/random_uniform/mul
# hidden3/kernel/Initializer/random_uniform
# hidden3/kernel
# hidden3/kernel/Assign
# hidden3/kernel/read
# hidden3/bias/Initializer/zeros
# hidden3/bias
# hidden3/bias/Assign
# hidden3/bias/read
# dnn/hidden3/MatMul
# dnn/hidden3/BiasAdd
# dnn/hidden3/Relu
# hidden4/kernel/Initializer/random_uniform/shape
# hidden4/kernel/Initializer/random_uniform/min
# hidden4/kernel/Initializer/random_uniform/max
# hidden4/kernel/Initializer/random_uniform/RandomUniform
# hidden4/kernel/Initializer/random_uniform/sub
# hidden4/kernel/Initializer/random_uniform/mul
# hidden4/kernel/Initializer/random_uniform
# hidden4/kernel
# hidden4/kernel/Assign
# hidden4/kernel/read
# hidden4/bias/Initializer/zeros
# hidden4/bias
# hidden4/bias/Assign
# hidden4/bias/read
# dnn/hidden4/MatMul
# dnn/hidden4/BiasAdd
# dnn/hidden4/Relu
# hidden5/kernel/Initializer/random_uniform/shape
# hidden5/kernel/Initializer/random_uniform/min
# hidden5/kernel/Initializer/random_uniform/max
# hidden5/kernel/Initializer/random_uniform/RandomUniform
# hidden5/kernel/Initializer/random_uniform/sub
# hidden5/kernel/Initializer/random_uniform/mul
# hidden5/kernel/Initializer/random_uniform
# hidden5/kernel
# hidden5/kernel/Assign
# hidden5/kernel/read
# hidden5/bias/Initializer/zeros
# hidden5/bias
# hidden5/bias/Assign
# hidden5/bias/read
# dnn/hidden5/MatMul
# dnn/hidden5/BiasAdd
# dnn/hidden5/Relu
# outputs/kernel/Initializer/random_uniform/shape
# outputs/kernel/Initializer/random_uniform/min
# outputs/kernel/Initializer/random_uniform/max
# outputs/kernel/Initializer/random_uniform/RandomUniform
# outputs/kernel/Initializer/random_uniform/sub
# outputs/kernel/Initializer/random_uniform/mul
# outputs/kernel/Initializer/random_uniform
# outputs/kernel
# outputs/kernel/Assign
# outputs/kernel/read
# outputs/bias/Initializer/zeros
# outputs/bias
# outputs/bias/Assign
# outputs/bias/read
# dnn/outputs/MatMul
# dnn/outputs/BiasAdd
# dnn/outputs/Relu
# loss/SparseSoftmaxCrossEntropyWithLogits/Shape
# loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
# loss/Const
# loss/loss
# gradients/Shape
# gradients/grad_ys_0
# gradients/Fill
# gradients/loss/loss_grad/Reshape/shape
# gradients/loss/loss_grad/Reshape
# gradients/loss/loss_grad/Shape
# gradients/loss/loss_grad/Tile
# gradients/loss/loss_grad/Shape_1
# gradients/loss/loss_grad/Shape_2
# gradients/loss/loss_grad/Const
# gradients/loss/loss_grad/Prod
# gradients/loss/loss_grad/Const_1
# gradients/loss/loss_grad/Prod_1
# gradients/loss/loss_grad/Maximum/y
# gradients/loss/loss_grad/Maximum
# gradients/loss/loss_grad/floordiv
# gradients/loss/loss_grad/Cast
# gradients/loss/loss_grad/truediv
# gradients/zeros_like
# gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient
# gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim
# gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
# gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul
# gradients/dnn/outputs/Relu_grad/ReluGrad
# gradients/dnn/outputs/BiasAdd_grad/BiasAddGrad
# gradients/dnn/outputs/BiasAdd_grad/tuple/group_deps
# gradients/dnn/outputs/BiasAdd_grad/tuple/control_dependency
# gradients/dnn/outputs/BiasAdd_grad/tuple/control_dependency_1
# gradients/dnn/outputs/MatMul_grad/MatMul
# gradients/dnn/outputs/MatMul_grad/MatMul_1
# gradients/dnn/outputs/MatMul_grad/tuple/group_deps
# gradients/dnn/outputs/MatMul_grad/tuple/control_dependency
# gradients/dnn/outputs/MatMul_grad/tuple/control_dependency_1
# gradients/dnn/hidden5/Relu_grad/ReluGrad
# gradients/dnn/hidden5/BiasAdd_grad/BiasAddGrad
# gradients/dnn/hidden5/BiasAdd_grad/tuple/group_deps
# gradients/dnn/hidden5/BiasAdd_grad/tuple/control_dependency
# gradients/dnn/hidden5/BiasAdd_grad/tuple/control_dependency_1
# gradients/dnn/hidden5/MatMul_grad/MatMul
# gradients/dnn/hidden5/MatMul_grad/MatMul_1
# gradients/dnn/hidden5/MatMul_grad/tuple/group_deps
# gradients/dnn/hidden5/MatMul_grad/tuple/control_dependency
# gradients/dnn/hidden5/MatMul_grad/tuple/control_dependency_1
# gradients/dnn/hidden4/Relu_grad/ReluGrad
# gradients/dnn/hidden4/BiasAdd_grad/BiasAddGrad
# gradients/dnn/hidden4/BiasAdd_grad/tuple/group_deps
# gradients/dnn/hidden4/BiasAdd_grad/tuple/control_dependency
# gradients/dnn/hidden4/BiasAdd_grad/tuple/control_dependency_1
# gradients/dnn/hidden4/MatMul_grad/MatMul
# gradients/dnn/hidden4/MatMul_grad/MatMul_1
# gradients/dnn/hidden4/MatMul_grad/tuple/group_deps
# gradients/dnn/hidden4/MatMul_grad/tuple/control_dependency
# gradients/dnn/hidden4/MatMul_grad/tuple/control_dependency_1
# gradients/dnn/hidden3/Relu_grad/ReluGrad
# gradients/dnn/hidden3/BiasAdd_grad/BiasAddGrad
# gradients/dnn/hidden3/BiasAdd_grad/tuple/group_deps
# gradients/dnn/hidden3/BiasAdd_grad/tuple/control_dependency
# gradients/dnn/hidden3/BiasAdd_grad/tuple/control_dependency_1
# gradients/dnn/hidden3/MatMul_grad/MatMul
# gradients/dnn/hidden3/MatMul_grad/MatMul_1
# gradients/dnn/hidden3/MatMul_grad/tuple/group_deps
# gradients/dnn/hidden3/MatMul_grad/tuple/control_dependency
# gradients/dnn/hidden3/MatMul_grad/tuple/control_dependency_1
# gradients/dnn/hidden2/Relu_grad/ReluGrad
# gradients/dnn/hidden2/BiasAdd_grad/BiasAddGrad
# gradients/dnn/hidden2/BiasAdd_grad/tuple/group_deps
# gradients/dnn/hidden2/BiasAdd_grad/tuple/control_dependency
# gradients/dnn/hidden2/BiasAdd_grad/tuple/control_dependency_1
# gradients/dnn/hidden2/MatMul_grad/MatMul
# gradients/dnn/hidden2/MatMul_grad/MatMul_1
# gradients/dnn/hidden2/MatMul_grad/tuple/group_deps
# gradients/dnn/hidden2/MatMul_grad/tuple/control_dependency
# gradients/dnn/hidden2/MatMul_grad/tuple/control_dependency_1
# gradients/dnn/hidden1/Relu_grad/ReluGrad
# gradients/dnn/hidden1/BiasAdd_grad/BiasAddGrad
# gradients/dnn/hidden1/BiasAdd_grad/tuple/group_deps
# gradients/dnn/hidden1/BiasAdd_grad/tuple/control_dependency
# gradients/dnn/hidden1/BiasAdd_grad/tuple/control_dependency_1
# gradients/dnn/hidden1/MatMul_grad/MatMul
# gradients/dnn/hidden1/MatMul_grad/MatMul_1
# gradients/dnn/hidden1/MatMul_grad/tuple/group_deps
# gradients/dnn/hidden1/MatMul_grad/tuple/control_dependency
# gradients/dnn/hidden1/MatMul_grad/tuple/control_dependency_1
# clip_by_value/Minimum/y
# clip_by_value/Minimum
# clip_by_value/y
# clip_by_value
# clip_by_value_1/Minimum/y
# clip_by_value_1/Minimum
# clip_by_value_1/y
# clip_by_value_1
# clip_by_value_2/Minimum/y
# clip_by_value_2/Minimum
# clip_by_value_2/y
# clip_by_value_2
# clip_by_value_3/Minimum/y
# clip_by_value_3/Minimum
# clip_by_value_3/y
# clip_by_value_3
# clip_by_value_4/Minimum/y
# clip_by_value_4/Minimum
# clip_by_value_4/y
# clip_by_value_4
# clip_by_value_5/Minimum/y
# clip_by_value_5/Minimum
# clip_by_value_5/y
# clip_by_value_5
# clip_by_value_6/Minimum/y
# clip_by_value_6/Minimum
# clip_by_value_6/y
# clip_by_value_6
# clip_by_value_7/Minimum/y
# clip_by_value_7/Minimum
# clip_by_value_7/y
# clip_by_value_7
# clip_by_value_8/Minimum/y
# clip_by_value_8/Minimum
# clip_by_value_8/y
# clip_by_value_8
# clip_by_value_9/Minimum/y
# clip_by_value_9/Minimum
# clip_by_value_9/y
# clip_by_value_9
# clip_by_value_10/Minimum/y
# clip_by_value_10/Minimum
# clip_by_value_10/y
# clip_by_value_10
# clip_by_value_11/Minimum/y
# clip_by_value_11/Minimum
# clip_by_value_11/y
# clip_by_value_11
# GradientDescent/learning_rate
# GradientDescent/update_hidden1/kernel/ApplyGradientDescent
# GradientDescent/update_hidden1/bias/ApplyGradientDescent
# GradientDescent/update_hidden2/kernel/ApplyGradientDescent
# GradientDescent/update_hidden2/bias/ApplyGradientDescent
# GradientDescent/update_hidden3/kernel/ApplyGradientDescent
# GradientDescent/update_hidden3/bias/ApplyGradientDescent
# GradientDescent/update_hidden4/kernel/ApplyGradientDescent
# GradientDescent/update_hidden4/bias/ApplyGradientDescent
# GradientDescent/update_hidden5/kernel/ApplyGradientDescent
# GradientDescent/update_hidden5/bias/ApplyGradientDescent
# GradientDescent/update_outputs/kernel/ApplyGradientDescent
# GradientDescent/update_outputs/bias/ApplyGradientDescent
# GradientDescent
# eval/in_top_k/InTopKV2/k
# eval/in_top_k/InTopKV2
# eval/Cast
# eval/Const
# eval/accuracy
# init
# save/Const
# save/SaveV2/tensor_names
# save/SaveV2/shape_and_slices
# save/SaveV2
# save/control_dependency
# save/RestoreV2/tensor_names
# save/RestoreV2/shape_and_slices
# save/RestoreV2
# save/Assign
# save/Assign_1
# save/Assign_2
# save/Assign_3
# save/Assign_4
# save/Assign_5
# save/Assign_6
# save/Assign_7
# save/Assign_8
# save/Assign_9
# save/Assign_10
# save/Assign_11
# save/restore_all

# X = tf.get_default_graph().get_tensor_by_name("X:0")
# y = tf.get_default_graph().get_tensor_by_name("y:0")
# accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")
# training_op = tf.get_default_graph().get_operation_by_name("GradientDescent")
#
# for op in (X, y, accuracy, training_op):
#     tf.add_to_collection("my_important_ops", op)
# X, y, accuracy, training_op = tf.get_collection("my_important_ops")

# reset_graph()
#
# import tensorflow as tf
#
# mnist=tf.keras.datasets.mnist.load_data()
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
# X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)
# X_valid, X_train = X_train[:5000], X_train[5000:]
# y_valid, y_train = y_train[:5000], y_train[5000:]
#
# def shuffle_batch(X, y, batch_size):
#     rnd_idx = np.random.permutation(len(X))
#     n_batches = len(X) // batch_size
#     for batch_idx in np.array_split(rnd_idx, n_batches):
#         X_batch, y_batch = X[batch_idx], y[batch_idx]
#         yield X_batch, y_batch
#
#
# n_inputs = 28 * 28  # MNIST
# n_hidden1 = 300
# n_hidden2 = 50
# n_hidden3 = 50
# n_hidden4 = 50
# n_hidden5 = 50
# n_outputs = 10
#
# X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
# y = tf.placeholder(tf.int64, shape=(None), name="y")
#
# with tf.name_scope("dnn"):
#     hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
#     hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
#     hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
#     hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
#     hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
#     logits = tf.layers.dense(hidden5, n_outputs, name="outputs")
#
# with tf.name_scope("loss"):
#     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#     loss = tf.reduce_mean(xentropy, name="loss")
#
# with tf.name_scope("eval"):
#     correct = tf.nn.in_top_k(logits, y, 1)
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
#
# learning_rate = 0.01
# threshold = 1.0
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# grads_and_vars = optimizer.compute_gradients(loss)
# capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
#               for grad, var in grads_and_vars]
# training_op = optimizer.apply_gradients(capped_gvs)
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# n_epochs = 20
# batch_size = 200
#
# with tf.Session() as sess:
#     saver.restore(sess, "./tf_logs/run-2019013101001/tensorflowmodel01clip.ckpt")
#
#     for epoch in range(n_epochs):
#         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
#         print(epoch, "Validation accuracy:", accuracy_val)
#
#     save_path = saver.save(sess, "./tf_logs/run-2019013101002/tensorflowmodel01reuse.ckpt")
#



# save_path = saver.save(sess, "./tf_logs/run-2019013101002/tensorflowmodel01reuse.ckpt")

'''
**说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。**
进入第二部分深度学习  

第十一章训练深层神经网络
在第十章以及之前tf练习中，训练的深度神经网络都只是简单的demo，如果增大数据量或是面对更多的特征，遇到的问题就会棘手起来。  
- 梯度消失（梯度爆炸），这会影响深度神经网络，并使较低层难以训练
- 训练效率
- 包含数百万参数的模型将会有严重的过拟合训练集的风险
本章中，教程从解释梯度消失问题开始，并探讨解决这个问题的一些最流行的解决方案。   
接下来讨论各种优化器，与普通梯度下降相比，它们可以加速大型模型的训练。   
介绍大型神经网络正则化技术。

### 9.复用来自其它框架的模型
在之后的几个小节中，教程讨论了对其他框架模型的复用问题。看着很乏味……  
下面的代码显示了如何复制使用另一个框架训练的模型的第一个隐藏层的权重和偏置
对于想要复用的每个变量，我们找到它的初始化器的赋值操作，并得到它的第二个输入，它对应于初始化值。 当我们运行初始化程序时，我们使用feed_dict将初始化值替换为我们想要的值

#### 冻结较低层

#### 冻结缓存层
#### 调整，删除或替换较高层
......
我略过了很多，因为实在不想看了

#### Model Zoo
在哪里可以找到一个类似于目标任务训练的神经网络？ 
- 在自己的模型目录。 这是保存所有模型并组织它们的一个很好的理由，以便以后可以轻松地检索。 
- 在模型动物园中搜索。 许多人为了各种不同的任务而训练机器学习模型，并且善意地向公众发布预训练模型。

TensorFlow 在 https://github.com/tensorflow/models 中有自己的模型动物园。 包含了大多数最先进的图像分类网络，如 VGG，Inception 和 ResNet，包括代码，预训练模型和 工具来下载流行的图像数据集。
另一个流行的模型动物园是 **Caffe 模型动物园**。 它还包含许多在各种数据集（例如，ImageNet，Places 数据库，CIFAR10 等）上训练的计算机视觉模型（例如，LeNet，AlexNet，ZFNet，GoogLeNet，VGGNet，开始）。 

Saumitro Dasgupta 写了一个转换器，https://github.com/ethereon/caffetensorflow。

### n.实践指南
在本章中，我们已经涵盖了很多技术，你可能想知道应该使用哪些技术。当然，如果你能找到解决类似问题的方法，你应该尝试重用预训练的神经网络的一部分。
这个默认配置可能需要调整：
- 如果你找不到一个好的学习率（收敛速度太慢，所以你增加了训练速度，现在收敛速度很快，但是网络的准确性不是最理想的），那么你可以尝试添加一个学习率调整，如指数衰减。
- 如果你的训练集太小，你可以实现数据增强。
- 如果你需要一个稀疏的模型，你可以添加 l1 正则化混合（并可以选择在训练后将微小的权重归零）。 如果您需要更稀疏的模型，您可以尝试使用 FTRL 而不是 Adam 优化以及 l1 正则化。
- 如果在运行时需要快速模型，则可能需要删除批量标准化，并可能用 leakyReLU 替换 ELU 激活函数。 有一个稀疏的模型也将有所帮助。
有了这些指导方针，你现在已经准备好训练非常深的网络
'''

