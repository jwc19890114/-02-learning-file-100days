'''
**说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。**
进入第二部分深度学习  

第十四章循环神经网络
循环神经网络可以分析时间序列数据，诸如股票价格，并告诉你什么时候买入和卖出。在自动驾驶系统中，他们可以预测行车轨迹，避免发生交通意外。  
循环神经网络可以在任意长度的序列上工作，而不是之前讨论的只能在固定长度的输入上工作的网络。  
举个例子，它们可以把语句，文件，以及语音范本作为输入，使得它们在诸如自动翻译，语音到文本或者情感分析（例如，读取电影评论并提取评论者关于该电影的感觉）的自然语言处理系统中极为有用。

另外，循环神经网络的预测能力使得它们具备令人惊讶的创造力。  
可以要求它们去预测一段旋律的下几个音符，随机选取这些音符的其中之一并演奏它。然后要求网络给出接下来最可能的音符，演奏它，如此周而复始。  
同样，循环神经网络可以生成语句，图像标注等。

在本章中，教程介绍以下几点
- 循环神经网络背后的基本概念
- 循环神经网络所面临的主要问题（在第11章中讨论的消失／爆炸的梯度），广泛用于反抗这些问题的方法：LSTM 和 GRU cell（单元）。
- 展示如何用 TensorFlow 实现循环神经网络。最终我们将看看及其翻译系统的架构。

### 6.预测时间序列
RNN是如何处理时间序列，如股价，气温，脑电波模式等等。在这里将训练一个 RNN 来预测生成的时间序列中的下一个值。  
每个训练实例是从时间序列中随机选取的 20 个连续值的序列，目标序列与输入序列相同，除了向后移动一个时间步。
在这里生成一张时序图，从中间抽取12-14.5的时间段生成右图作为训练实例。
~~~python
t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title("A time series (generated)", fontsize=14)
plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0, 30, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.title("A training instance", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.legend(loc="upper left")
plt.xlabel("Time")


save_fig("time_series_plot")
plt.show()

X_batch,y_batch=next_batch(1,n_steps)
np.c_[X_batch[0],y_batch[0]]

~~~
构建一个RNN，包含有100个循环神经元，要在20个时间步长中展开，每次输入仅包含一个特征。
~~~python
reset_graph()
n_steps=20
n_inputs=1
n_neurons=100
n_outputs=1

X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None, n_steps, n_outputs])

cell=tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu)
outputs,states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
~~~
每一个时间步长都会有一个大小为100的输出向量，但是如前面参数设置，只想要大小为1的输出值，解决方法就是**打包**，放入OutputProjectionWrapper。
~~~python
reset_graph()
n_steps=20
n_inputs=1
n_neurons=100
n_outputs=1

X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None, n_steps, n_outputs])

# cell=tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu)
# outputs,states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
cell=tf.contrib.rnn.OutputProjectionWrapper(
    tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu),
    output_size=n_outputs
)
outputs, states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
~~~
使用均方误差（MSE），就像我们在之前的回归任务中所做的那样。 接下来，我们将像往常一样创建一个 Adam 优化器，训练操作和变量初始化操作
~~~python
learning_rate=0.001
loss=tf.reduce_mean(tf.square(outputs-y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(loss)

init=tf.global_variables_initializer()

saver=tf.train.Saver()

n_iterations = 1500
batch_size = 50
~~~
保存模型
~~~python
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    saver.save(sess, "./tf_logs/20190203001/my_time_series_model")  # not shown in the book

~~~
加载刚才保存的模型，并生成训练图
~~~python
with tf.Session() as sess:
    saver.restore(sess,"./tf_logs/20190203001/my_time_series_model")

    X_new=time_series(np.array(t_instance[:-1].reshape(-1,n_steps,n_inputs)))
    y_pred=sess.run(outputs,feed_dict={X:X_new})

    print(y_pred)

plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")

save_fig("time_series_pred_plot")
plt.show()
~~~
#### 现在可以做一些有意思的事情了
在前面对RNN介绍的时候提到过，RNN是可以预测未来时刻样本的，即是说，可以用来生成新的模型。  
- 教程中在这里为模型提供长度为n_steps的种子序列（比如全零序列），通过模型预测下一时刻的值；
- 把该预测值添加到种子序列的末尾，用最后面长度为n_steps的序列做为新的种子序列，做下一次预测，以此类推生成预测序列。

同样，加载之前生成的模型，按照教程中的处理，生成一个种子序列sequence，通过模型预测下一时刻值。
~~~python
with tf.Session() as sess:
    saver.restore(sess,"./tf_logs/20190203001/my_time_series_model")

    sequence=[0.]*n_steps
    for iteration in range(300):
        X_batch=np.array(sequence[-n_steps:]).reshape(1,n_steps,1)
        y_pred=sess.run(outputs,feed_dict={X:X_batch})
        sequence.append(y_pred[0,-1,0])

plt.figure(figsize=(8,4))
plt.plot(np.arange(len(sequence)), sequence, "b-")
plt.plot(t[:n_steps], sequence[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
~~~
进一步生成两组时间序列，这块没看懂，感觉意思是要对比，右侧的时间序列与原图类似，但是我做出来的不太一致。

~~~python
with tf.Session() as sess:
    saver.restore(sess,"./tf_logs/20190203001/my_time_series_model")

    sequence1=[0.]*n_steps
    for iteration in range(len(t)-n_steps):
        X_batch=np.array(sequence1[-n_steps:]).reshape(1,n_steps,1)
        y_pred=sess.run(outputs,feed_dict={X:X_batch})
        sequence1.append(y_pred[0,-1,0])

    sequence2 = [time_series(i * resolution + t_min + (t_max-t_min/3)) for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence2[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence2.append(y_pred[0, -1, 0])

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.plot(t, sequence1, "b-")
plt.plot(t[:n_steps], sequence1[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.plot(t, sequence2, "b-")
plt.plot(t[:n_steps], sequence2[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
save_fig("creative_sequence_plot")
plt.show()
~~~

'''

# Common imports
import numpy as np
import os
import tensorflow as tf


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# t_min, t_max = 0, 30
# resolution = 0.1
#
# def time_series(t):
#     return t * np.sin(t) / 3 + 2 * np.sin(t*5)
#
# def next_batch(batch_size, n_steps):
#     t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
#     Ts = t0 + np.arange(0., n_steps + 1) * resolution
#     ys = time_series(Ts)
#     return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)
#
# t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))
#
# n_steps = 20
# t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)
#
# plt.figure(figsize=(11,4))
# plt.subplot(121)
# plt.title("A time series (generated)", fontsize=14)
# plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
# plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
# plt.legend(loc="lower left", fontsize=14)
# plt.axis([0, 30, -17, 13])
# plt.xlabel("Time")
# plt.ylabel("Value")
#
# plt.subplot(122)
# plt.title("A training instance", fontsize=14)
# plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
# plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
# plt.legend(loc="upper left")
# plt.xlabel("Time")


# save_fig("time_series_plot")
# plt.show()

# X_batch,y_batch=next_batch(1,n_steps)
# np.c_[X_batch[0],y_batch[0]]


# reset_graph()
# n_steps=20
# n_inputs=1
# n_neurons=100
# n_outputs=1
#
# X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
# y=tf.placeholder(tf.float32,[None, n_steps, n_outputs])
#
# # cell=tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu)
# # outputs,states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
# cell=tf.contrib.rnn.OutputProjectionWrapper(
#     tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu),
#     output_size=n_outputs
# )
# outputs, states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
#
# learning_rate=0.001
# loss=tf.reduce_mean(tf.square(outputs-y))
# optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
# training_op=optimizer.minimize(loss)
#
# init=tf.global_variables_initializer()
#
# saver=tf.train.Saver()
#
# n_iterations = 1500
# batch_size = 50

# with tf.Session() as sess:
#     init.run()
#     for iteration in range(n_iterations):
#         X_batch, y_batch = next_batch(batch_size, n_steps)
#         sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         if iteration % 100 == 0:
#             mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
#             print(iteration, "\tMSE:", mse)
#
#     saver.save(sess, "./tf_logs/20190203001/my_time_series_model")  # not shown in the book

# with tf.Session() as sess:
#     saver.restore(sess,"./tf_logs/20190203001/my_time_series_model")
#
#     X_new=time_series(np.array(t_instance[:-1].reshape(-1,n_steps,n_inputs)))
#     y_pred=sess.run(outputs,feed_dict={X:X_new})
#
#     print(y_pred)
#
# plt.title("Testing the model", fontsize=14)
# plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
# plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
# plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
# plt.legend(loc="upper left")
# plt.xlabel("Time")
#
# save_fig("time_series_pred_plot")
# plt.show()

# with tf.Session() as sess:
#     saver.restore(sess,"./tf_logs/20190203001/my_time_series_model")
#
#     sequence=[0.]*n_steps
#     for iteration in range(300):
#         X_batch=np.array(sequence[-n_steps:]).reshape(1,n_steps,1)
#         y_pred=sess.run(outputs,feed_dict={X:X_batch})
#         sequence.append(y_pred[0,-1,0])
#
# plt.figure(figsize=(8,4))
# plt.plot(np.arange(len(sequence)), sequence, "b-")
# plt.plot(t[:n_steps], sequence[:n_steps], "b-", linewidth=3)
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.show()


# with tf.Session() as sess:
#     saver.restore(sess,"./tf_logs/20190203001/my_time_series_model")
#
#     sequence1=[0.]*n_steps
#     for iteration in range(len(t)-n_steps):
#         X_batch=np.array(sequence1[-n_steps:]).reshape(1,n_steps,1)
#         y_pred=sess.run(outputs,feed_dict={X:X_batch})
#         sequence1.append(y_pred[0,-1,0])
#
#     sequence2 = [time_series(i * resolution + t_min + (t_max-t_min/3)) for i in range(n_steps)]
#     for iteration in range(len(t) - n_steps):
#         X_batch = np.array(sequence2[-n_steps:]).reshape(1, n_steps, 1)
#         y_pred = sess.run(outputs, feed_dict={X: X_batch})
#         sequence2.append(y_pred[0, -1, 0])
#
# plt.figure(figsize=(11,4))
# plt.subplot(121)
# plt.plot(t, sequence1, "b-")
# plt.plot(t[:n_steps], sequence1[:n_steps], "b-", linewidth=3)
# plt.xlabel("Time")
# plt.ylabel("Value")
#
# plt.subplot(122)
# plt.plot(t, sequence2, "b-")
# plt.plot(t[:n_steps], sequence2[:n_steps], "b-", linewidth=3)
# plt.xlabel("Time")
# save_fig("creative_sequence_plot")
# plt.show()

'''
**说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。**
进入第二部分深度学习  

第十四章循环神经网络
循环神经网络可以分析时间序列数据，诸如股票价格，并告诉你什么时候买入和卖出。在自动驾驶系统中，他们可以预测行车轨迹，避免发生交通意外。  
循环神经网络可以在任意长度的序列上工作，而不是之前讨论的只能在固定长度的输入上工作的网络。  
举个例子，它们可以把语句，文件，以及语音范本作为输入，使得它们在诸如自动翻译，语音到文本或者情感分析（例如，读取电影评论并提取评论者关于该电影的感觉）的自然语言处理系统中极为有用。

另外，循环神经网络的预测能力使得它们具备令人惊讶的创造力。  
可以要求它们去预测一段旋律的下几个音符，随机选取这些音符的其中之一并演奏它。然后要求网络给出接下来最可能的音符，演奏它，如此周而复始。  
同样，循环神经网络可以生成语句，图像标注等。

在本章中，教程介绍以下几点
- 循环神经网络背后的基本概念
- 循环神经网络所面临的主要问题（在第11章中讨论的消失／爆炸的梯度），广泛用于反抗这些问题的方法：LSTM 和 GRU cell（单元）。
- 展示如何用 TensorFlow 实现循环神经网络。最终我们将看看及其翻译系统的架构。

### 7.长时训练困难
在训练长序列的 RNN 模型时，那么就需要把 RNN 在时间维度上展开成很深的神经网络。正如任何深度神经网络一样，其面临着梯度消失/爆炸的问题，使训练无法终止或收敛。  
在之前教程中提到过不少方法（参数初始化方式，非饱和的激活函数（如 ReLU），批量规范化（Batch Normalization）， 梯度截断（Gradient Clipping），更快的优化器），但是在实际使用中仍会遇到训练缓慢的问题。  
最简单的方法是在训练阶段仅仅展开限定时间步长的 RNN 网络，称为**截断时间反向传播算法**。  

在TensorFlow中通过截断输入序列来简单实现这种功能。例如在时间序列预测问题上可以在训练时减小n_steps来实现截断。这种方法会限制模型在长期模式的学习能力。  
另一种变通方案是确保缩短的序列中包含旧数据和新数据，从而使模型获得两者信息（如序列同时包含最近五个月的数据，最近五周的和最近五天的数据）。  
其实就是在解决RNN的记忆问题，包括两个问题：
1.如何确保从之前的细分类中获取的数据有效性。
2.第一个输入的记忆会在长时间运行的RNN网络中逐渐淡去。

为了解决这些问题，就出现了各种能够携带长时记忆的神经单元的变体。

### 8.长短时记忆LSTM
长短时记忆单元，如果把 LSTM 单元看作一个黑盒，从外围看它和基本形式的记忆单元很相似，但 LSTM 单元会比基本单元性能更好，收敛更快，能够感知数据的长时依赖。TensorFlow 中通过BasicLSTMCell实现 LSTM 单元。
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)  
图
LSTM 单元状态分为两个向量：h^{(t)} 和 c^{(t)}（c代表 cell）。可以简单认为 h^{(t)} 是短期记忆状态，c^{(t)} 是长期记忆状态。  
**LSTM 单元的核心思想是其能够学习从长期状态中存储什么，忘记什么，读取什么**  

- 长期状态 c^{(t-1)} 从左向右在网络中传播，依次经过遗忘门（forget gate）时丢弃一些记忆；
- 加法操作增加一些记忆（从输入门中选择一些记忆）。
- 输出 c^{(t)} 不经任何转换直接输出。

每个单位时间步长后，都有一些记忆被抛弃，新的记忆被添加进来。  
另一方面，长时状态经过 tanh 激活函数通过输出门得到短时记忆 h^{(t)}，同时它也是这一时刻的单元输出结果 y^{(t)}。  
LSTM 单元能够学习到识别重要输入（输入门作用），存储进长时状态，并保存必要的时间（遗忘门功能），并学会提取当前输出所需要的记忆。  
这也解释了 LSTM 单元能够在提取长时序列，长文本，录音等数据中的长期模式的惊人成功的原因。  

其中主要的全连接层输出 g^{(t)}，它的常规任务就是解析当前的输入 x^{(t)} 和前一时刻的短时状态 h^{(t-1)}。在基本形式的 RNN 单元中，就与这种形式一样，直接输出了 h^{(t)} 和 y^{(t)}。与之不同的是 LSTM 单元会将一部分 g^{(t)} 存储在长时状态中。
其它三个全连接层被称为门控制器（gate controller）。其采用 Logistic 作为激活函数，输出范围在 0 到 1 之间。在图中，这三个层的输出提供给了逐元素乘法操作，当输入为 0 时门关闭，输出为 1 时门打开。分别为：
- 遗忘门（forget gat）由 f^{(t)} 控制，来决定哪些长期记忆需要被擦除；
- 输入门（input gate） 由 i^{(t)} 控制，它的作用是处理哪部分 g^{(t)} 应该被添加到长时状态中，也就是为什么被称为部分存储。
- 输出门（output gate）由 o^{(t)} 控制，在这一时刻的输出 h^{(t)} 和 y^{(t)} 就是由输出门控制的，从长时状态中读取的记忆。

### 9.窥孔连接
基本形式的 LSTM 单元中，门的控制仅有当前的输入 x^{(t)} 和前一时刻的短时状态 h^{(t-1)}。不妨让各个控制门窥视一下长时状态，获取一些上下文信息不失为一种尝试。  
在tensorflow中使用use_peepholes=True来进行设置。  
lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, use_peepholes=True)  

### 10.GRU单元
门控循环单元是 LSTM 单元的简化版本，能实现同样的性能，这也说明了为什么它能越来越流行。  
- 长时状态和短时状态合并为一个向量 h^{(t)}。
- 用同一个门控制遗忘门和输入门。如果门控制输入 1，输入门打开，遗忘门关闭，反之亦然。也就是说，如果当有新的记忆需要存储，那么就必须实现在其对应位置事先擦除该处记忆。这也构成了 LSTM 本身的常见变体。（没看懂）
- GRU 单元取消了输出门，单元的全部状态就是该时刻的单元输出。与此同时，增加了一个控制门 r^{(t)} 来控制哪部分前一时间步的状态在该时刻的单元内呈现。
在tensorflow中创建GRU  
gru_cell = tf.contrib.rnn.GRUCell(n_units=n_neurons)


~~~python
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
n_layers = 3

reset_graph()
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
lstm_cell = [
    tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)
    for layer in range(n_layers)
]
multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

top_layer_h_state = states[-1][1]
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
# print(states)

n_epochs = 10
batch_size = 150

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
X_test=X_test.reshape((-1,n_steps,n_inputs))

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
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch,"Test accuracy:", acc_test)

0 Last batch accuracy: 0.97333336 Test accuracy: 0.9558
1 Last batch accuracy: 0.96 Test accuracy: 0.9665
2 Last batch accuracy: 0.96 Test accuracy: 0.968
3 Last batch accuracy: 1.0 Test accuracy: 0.9813
4 Last batch accuracy: 0.9866667 Test accuracy: 0.9843
5 Last batch accuracy: 0.98 Test accuracy: 0.9787
6 Last batch accuracy: 0.9866667 Test accuracy: 0.9863
7 Last batch accuracy: 0.9866667 Test accuracy: 0.9875
8 Last batch accuracy: 0.9866667 Test accuracy: 0.9843
9 Last batch accuracy: 0.99333334 Test accuracy: 0.9884
~~~
本节，使用tensorflow构建了一个LSTM，但是如何将这个RNN使用到NLP中，将是下一节的重点。
'''
# n_steps = 28
# n_inputs = 28
# n_neurons = 150
# n_outputs = 10
# n_layers = 3
#
# reset_graph()
# learning_rate = 0.001
#
# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# y = tf.placeholder(tf.int32, [None])
# lstm_cell = [
#     tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)
#     for layer in range(n_layers)
# ]
# multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
# outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
#
# top_layer_h_state = states[-1][1]
# logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
# xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
# loss = tf.reduce_mean(xentropy, name="loss")
#
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(loss)
# correct = tf.nn.in_top_k(logits, y, 1)
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# init = tf.global_variables_initializer()
# # print(states)
#
# n_epochs = 10
# batch_size = 150
#
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
# X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)
# X_valid, X_train = X_train[:5000], X_train[5000:]
# y_valid, y_train = y_train[:5000], y_train[5000:]
# X_test=X_test.reshape((-1,n_steps,n_inputs))
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
#             X_batch = X_batch.reshape((-1, n_steps, n_inputs))
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#         acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#         print(epoch, "Last batch accuracy:", acc_batch,"Test accuracy:", acc_test)



