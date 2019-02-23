'''
本例是为了配合NLP学习中的RNN网络，斯坦福CS224n课程里面使用的是Tensorflow进行，所以提前熟悉一下，使用Tensorflow生成一个echo-rnn。
说实话，这个例子是照着教程敲出来的，仅仅实现了，但是没有对后面的原理进行分析，目前还是在一步一步往前推。
代码同样更新在github：https://github.com/jwc19890114/-02-learning-file-100days
## 什么是RNN？
RNN是循环神经网络（Recurrent Neural Network）的英文缩写，它能结合数据点之间的特定顺序和幅值大小等多个特征，来处理序列数据。更重要的是，这种网络的输入序列可以是任意长度的。  
举一个简单的例子：数字时间序列，具体任务是根据先前值来预测后续值。在每个时间步中，循环神经网络的输入是当前值，以及一个表征该网络在之前的时间步中已经获得信息的状态向量。该状态向量是RNN网络的编码记忆单元，在训练网络之前初始化为零向量。
## RNN和CNN、DNN
- CNN 专门解决图像问题的，可用把它看作特征提取层，放在输入层上，最后用MLP 做分类。对于输入数据的维度约束是比较严重的，其训练和预测的输入数据都必须完全相同，但是如果用CNN去做一个智能问答系统，CNN会需要所有的问答数据都是固定的长度，这就很可怕了，这种模型会让问答变成对对子，必须每句话长短固定。而RNN没有这种约束。
- RNN 专门解决时间序列问题的，用来提取时间序列信息，放在特征提取层（如CNN）之后。RNN更多的考虑了神经元之间的联系，例如训练机器翻译，那么对于一个短语的翻译一定要考虑前因后果，这就需要模型对于数据输入的前后因素都要考虑
- DNN 说白了就是 多层网络，只是用了很多技巧，让它能够 deep 。

## RNN和LSTM
为了解决原始RNN网络结构存在的梯度消失（vanishing gradient）问题，设计了LSTM这种新的网络结构。但从本质上来讲，LSTM是一种特殊的循环神经网络。  
其和RNN的区别在于，对于特定时刻t，隐藏层输出st的计算方式不同。故对LSTM网络的训练的思路与RNN类似，仅前向传播关系式不同。
**本文借鉴[Erik Hallström教程](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767)**

'''

### 引入所需库
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# tensorflow警告记录
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

### 构建一个回声状态网络Echo-RNN，能记忆输入数据信息，在若干时间步后将其回传。
num_epchos = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length // batch_size // truncated_backprop_length


### 生成随机训练数据，输入为一个随机的二元向量，在echo_step（3）个时间步后，得到输入的回声
def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    '''
    将数据重构为新矩阵。神经网络的训练，需要利用小批次数据（mini-batch），
    来近似得到关于神经元权重的损失函数梯度。
    在训练过程中，随机批次操作能防止过拟合和降低硬件压力。
    整个数据集通过数据重构转化为一个矩阵，并将其分解为多个小批次数据。
    '''
    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))
    # print(x)
    # print("**************")
    # print(y)
    return x, y


### 构建计算视图
# 占位符是计算图的“起始节点”。在运行每个计算图时，批处理数据被传递到占位符中。
# 另外，RNN状态向量也是存储在占位符中，在每一次运行后更新输出。
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

### 权重
W = tf.Variable(np.random.rand(state_size + 1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)
W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

### 拆分序列
# 开始构建RNN计算视图的下个部分，首先我们要以相邻的时间步分割批数据。
# 可以按批次分解各列，转成list格式文件。RNN会同时从不同位置开始训练时间序列
# 在我们的时间序列数据中，在三个位置同时开启训练，所以在前向传播时需要保存三个状态。我们在参数定义时就已经考虑到这一点了，
# 故将init_state设置为3，[batch_size, state_size]。
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

### 前向传播
current_state=init_state
states_series=[]
for current_input in inputs_series:
    current_input=tf.reshape(current_input,[batch_size,1])
    input_and_state_concatenated=tf.concat([current_input, current_state],1)

    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    states_series.append(next_state)
    current_state = next_state

### 计算损失
# 这里调用的tosparse_softmax_cross_entropy_with_logits函数，能在内部算得softmax函数值后，继续计算交叉熵。
logits_series=[tf.matmul(state, W2)+b2 for state in states_series]
predictions_series=[tf.nn.softmax(logits) for logits in logits_series]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss=tf.reduce_mean(losses)

train_step=tf.train.AdagradOptimizer(0.3).minimize(total_loss)

### 可视化结果
def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list=[]

    for epoch_idx in range(num_epchos):
        x,y=generateData()
        _current_state=np.zeros((batch_size,state_size))
        print('New data, epoch', epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:, start_idx:end_idx]
            batchY = y[:, start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    init_state: _current_state
                })

            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()