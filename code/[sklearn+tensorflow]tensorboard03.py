"""
**源代码来自莫烦python(https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/4-1-tensorboard1/)**  
**今日重点**
- 读懂教程中代码，手动重写一遍，在浏览器中获取到训练数据
 
Tensorboard是一个神经网络可视化工具，通过使用本地服务器在浏览器上查看神经网络训练日志，生成相应的可是画图，帮助炼丹师优化神经网络。  
油管上有单迪伦·马内在2017年做的汇报，很惊艳。主要包括了以下主要功能  
- 可视化网络
- 可视化训练过程
- 多模型效果可视化对比

先看一下教程提供的原始代码（不包括tensorboard构造），就是一个两层（包括输出）的线性回归网络。
因为有了训练的数据，教程使用numpy生成部分模拟数据。
~~~python
import tensorflow as tf
import numpy as np

#构造模拟数据
x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
#使用np.random.normal()生成一些噪声
noise=np.random.normal(0,0.05,x_data).astype(np.float32)
y_data=np.square(x_data)-0.5+noise
~~~
制作对Weights和biases的变化图表distributions。
我们为层中的Weights设置变化图, tensorflow中提供了tf.histogram_summary()方法,用来绘制图片, 第一个参数是图表的名称, 第二个参数是图表要记录的变量
~~~python
def add_layer(inputs, in_size, out_size, n_layer,activation_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer'):
        # add one more layer and return the output of this layer
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='W')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='b')
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs
~~~

~~~python
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='X_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10,n_layer=1, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2,activation_function=None)
~~~

Loss 的变化图和之前设置的方法略有不同. loss是在tesnorBorad 的event下面的, 这是由于我们使用的是tf.scalar_summary() 方法.
~~~python
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
~~~

以上这些仅仅可以记录很绘制出训练的图表， 但是不会记录训练的数据。 为了较为直观显示训练过程中每个参数的变化，我们每隔上50次就记录一次结果 , 同时我们也应注意, merged 也是需要run 才能发挥作用的.
~~~python
sess=tf.Session()
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('./logs/layer2',sess.graph)

sess.run(tf.global_variables_initializer())

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        rs=sess.run(merged, feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(rs,i)
~~~


"""

# import tensorflow as tf
# import numpy as np
#
# #构造模拟数据
# x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
# #使用np.random.normal()生成一些噪声
# noise=np.random.normal(0,0.05,x_data.shape)
# y_data=np.square(x_data)-0.5+noise
#
#
# def add_layer(inputs, in_size, out_size, n_layer,activation_function=None):
#     layer_name='layer%s'%n_layer
#     with tf.name_scope('layer'):
#         # add one more layer and return the output of this layer
#         with tf.name_scope('weights'):
#             Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='W')
#             tf.summary.histogram(layer_name+'/weights',Weights)
#         with tf.name_scope('biases'):
#             biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='b')
#             tf.summary.histogram(layer_name+'/biases',biases)
#         with tf.name_scope('wx_plus_b'):
#             Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
#
#         if activation_function is None:
#             outputs = Wx_plus_b
#         else:
#             outputs = activation_function(Wx_plus_b)
#
#         tf.summary.histogram(layer_name + '/outputs', outputs)
#     return outputs
#
# with tf.name_scope('inputs'):
#     xs = tf.placeholder(tf.float32, [None, 1], name='X_input')
#     ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
#
# # add hidden layer
# l1 = add_layer(xs, 1, 10,n_layer=1, activation_function=tf.nn.relu)
# # add output layer
# prediction = add_layer(l1, 10, 1, n_layer=2,activation_function=None)
#
# with tf.name_scope('loss'):
#     loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
#     tf.summary.scalar('loss',loss)
#
# sess=tf.Session()
# merged=tf.summary.merge_all()
# writer=tf.summary.FileWriter('./logs/layer2',sess.graph)
#
# sess.run(tf.global_variables_initializer())
#
# with tf.name_scope('train'):
#     train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# for i in range(1000):
#     sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
#     if i%50==0:
#         rs=sess.run(merged, feed_dict={xs:x_data,ys:y_data})
#         writer.add_summary(rs,i)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tqdm

# ####  1. 生成变量监控信息并定义生成监控信息日志的操作。

SUMMARY_DIR = "./logs/rnn3"
BATCH_SIZE = 100
TRAIN_STEPS = 3000


# var给出了需要记录的张量,name给出了在可视化结果中显示的图表名称，这个名称一般和变量名一致
def variable_summaries(var, name):
    # 将生成监控信息的操作放在同一个命名空间下
    with tf.name_scope('summaries'):
        # 通过tf.histogram_summary函数记录张量中元素的取值分布
        # tf.summary.histogram函数会生成一个Summary protocol buffer.
        # 将Summary 写入TensorBoard 门志文件后，在HISTOGRAMS 栏，和
        # DISTRIBUTION 栏下都会出现对应名称的图表。和TensorFlow 中其他操作类似，
        # tf.summary.histogram 函数不会立刻被执行，只有当sess.run 函数明确调用这个操作时， TensorFlow
        # 才会具正生成并输出Summary protocol buffer.

        tf.summary.histogram(name, var)

        # 计算变量的平均值，并定义生成平均值信息日志的操作，记录变量平均值信息的日志标签名
        # 为'mean/'+name,其中mean为命名空间，/是命名空间的分隔符
        # 在相同命名空间中的监控指标会被整合到同一栏中，name则给出了当前监控指标属于哪一个变量

        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)

        # 计算变量的标准差，并定义生成其日志文件的操作
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


# #### 2.生成一层全链接的神经网络
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 将同一层神经网络放在一个统一的命名空间下
    with tf.name_scope(layer_name):
        # 声明神经网络边上的权值，并调用权重监控信息日志的函数
        with tf.name_scope('weight'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')

        # 声明神经网络边上的偏置，并调用权重监控信息日志的函数
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            # 记录神经网络节点输出在经过激活函数之前的分布
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, name='activation')

        """
        对于layerl ，因为使用了ReLU函数作为激活函数，所以所有小于0的值部被设为了0。于是在激活后
        的layerl/activations 图上所有的值都是大于0的。而对于layer2 ，因为没有使用激活函数，
        所以layer2/activations 和layer2/pre_activations 一样。
        """
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations


def main():
    mnist = input_data.read_data_sets(".\MNIST_data/", one_hot=True)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x_input')
        y = tf.placeholder(tf.float32, [None, 10], name='y_input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)
        # 将输入变量还原成图片的像素矩阵，并通过tf.iamge_summary函数定义将当前的图片信息写入日志的操作
    hidden1 = nn_layer(x, 784, 500, 'layer1')
    y_out = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)

    # 计算交叉熵
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=y))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    """
        计算模型在当前给定数据上的正确率，并定义生成正确率监控日志的操作。如果在sess.run()
        时给定的数据是训练batch，那么得到的正确率就是在这个训练batch上的正确率;如果
        给定的数据为验证或者测试数据，那么得到的正确率就是在当前模型在验证或者测试数据上
        的正确率。
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()

        for i in tqdm.tqdm(range(TRAIN_STEPS)):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 运行训练步骤以及所有的日志生成操作，得到这次运行的日志。
            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y: ys})
            # 将得到的所有日志写入日志文件，这样TensorBoard程序就可以拿到这次运行所对应的
            # 运行信息。
            writer.add_summary(summary, i)

    writer.close()


if __name__ == '__main__':
    main()
