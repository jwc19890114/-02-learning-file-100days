'''
## 使用Tensorflow实现逻辑回归
讨论如何通过系统中的线性模型预测连续值参数。  
如果对象要求在两个选择中抉择呢？  
答案简单，我们转换为可以解决一个分类问题。  
### 数据集
使用MNIST数据集（看来终于要试试这个大名鼎鼎的数据集了，暑假的时候看了一下。）  
MNIST数据集包括55000训练集和10000测试集。  
图片是28*28*1，每一张图代表一个手写数字（0-9任意一个）。  
我们创建一个大小为764的特征向量，仅使用0和1图。
### 逻辑回归
在线性回归中，使用线性方程进行预测。另一方面，逻辑回归中我们通过预测一个二元标签0或1。  
在逻辑回归中，预测输出是输入样本属于我们的情况下数字“1”的目标类的概率。  
损失函数包括两方面，并且在每一个采样中考虑到二进制标记，只有一个是非0的。  
## 处理流程
- 引入数据集，包含0和1的部分。
- 实现逻辑回归。用于逻辑回归的代码很大程度上受到[《训练卷积神经网络作为分类器》](http://www.machinelearninguru.com/deep_learning/tensorflow/neural_networks/cnn_classifier/cnn_classifier.html)的启发。我们参考上述帖子以更好地理解实现细节。  
- 教程中，仅仅解释数据处理、实现逻辑回归，其余部分可以参考之前CNN分类器。
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import numpy as np
import pandas as pd

# mnist=input_data.read_data_sets(".\MNIST_data/", reshape=True, one_hot=False)

######################################
######### Necessary Flags ############
######################################

tf.app.flags.DEFINE_string(
    'train_path', os.path.dirname(os.path.abspath(__file__)) + '/train_logs',
    'Directory where event logs are written to.')

tf.app.flags.DEFINE_string(
    'checkpoint_path',
    os.path.dirname(os.path.abspath(__file__)) + '/checkpoints',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer('max_num_checkpoint', 10,
                            'Maximum number of checkpoints that TensorFlow will keep.')

tf.app.flags.DEFINE_integer('num_classes', 2,
                            'Number of model clones to deploy.')
## **注意这部分要求参数为int，np.power需要转为int**
tf.app.flags.DEFINE_integer('batch_size', int(np.power(2, 9)),
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('num_epochs', 10,
                            'Number of epochs for training.')

##########################################
######## Learning rate flags #############
##########################################
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 1, 'Number of epoch pass to decay learning rate.')

#########################################
########## status flags #################
#########################################
tf.app.flags.DEFINE_boolean('is_training', False,
                            'Training/Testing.')

tf.app.flags.DEFINE_boolean('fine_tuning', False,
                            'Fine tuning is desired or not?.')

tf.app.flags.DEFINE_boolean('online_test', True,
                            'Fine tuning is desired or not?.')

tf.app.flags.DEFINE_boolean('allow_soft_placement', True,
                            'Automatically put the variables on CPU if there is no GPU support.')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Demonstrate which variables are on what device.')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

################################################
################# handling errors!##############
################################################
if not os.path.isabs(FLAGS.train_path):
    raise ValueError('You must assign absolute path for --train_path')

if not os.path.isabs(FLAGS.checkpoint_path):
    raise ValueError('You must assign absolute path for --checkpoint_path')

# 注意，需要提前下载MNIST数据集
# http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
# http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
# http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
# http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
# 数据集下载好之后存在项目文件目录下，进行引用
mnist = input_data.read_data_sets(".\MNIST_data/", reshape=True, one_hot=False)

### 数据处理
data = {}
data['train/image'] = mnist.train.images
data['train/label'] = mnist.train.labels
data['test/image'] = mnist.test.images
data['test/label'] = mnist.test.labels
# print(data['train/image'][0:3])

# 只获取训练集中标记有0和1的样本
index_list_train = []
for sample_index in range(data['train/label'].shape[0]):
    # 拿到标签
    label = data['train/label'][sample_index]
    if label == 1 or label == 0:
        index_list_train.append(sample_index)

data['train/image'] = mnist.train.images[index_list_train]
data['train/label'] = mnist.train.labels[index_list_train]

# 同样，只获取测试集中标记有0和1的样本
index_list_test = []
for sample_index in range(data['test/label'].shape[0]):
    # 拿到标签
    label = data['test/label'][sample_index]
    if label == 1 or label == 0:
        index_list_test.append(sample_index)

data['test/image'] = mnist.test.images[index_list_test]
data['test/label'] = mnist.test.labels[index_list_test]

# 训练维度
dimensionality_train = data['train/image'].shape
# 维度
num_train_samples = dimensionality_train[0]
num_features = dimensionality_train[1]

### 逻辑回归实现
# 逻辑回归的结构：是一个简单的通过完全连接的层来馈送转发输入特征，
# 其中最后一层仅具有两个类。
### 设定图示（Tensorflow的图）
graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)

    #步长策略
    decay_steps = int(num_train_samples / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               FLAGS.learning_rate_decay_factor,
                                               staircase=True,
                                               name='exponential_decay_learning_rate')


    #设定占位符
    image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
    label_place=tf.placeholder(tf.float32, shape=([None, ]), name='gt')
    label_one_hot=tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
    dropout_param=tf.placeholder(tf.float32)

    #模型，损失函数，正确率
    #一个简单的完全连接两个类和一个softmax相当于Logistic回归
    logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs=FLAGS.num_classes, scope='fc')
    #损失
    with tf.name_scope('loss'):
        loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))
    #正确率
    #评估模型
    prediction_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
    #正确率计算
    accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))

    #训练操作
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    with tf.name_scope('train_op'):
        gradients_and_variables = optimizer.compute_gradients(loss_tensor)
        train_op = optimizer.apply_gradients(gradients_and_variables, global_step=global_step)

    #运行会话操作
    session_conf=tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_palcement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess=tf.Session(graph=graph,config=session_conf)
    with sess.as_default():
        saver=tf.train.Saver()
        #初始化所有变量
        sess.run(tf.global_variables_initializer())
        #已经完成拟合的文件
        #TensorFlow的Saver类是通过操作checkpoint文件来实现对变量（Variable）的存储和恢复。checkpoint文件是二进制的文件，存放着按照固定格式存储的“变量名-Tensor值”map对
        #checkpoint文件可以直接用记事本打开，里面存放的是最新模型的path和所有模型的path在
        checkpoint_prefix='model'
        #如果完成好的调谐，模型将被重新保存
        if FLAGS.fin_tuning:
            saver.restore(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
            print("Model restored for fine-tuning...")
        '''
        [batch、batch size、epoch](https://blog.csdn.net/menc15/article/details/71628019)
        batch: batch是批。深度学习每一次参数的更新所需要损失函数并不是由一个{data：label}获得的，而是由一组数据加权得到的，这一组数据的数量就是[batch size]。
        batch的思想，至少有两个作用，一是更好的处理非凸的损失函数，非凸的情况下， 全样本就算工程上算的动， 也会卡在局部优上， 批表示了全样本的部分抽样实现， 相当于人为引入修正梯度上的采样噪声，使“一路不通找别路”更有可能搜索最优值；二是合理利用内存容量。
        如果数据集较小，可以采用全数据集（Full batch learning）的形式，这样有两个显然的好处：1.由全数据集计算的梯度能够更好的代表样本总体，从而更准确的朝向极值所在的方向；2.不同权重的梯度值差别很大，因此选取一个全局的学习率会比较困难（？）
        batch size最大是样本总数N，此时就是Full batch learning；最小是1，即每次只训练一个样本，这就是在线学习（Online Learning）。当我们分批学习时，每次使用过全部训练数据完成一次Forword运算以及一次BP运算，成为完成了一次epoch。
        '''
        #运行训练和循环
        test_accuracy=0
        for epoch in range(FLAGS.num_epochs):
            total_batch_training=int(data['train/image'].shape[0]/FLAGS.batch_size)

            for batch_num in range(total_batch_training):
                start_idx=batch_num*FLAGS.batch_size
                end_idx=(batch_num+1)*FLAGS.batch_size
                #使用batch数据进行拟合
                train_batch_data, train_batch_label = data['train/image'][start_idx:end_idx], data['train/label'][start_idx:end_idx]
                #运行优化操作（反向传播），计算损失和正确率
                batch_loss, _, training_step=sess.run(
                    [loss_tensor,train_op,global_step],
                    feed_dict={image_place:train_batch_data,label_place:train_batch_label,dropout_param:0.5}
                )
            print('Epoch'+str(epoch+1)+',Training loss='+'{:.5f}'.format(batch_loss))

        #保存模型
        if not os.path.exists(FLAGS.checkpoint_path):
            os.makedirs(FLAGS.checkpoint_path)

        save_path=saver.save(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
        print("Model saved in file: %s" % save_path)

        # The prefix for checkpoint files
        checkpoint_prefix = 'model'

        #更新权重
        saver.restore(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
        print("Model restored...")

        #评估模型
        test_accuracy = 100 * sess.run(accuracy, feed_dict={
            image_place: data['test/image'],
            label_place: data['test/label'],
            dropout_param: 1.})

        print("Final Test Accuracy is %% %.2f" % test_accuracy)
