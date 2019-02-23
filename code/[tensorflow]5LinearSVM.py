'''
## 线性支持向量机（Linear SVM）
在教程中，我们将创建一个线性SVM来分离数据。用于此代码的数据是线性可分的。
### 线性支持向量机
关于支持向量机，建议看一下这个博文[支持向量机通俗导论（理解SVM的三层境界）](https://blog.csdn.net/v_JULY_v/article/details/7624837)
SVM的应用领域很广，分类、回归、密度估计、聚类等，但我觉得最成功的还是在分类这一块。
用于分类问题时，SVM可供选择的参数并不多，**惩罚参数C，核函数及其参数**选择。对于一个应用，是选择线性核，还是多项式核，还是高斯核？还是有一些规则的。
在实际应用中，多数情况是特征维数非常高。如OCR中的汉字识别，提取8方向梯度直方图特征，归一化的字符被等分成8*8的网格，每个网格计算出长度为8的方向直方图，特征维数是8*8*8 = 512维。在这样的高维空间中，想把两个字符类分开，用线性SVM是轻而易举的事，当然用其它核也能把它们分开。那为什么要选择线性核，因为，线性核有两个非常大的优点：
1. 预测函数简单f(x) = w’*x+b，分类速度快。对于类别多的问题，分类速度的确需要考虑到，线性分类器的w可以事先计算出来，而非线性分类器在高维空间时支持向量数会非常多，分类速度远低于线性分类器。
2. 线性SVM的推广性有保证，而非线性如高斯核有可能过学习。再举个例子，**基于人脸的性别识别，即给定人脸图像，判断这个人是男还是女**。我们提取了3700多维的特征，用线性SVM就能在测试集上达到96%的识别正确率。因此，线性SVM是实际应用最多的，实用价值最大的。
如果在你的应用中，特征维数特别低，样本数远超过特征维数，则选用非线性核如高斯核是比较合理的。
如果两类有较多重叠，则非线性SVM的支持向量特别多，选择稀疏的非线性SVM会是一个更好的方案，支持向量少分类速度更快，如下图。
'''
### 引入相关库
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
import random
import sys

'''
### 关键的FLAGS，在用命令行执行程序时，需要传的关键参数
batch_size:32
num_steps:500
C_param:0.1
Reg_param:1.0
delta:1.0
initial_learning_rate:0.1
'''

tf.app.flags.DEFINE_integer('batch_size', 32,
                            'Number of samples per batch.')
tf.app.flags.DEFINE_integer('num_steps', 1000,
                            'Number of steps for training.')
tf.app.flags.DEFINE_boolean('is_evaluation', True,
                            'Whether or not the model should be evaluated.')
tf.app.flags.DEFINE_float(
    'C_param', 0.1,
    'penalty parameter of the error term.')
tf.app.flags.DEFINE_float(
    'Reg_param', 1.0,
    'penalty parameter of the error term.')
tf.app.flags.DEFINE_float(
    'delta', 1.0,
    'The parameter set for margin.')
tf.app.flags.DEFINE_float(
    'initial_learning_rate', 0.1,
    'The initial learning rate for optimization.')

FLAGS = tf.app.flags.FLAGS

'''
### 所需方法
loss_fn
inference_fn
next_batch_fn
这里用到一些tensorflow常用算数操作
可以看一下这两个博客[TensorFlow变量常用操作](https://blog.csdn.net/hongxue8888/article/details/79903870)；[tensorflow之算术运算符](https://blog.csdn.net/u013230189/article/details/82721401)
这里出现的几个api注解
- substract：减法操作。
- multiply: 乘法操作。
- transpose: 转置操作。tf.transpose(input, [dimension_1, dimenaion_2,..,dimension_n]):这个函数主要适用于交换输入张量的不同维度用的，如果输入张量是二维，就相当是转置。dimension_n是整数，如果张量是三维，就是用0,1,2来表示。这个列表里的每个数对应相应的维度。如果是[2,1,0]，就把输入张量的第三维度和第一维度交换。
'''


def loss_fn(W, b, x_data, y_target):
    logits = tf.subtract(tf.matmul(x_data, W), b)
    norm_term = tf.divide(tf.reduce_sum(tf.multiply(tf.transpose(W), W)), 2)
    classification_loss = tf.reduce_mean(tf.maximum(0., tf.subtract(FLAGS.delta, tf.multiply(logits, y_target))))
    total_loss = tf.add(tf.multiply(FLAGS.C_param, classification_loss), tf.multiply(FLAGS.Reg_param, norm_term))
    return total_loss


def inference_fn(W, b, x_data, y_target):
    prediction = tf.sign(tf.subtract(tf.matmul(x_data, W), b))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
    return accuracy


# 参数意思分别 是从a中以概率P，随机选择num_samples个, p没有指定的时候相当于是一致的分布
# replace 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了。
def next_batch_fn(x_train, y_train, num_samples=FLAGS.batch_size):
    index = np.random.choice(a=len(x_train), size=num_samples)
    x_batch = x_train[index]
    y_batch = np.transpose([y_train[index]])
    return x_batch, y_batch


'''
### 数据操作，操作iris鲜花数据库
只使用前两个特征，[:,:2]获取所有行，每一行前两列
'''
iris = datasets.load_iris()
X = iris.data[:, :2]
# 获取label数据
y = np.array([1 if label == 0 else -1 for label in iris.target])
my_randoms = np.random.choice(X.shape[0], X.shape[0], replace=False)
# 对数据进行切割，其实可以使用sklearn的函数进行操作
train_indices = my_randoms[0:int(0.5 * X.shape[0])]
test_indices = my_randoms[int(0.5 * X.shape[0]):]

x_train = X[train_indices]
y_train = y[train_indices]
x_test = X[test_indices]
y_test = y[test_indices]

'''
### 生成占位符
'''
x_data = tf.placeholder(shape=[None, X.shape[1]], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
W = tf.Variable(tf.random_normal(shape=[X.shape[1], 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

'''
### 计算损失函数和正确率
'''
total_loss = loss_fn(W, b, x_data, y_target)
accuracy = inference_fn(W, b, x_data, y_target)
### 训练优化模型
train_op = tf.train.GradientDescentOptimizer(FLAGS.initial_learning_rate).minimize(total_loss)

### 变量初始化
# tf.Session().run(tf.initialize_all_variables())
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

'''
### 训练线性支持向量机
'''
for step_idx in range(FLAGS.num_steps):
    x_batch, y_batch = next_batch_fn(x_train, y_train, num_samples=FLAGS.batch_size)
    sess.run(train_op, feed_dict={x_data: x_batch, y_target: y_batch})
    loss_step = sess.run(total_loss, feed_dict={x_data: x_batch, y_target: y_batch})
    train_acc_step = sess.run(accuracy, feed_dict={x_data: x_train, y_target: np.transpose([y_train])})
    test_acc_step = sess.run(accuracy, feed_dict={x_data: x_test, y_target: np.transpose([y_test])})

    if step_idx % 100 == 0:
        print('Step #%d, training accuracy= %% %.2f, testing accuracy= %% %.2f ' % (
            step_idx, float(100 * train_acc_step), float(100 * test_acc_step)))

if FLAGS.is_evaluation:
    [[w1], [w2]] = sess.run(W)
    [[bias]] = sess.run(b)
    x_line = [data[1] for data in X]

    # Find the separator line.
    line = []
    line = [-w2 / w1 * i + bias / w1 for i in x_line]

    # coor_pos_list = [positive_X, positive_y]
    # coor_neg_list = [negative_X, negative_y]

    for index, data in enumerate(X):
        if y[index] == 1:
            positive_X = data[1]
            positive_y = data[0]
        elif y[index] == -1:
            negative_X = data[1]
            negative_y = data[0]
        else:
            sys.exit("Invalid label!")

plt.plot(positive_X, positive_y, '+', label='Positive')
plt.plot(negative_X, negative_y, 'o', label='Negative')
plt.plot(x_line, line, 'r-', label='Separator', linewidth=3)
plt.legend(loc='best')
plt.title('Linear SVM')
plt.show()
