'''
**说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。**
进入第二部分深度学习  

第十章人工神经网络介绍
人工神经网络是深度学习的核心。具有通用性、强大性和可扩展性，使得它们能够很好地解决大型和高度复杂的机器学习任务。  
在本章中，介绍人工神经网络，从快速游览的第一个ANN架构开始。然后提出多层感知器（MLP），并基于TensorFlow实现MNIST数字分类问题。

### 1.神经元的逻辑计算
人工神经元有一个或更多的二进制（ON/OFF）输入和一个二进制输出。当超过一定数量的输入是激活时，人工神经元会激活其输出。  
McCulloch 和 Pitts 表明，即使用这样一个简化的模型，也有可能建立一个人工神经元网络来计算任何你想要的逻辑命题。
- 左边的第一个网络仅仅是确认函数：如果神经元 A 被激活，那么神经元 C 也被激活（因为它接收来自神经元 A 的两个输入信号），但是如果神经元 A 关闭，那么神经元 C 也关闭。
- 第二网络执行逻辑 AND：有在激活神经元 A 和 B（单个输入信号不足以激活神经元 C）时才激活神经元 C。
- 第三网络执行逻辑 OR：如果神经元 A 或神经元 B 被激活（或两者），神经元 C 被激活。
- 如果我们假设输入连接可以抑制神经元的活动（生物神经元是这样的情况），那么第四个网络计算一个稍微复杂的逻辑命题：如果神经元 B 关闭，只有当神经元A是激活的，神经元 C 才被激活。如果神经元 A 始终是激活的，那么你得到一个逻辑 NOT：神经元 C 在神经元 B 关闭时是激活的，反之亦然。

### 2.感知器
感知器是最简单的人工神经网络结构之一，是基于一种稍微不同的人工神经元，称为**线性阈值单元（LTU）**：输入和输出现在是数字（而不是二进制开/关值），并且每个输入连接都与权重相连。
单一的 LTU 可被用作简单线性二元分类。它计算输入的线性组合，如果结果超过阈值，它输出正类或者输出负类（就像一个逻辑回归分类或线性 SVM）。  
例如，你可以使用单一的 LTU 基于花瓣长度和宽度去分类鸢尾花（也可添加额外的偏置特征x0=1）。训练一个 LTU 意味着去寻找合适的W0和W1值。
感知器简单地由一层 LTU 组成，每个神经元连接到所有输入。这些连接通常用特殊的被称为输入神经元的传递神经元来表示：它们只输出它们所输入的任何输入。此外，通常添加额外偏置特征（X0=1）。这种偏置特性通常用一种称为偏置神经元的特殊类型的神经元来表示，它总是输出 1。

### 3.多层感知器与反向传播
**注意，这里的神经网络图是一个从下到上的顺序，和平时从左到右有区别**
在LTU的基础上，多层感知器由一个输入层、一个或多个隐藏层的LTU和一个最终LTU输出层构成。MLP除了输出层之外的每一层包括偏置神经元，并且全连接到下一层。当人工神经网络有两个或多个隐含层时，称为深度神经网络（DNN）。  
如何训练MLP？目前使用的是反向传播训练算法（BP算法）。  
对于每个训练实例，反向传播算法首先进行预测（前向），测量误差，然后反向遍历每个层来测量每个连接（反向传递）的误差贡献，最后稍微调整连接器权值以减少误差（梯度下降步长）。
MLP 通常用于分类，每个输出对应于不同的二进制类（例如，垃圾邮件/正常邮件，紧急/非紧急，等等）。当类是多类的（例如，0 到 9 的数字图像分类）时，输出层通常通过用共享的 softmax 函数替换单独的激活函数来修改。每个神经元的输出对应于相应类的估计概率。注意，信号只在一个方向上流动（从输入到输出），因此这种结构是前馈神经网络（FNN）的一个例子。
生物神经元是用 sigmoid（S 型）激活函数活动的，因此研究人员在很长一段时间内坚持 sigmoid 函数。但事实证明，Relu 激活函数通常在 ANN 工作得更好。这是生物研究误导的例子之一。



**说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。**
进入第二部分深度学习  

第十章人工神经网络介绍
人工神经网络是深度学习的核心。具有通用性、强大性和可扩展性，使得它们能够很好地解决大型和高度复杂的机器学习任务。  
在本章中，介绍人工神经网络，从快速游览的第一个ANN架构开始。然后提出多层感知器（MLP），并基于TensorFlow实现MNIST数字分类问题。
### 4.用 TensorFlow 高级 API 训练 MLP
Tensorflow高级API TF.learn，与Sklearn相似，教程在这里使用DNNClassifier构建了一个深度神经网络。  
输出层使用softmax输出概率。
使用的数据集是mnist，但是这里不推荐使用tf.examples.tutorials.mnist，教程改用tf.keras.datasets.mnist。  
另外，tf.contrib.learn改进为tf.estimators 和 tf.feature_columns，并且改进不小。   
运行结果来看，正确率为98%，测试一下，和之前一样，会对10个数值进行预测。

~~~python
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data

### tensorflow警告记录，可以避免在运行文件时出现红色警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# mnist = input_data.read_data_sets(".\MNIST_data/", reshape=True, one_hot=True)
# X_train = mnist.train.images
# X_test = mnist.test.images
# y_train = mnist.train.labels
# y_test = mnist.test.labels
print(y_train[3])

X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

print(y_valid[3])
feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
# 下面的代码训练两个隐藏层的 DNN（一个具有 300 个神经元，另一个具有 100 个神经元）和一个具有 10 个神经元的 SOFTMax 输出层
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                     feature_columns=feature_cols)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True)
dnn_clf.train(input_fn=input_fn)


test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results = dnn_clf.evaluate(input_fn=test_input_fn)
print(eval_results) #{'accuracy': 0.98, 'average_loss': 0.10057722, 'loss': 12.731294, 'global_step': 44000}


y_pred_iter = dnn_clf.predict(input_fn=test_input_fn)
y_pred = list(y_pred_iter)
print(y_pred[0])

# {'logits': 
# array([ -7.9471016,   5.2029753,  -3.5739818,   2.1981955, -11.298469 ,
#         -7.5786734, -21.987339 ,  25.491554 ,  -3.924615 ,   2.3928545],
#       dtype=float32), 
# 'probabilities': 
# array([3.0045281e-15, 1.5444808e-09, 2.3823079e-13, 7.6528506e-11,
#        1.0526783e-16, 4.3429238e-15, 2.3998198e-21, 1.0000000e+00,
#        1.6777237e-13, 9.2974156e-11], dtype=float32), 
# 'class_ids': 
# array([7], dtype=int64), 'classes': array([b'7'], dtype=object)}
~~~

DNNClassifier基于 Relu 激活函数创建所有神经元层（我们可以通过设置超参数activation_fn来改变激活函数）。输出层基于 SoftMax 函数，损失函数是交叉熵。  
其实个人认为可以好好看一下TF.Learn，毕竟现在pytorch出来了，之后更高的封装会让训练变得更简便。
### 5.使用普通 TensorFlow 训练 DNN
这一节教程中使用了一个较低级别的api来构建DNN，可以过一下。

~~~python



~~~

### 5.微调神经网络超参数
神经网络的灵活性也是其主要缺点之一：有很多超参数要进行调整。 不仅可以使用任何可想象的网络拓扑（如何神经元互连），而且即使在简单的 MLP 中，您可以更改层数，每层神经元数，每层使用的激活函数类型，权重初始化逻辑等等。 如何确定什么组合的超参数是最适合你的任务？

- 可以使用具有交叉验证的网格搜索来查找正确的超参数。但是由于要调整许多超参数，并且由于在大型数据集上训练神经网络需要很多时间。
- 使用随机搜索要好得多。
- 使用诸如 Oscar 之类的工具，它可以实现更复杂的算法，以帮助您快速找到一组好的超参数.

### 6.隐藏层数量
实际上已经表明一个隐藏层的 MLP 可以建模甚至最复杂的功能，只要它具有足够的神经元。 但是深层网络具有比浅层网络更高的参数效率：可以使用比浅网格更少的神经元来建模复杂的函数，使得训练更快。
- 对于许多问题，可以从一个或两个隐藏层开始，它可以正常工作（例如，您可以使用只有一个隐藏层和几百个神经元，在 MNIST 数据集上容易达到 97% 以上的准确度使用两个具有相同总神经元数量的隐藏层，在大致相同的训练时间量中精确度为 98%）。
- 对于更复杂的问题，可以逐渐增加隐藏层的数量，直到覆盖训练集。
- 对于非常复杂的任务，如大型图像分类或语音识别，通常需要具有数十个层（或甚至数百个但不完全相连的网络）的网络，并且需要大量的训练数据。但是，您将很少从头开始训练这样的网络：重用预先训练的最先进的网络执行类似任务的部分更为常见。训练将会更快，需要更少的数据。

### 7.每层隐藏层的神经元数量
输入和输出层中神经元的数量由任务需要的输入和输出类型决定。例如，MNIST 任务需要28*28 = 784个输入神经元和 10 个输出神经元。  
对于隐藏的层次来说，通常的做法是将其设置为形成一个**漏斗**，每个层面上的神经元越来越少，原因在于许多低级别功能可以合并成更少的高级功能。例如，MNIST 的典型神经网络可能具有两个隐藏层，第一个具有 300 个神经元，第二个具有 100 个。  
但是，这种做法现在并不常见，您可以为所有隐藏层使用相同的大小 - 例如，所有隐藏的层与 150 个神经元：这样只用调整一次超参数而不是每层都需要调整（因为如果每层一样，比如 150，之后调就每层都调成 160）。就像层数一样，您可以尝试逐渐增加神经元的数量，直到网络开始过度拟合。一般来说，通过增加每层的神经元数量，可以增加层数，从而获得更多的消耗。

最后教程中说“找到完美的神经元数量仍然是黑色的艺术。”

一个更简单的方法是选择一个具有比实际需要的更多层次和神经元的模型，然后使用早期停止来防止它过度拟合（以及其他正则化技术，特别是 drop out，我们将在第 11 章中看到）。 这被称为“拉伸裤”的方法：而不是浪费时间寻找完美匹配您的大小的裤子，只需使用大型伸缩裤，缩小到合适的尺寸。

### 9.激活函数
在大多数情况下，您可以在隐藏层中使用 ReLU 激活函数（或其中一个变体）。 与其他激活函数相比，计算速度要快一些，而梯度下降在局部最高点上并不会被卡住，因为它不会对大的输入值饱和（与逻辑函数或双曲正切函数相反, 他们容易在 1 饱和)

对于输出层，softmax 激活函数通常是分类任务的良好选择（当这些类是互斥的时）。 对于回归任务，您完全可以不使用激活函数。


'''
# import tensorflow as tf
# import numpy as np
# import os
# from sklearn.metrics import accuracy_score
# from tensorflow.examples.tutorials.mnist import input_data
#
# ### tensorflow警告记录，可以避免在运行文件时出现红色警告
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
#
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
#
# # mnist = input_data.read_data_sets(".\MNIST_data/", reshape=True, one_hot=True)
# # X_train = mnist.train.images
# # X_test = mnist.test.images
# # y_train = mnist.train.labels
# # y_test = mnist.test.labels
# print(y_train[3])
#
# X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
# X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)
#
# X_valid, X_train = X_train[:5000], X_train[5000:]
# y_valid, y_train = y_train[:5000], y_train[5000:]
#
# print(y_valid[3])
# feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
# # 下面的代码训练两个隐藏层的 DNN（一个具有 300 个神经元，另一个具有 100 个神经元）和一个具有 10 个神经元的 SOFTMax 输出层
# dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10,
#                                      feature_columns=feature_cols)
#
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True)
# dnn_clf.train(input_fn=input_fn)
#
#
# test_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"X": X_test}, y=y_test, shuffle=False)
# eval_results = dnn_clf.evaluate(input_fn=test_input_fn)
# print(eval_results) #{'accuracy': 0.98, 'average_loss': 0.10057722, 'loss': 12.731294, 'global_step': 44000}
#
# y_pred_iter = dnn_clf.predict(input_fn=test_input_fn)
# y_pred = list(y_pred_iter)
# print(y_pred[0])
from tensorflow.contrib.labeled_tensor import shuffle_batch

'''
{'logits': 
array([ -7.9471016,   5.2029753,  -3.5739818,   2.1981955, -11.298469 ,
        -7.5786734, -21.987339 ,  25.491554 ,  -3.924615 ,   2.3928545],
      dtype=float32), 
'probabilities': 
array([3.0045281e-15, 1.5444808e-09, 2.3823079e-13, 7.6528506e-11,
       1.0526783e-16, 4.3429238e-15, 2.3998198e-21, 1.0000000e+00,
       1.6777237e-13, 9.2974156e-11], dtype=float32), 
'class_ids': 
array([7], dtype=int64), 'classes': array([b'7'], dtype=object)}
'''

# y_pred = list(dnn_clf.predict(X_test))
# accuracy=accuracy_score(y_test, y_pred)
# print(accuracy)

import tensorflow as tf
import numpy as np
import os
### tensorflow警告记录，可以避免在运行文件时出现红色警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# 设置基本参数，输入的单体28*28，第一层300，第二层100，输出10（就是0-9十个数值的概率）
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

'''
接下来，与第 9 章一样，使用占位符节点来表示训练数据和目标。X的形状仅有部分被定义。 我们知道它将是一个 2D 张量（即一个矩阵），沿着第一个维度的实例和第二个维度的特征，我们知道特征的数量将是28×28（每像素一个特征） 但是我们不知道每个训练批次将包含多少个实例。 所以X的形状是(None, n_inputs)。 同样，我们知道y将是一个 1D 张量，每个实例有一个入口，但是我们再次不知道在这一点上训练批次的大小，所以形状(None)。
'''

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

'''
创建实际的神经网络，X占位符是输入层，同时创建两个隐藏层和一个输出层，输出层激活参数调用Relu。
'''


def neural_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        # 使用具有标准差为2/√n的截断的正态（高斯）分布(使用截断的正态分布而不是常规正态分布确保不会有任何大的权重，这可能会减慢训练。).使用这个特定的标准差有助于算法的收敛速度更快（我们将在第11章中进一步讨论这一点），这是对神经网络的微小调整之一，对它们的效率产生了巨大的影响）
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="W")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z


with tf.name_scope("dnn"):
    hidden1 = neural_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neural_layer(X, n_hidden2, "hidden2", activation="relu")
    logits = neural_layer(hidden2, n_outputs, "outputs")

'''
构建和训练损失函数。  
使用交叉熵，sparse_softmax_cross_entropy_with_logits()：根据“logit”计算交叉熵（即，在通过 softmax 激活函数之前的网络输出），并且期望以 0 到 -1 数量的整数形式的标签（在我们的例子中，从 0 到 9）。   
这将生成一个包含每个实例的交叉熵的 1D 张量。   
然后，我们可以使用 TensorFlow 的reduce_mean()函数来计算所有实例的平均交叉熵。

这个sparse_softmax_cross_entropy_with_logits()函数等同于应用 SOFTMAX 激活函数，然后计算交叉熵，  
但它更高效，能够照顾边界情况（比如 logits=0），  
还有称为softmax_cross_entropy_with_logits()的另一个函数，该函数在标签独热形式（而不是整数 0 至类的数目减 1）。

'''
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")

'''
在构建了神经网络模型、损失函数之后，需要定义一个GradientDescentOptimizer来调整模型参数以最小化损失函数。
'''
learning_rate=0.01
with tf.name_scope("train"):
    optimazer=tf.train.GradientDescentOptimizer(learning_rate)
    training_op=optimazer.minimize(loss)

'''
建模阶段的最后一个重要步骤是指定如何评估模型。 我们将简单地将精度用作我们的绩效指标。  
使用in_top_k函数。需要将这些布尔值转换为浮点数，然后计算平均值。 这将给我们网络的整体准确性
'''
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 40
batch_size = 50

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
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

    save_path = saver.save(sess, "./tensorflow02.ckpt")

with tf.Session() as sess:
    saver.restore(sess, "./tensorflow02.ckpt") # or better, use save_path
    X_new_scaled = X_test[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)

print("Predicted classes:", y_pred)
print("Actual classes:   ", y_test[:20])