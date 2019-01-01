'''
第三章的作业
1. 尝试在 MNIST 数据集上建立一个分类器，使它在测试集上的精度超过 97%。提示：KNeighborsClassifier非常适合这个任务。你只需要找出一个好的超参数值（试一下对权重和超参数n_neighbors进行网格搜索）。
2. 写一个函数可以是 MNIST 中的图像任意方向移动（上下左右）一个像素。然后，对训练集上的每张图片，复制四个移动后的副本（每个方向一个副本），把它们加到训练集当中去。最后在扩展后的训练集上训练你最好的模型，并且在测试集上测量它的精度。你应该会观察到你的模型会有更好的表现。这种人工扩大训练集的方法叫做数据增强，或者训练集扩张。
3. 拿 Titanic 数据集去捣鼓一番。开始这个项目有一个很棒的平台：Kaggle！
4. 建立一个垃圾邮件分类器（这是一个更有挑战性的练习）：
- 下载垃圾邮件和非垃圾邮件的样例数据。地址是Apache SpamAssassin 的公共数据集
- 解压这些数据集，并且熟悉它的数据格式。
- 将数据集分成训练集和测试集
- 写一个数据准备的流水线，将每一封邮件转换为特征向量。你的流水线应该将一封邮件转换为一个稀疏向量，对于所有可能的词，这个向量标志哪个词出现了，哪个词没有出现。举例子，如果所有邮件只包含了"Hello","How","are", "you"这四个词，那么一封邮件（内容是："Hello you Hello Hello you"）将会被转换为向量[1, 0, 0, 1](意思是："Hello"出现，"How"不出现，"are"不出现，"you"出现)，或者[3, 0, 0, 2]，如果你想数出每个单词出现的次数。
- 你也许想给你的流水线增加超参数，控制是否剥过邮件头、将邮件转换为小写、去除标点符号、将所有 URL 替换成"URL"，将所有数字替换成"NUMBER"，或者甚至提取词干（比如，截断词尾。有现成的 Python 库可以做到这点）。
- 然后 尝试几个不同的分类器，看看你可否建立一个很棒的垃圾邮件分类器，同时有着高召回率和高准确率。
在这里完成第二个
'''
'''
- 写一个函数可以是 MNIST 中的图像任意方向移动（上下左右）一个像素。
- 然后，对训练集上的每张图片，复制四个移动后的副本（每个方向一个副本），把它们加到训练集当中去。
- 最后在扩展后的训练集上训练你最好的模型，并且在测试集上测量它的精度。你应该会观察到你的模型会有更好的表现。
- 这种人工扩大训练集的方法叫做数据增强，或者训练集扩张。

'''
### 1. 引入库
from scipy.ndimage.interpolation import shift
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import precision_recall_curve, accuracy_score

#### tensorflow警告记录，可以避免在运行文件时出现红色警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


### 2. 引入数据并打乱训练集
def InputData():
    global X_train, y_train
    mnist = input_data.read_data_sets(".\MNIST_data/", reshape=True, one_hot=False)
    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels
    shuffle_index_train = np.random.permutation(len(X_train))
    shuffle_index_test = np.random.permutation(len(X_test))
    X_train, y_train = X_train[shuffle_index_train], y_train[shuffle_index_train]
    X_test, y_test = X_test[shuffle_index_test], y_test[shuffle_index_test]
    return X_train, y_train, X_test, y_test


### 3.处理图片
def shift_image(image, dx, dy):
    # mnist数据集是28*28。
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


# def img(test_data):
#     img0 = test_data.reshape(28, 28)
#     fig = plt.figure(figsize=(10, 10))
#     ax0 = fig.add_subplot(111)
#     ax0.imshow(img0)
#     plt.ioff()
#     plt.show()

X_train, y_train, X_test, y_test = InputData()
image = X_train[100]
### 4.调用之前的shift_image函数移动像素点
shifted_image_down = shift_image(image, 0, 5)
shifted_image_left = shift_image(image, -5, 0)

#### 打印图片查看效果
plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title("Original", fontsize=14)
plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(132)
plt.title("Shifted down", fontsize=14)
plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(133)
plt.title("Shifted left", fontsize=14)
plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.show()

X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]
for dx, dy in (1, 0), (-1, 0), (0, 1), (0, -1):
    # zip()是Python的一个内建函数，它接受一系列可迭代的对象作为参数，将对象中对应的元素打包成一个个tuple（元组），然后返回由这些tuples组成的list（列表）。
    # 若传入参数的长度不等，则返回list的长度和参数中长度最短的对象相同。
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

# param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
# grid_search = GridSearchCV()
# grid_search.best_params_={'n_neighbors': 4, 'weights': 'distance'}
# **在生成KNN_clf的时候需要作业1中训练的grid_search.best_params，幸好当时保存了，要不又要重跑一次，在这里填进去**
KNN_clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
KNN_clf.fit(X_train_augmented, y_train_augmented)
y_pred = KNN_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)  # =>0.9761
