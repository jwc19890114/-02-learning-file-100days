'''
今天进入第八天，昨天使用sklearn中打包的**KNeighborsClassifier()**完成KNN分类器，今天使用Tensorflow来重写。
## 欧氏距离
$d(x,y)=\sqrt{\Sigma_{n=k}^1(x_k-y_k)^2}$  
## 算法思路
- 计算待分类样本和样本空间中已标记的样本的欧氏距离
- 取得最短距离的K个点并对K个点所属标签进行计数。
## 算法优缺点
- 优点
算法简单有效
- 缺点
一方面计算量大。当训练集比较大的时候，每一个样本分类都要计算与所有的已标记样本的距离。**解决方法是事先对已知样本点进行剪辑，事先去除对分类作用不大的样本（例如在样本空间进行划分区域）**。  
另一方面是当已标记样本是不平衡，分类会向占样本多数的类倾斜。**解决方案是引进权重**。

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials import mnist

mnist_image = mnist.input_data.read_data_sets('.\MNIST_data/', reshape=True, one_hot=True)
pixels, real_values = mnist_image.train.next_batch(100)

# 获取前10个图片，这段代码用来显示获取到的数据
# n=5
# image=pixels[n,:]
# image=np.reshape(image, [28,28])
# plt.imshow(image)
# plt.show()

traindata, trainlabel = mnist_image.train.next_batch(100)
testdata, testlabel = mnist_image.test.next_batch(100)

traindata_tensor = tf.placeholder(shape=[None, 784], dtype=tf.float32)
testdata_tensor = tf.placeholder(shape=[784], dtype=tf.float32)

# 几个基本运算
# tf.abs取绝对值
# tf.negative取负，y=-x
# 安照reduction_indices指定的轴进行求和
# tf.arg_min()返回张量维度上最小值的索引

distance = tf.reduce_sum(tf.abs(tf.add(traindata_tensor, tf.negative(testdata_tensor))), reduction_indices=1)
pred = tf.arg_min(distance, 0)
test_num = 100
accuracy = 1
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(test_num):
        idx = sess.run(pred, feed_dict={traindata_tensor: traindata, testdata_tensor: testdata[i]})
        print('test No.%d,the real label %d, the predict label %d' % (
        i, np.argmax(testlabel[i]), np.argmax(trainlabel[idx])))
        if np.argmax(testlabel[i]) == np.argmax(trainlabel[idx]):
            accuracy += 1
    print("result:%f" % (1.0 * accuracy / test_num))
