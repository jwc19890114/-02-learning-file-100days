'''
***说明：本文依据《Sklearn 与 TensorFlow 机器学习实用指南》完成，所有版权和解释权均归作者和翻译成员所有，我只是搬运和做注解。***
在第一章我们提到过最常用的监督学习任务是回归（用于预测某个值）和分类（预测某个类别）。  
在第二章我们探索了一个回归任务：预测房价。我们使用了多种算法，诸如线性回归，决策树，和随机森林（这个将会在后面的章节更详细地讨论）。  
在第三章，作者将转到分类任务上。
'''

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix,precision_score, recall_score,f1_score,precision_recall_curve,roc_auc_score,roc_curve
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

### tensorflow警告记录，可以避免在运行文件时出现红色警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

'''
### MNIST
使用 MNIST 这个数据集，它有着 70000 张规格较小的手写数字图片，由美国的高中生和美国人口调查局的职员手写而成。  
这相当于机器学习当中的“Hello World”，人们无论什么时候提出一个新的分类算法，都想知道该算法在这个数据集上的表现如何。  
机器学习的初学者迟早也会处理 MNIST 这个数据集。
在之前的文字中，我们使用Tensorflow加载mnist数据，原文中代码如下，但是使用后会报500的错误。在这里仍使用tf的mnist数据。  
~~~python
mnist = fetch_mldata('MNIST original',data_home=r'.\MNIST_data')
print(mnist)
~~~
下载数据，我这里已经下载完成，放在响应文件夹下，相关下载链接
注意，需要提前下载MNIST数据集
http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
数据集下载好之后存在项目文件目录下，进行引用
'''
mnist = input_data.read_data_sets(".\MNIST_data/", reshape=True, one_hot=False)
print(type(mnist))
'''
#### sklearn数据集中包括三个结构，在anaconda中可以直接通过mnist调用查看，在pycharm中则通过名称调用
- DESCR键描述数据集
- data键存放一个数组，数组的一行表示一个样例，一列表示一个特征
- target键存放一个标签数组
在tensorflow的mnist数据集中，包括train和test两类数据，类似接口调用的方式。
'''
print('Training data size: ', mnist.train.num_examples)
print('Validation data size: ', mnist.validation.num_examples)
print('Test data size: ', mnist.test.num_examples)

'''
#### 绘制相应图片
'''


def img(some_digit):
    img0 = some_digit.reshape(28, 28)
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)
    ax0.imshow(img0)
    plt.ioff()
    plt.show()


'''
#### 打乱训练集
使用numpy的**random.permutation()**打乱训练集。  
这可以保证交叉验证的每一折都是相似（你不会期待某一折缺少某类数字）。  
而且，一些学习算法对训练样例的顺序敏感，当它们在一行当中得到许多相似的样例，这些算法将会表现得非常差。打乱数据集将保证这种情况不会发生。
'''
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels
shuffle_index_train = np.random.permutation(len(X_train))
shuffle_index_test = np.random.permutation(len(X_test))
X_train, y_train = X_train[shuffle_index_train], y_train[shuffle_index_train]
X_test, y_test = X_test[shuffle_index_test], y_test[shuffle_index_test]
'''
#### 训练一个二分类器
在里面只对图片为5的进行分类，然后输出判断值和图片，预测X_test[500]，实际值为9。
'''
# 筛选出只有5的，如果y_train==5，反馈为True，其他均为False，下面y_test_5同理
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# 使用随机梯度下降，random_state取相同的值确保每次输出结果相同
sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_5)
some_digit = X_test[500]
# pred = sgd_clf.predict([some_digit])
# print(pred)
# print(y_test[500])
# img(some_digit) #=>输出为9的图片
'''
#### 评估分类器的性能
评估一个分类器，通常比评估一个回归器更加玄学。所以我们将会花大量的篇幅在这个话题上。
##### 使用交叉验证测量准确性
评估一个模型的好方法是使用交叉验证，就像第二章所做的那样。不用折回第二章看，大致说一下，第二章使用的是K折交叉验证函数***cross_val_score()***  
在这章中，徒手写一个，目的是为了有更高的定制化？。
StratifiedKFold类实现了分层采样，生成的折（fold）包含了各类相应比例的样例。在每一次迭代，上述代码生成分类器的一个克隆版本，在训练折（training folds）的克隆版本上进行训练，在测试折（test folds）上进行预测。然后它计算出被正确预测的数目和输出正确预测的比例。
完成之后，又使用***cross_val_score()***进行了验证。
正确率这么高的原因在于只有10%的图片是数字5，所以你总是猜测某张图片不是5，你也会有90%的可能性是对的。
**这证明了为什么精度通常来说不是一个好的性能度量指标，特别是当你处理有偏差的数据集，比方说其中一些类比其他类频繁得多。**
'''
# skfolds = StratifiedKFold(n_splits=3, random_state=42)
# for train_index, test_index in skfolds.split(X_train, y_train_5):
#     clone_clf = clone(sgd_clf)  # 复制之前的模型
#     # 生成训练折和测试折
#     X_train_folds = X_train[train_index]
#     y_train_folds = (y_train_5[train_index])
#     X_test_fold = X_train[train_index]
#     y_test_fold = (y_train_5[train_index])
#     # 训练
#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     # 预测正确的数目
#     n_correct = sum(y_pred == y_test_fold)
#     # 正确率
#     accuracy = n_correct / len(y_pred)
#     print(accuracy)  # =>最终正确率为0.9721547985927401

#### 使用***cross_val_score()***进行验证，与上面的对比会发现正确率相近
# accuracy_cross = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# print(accuracy_cross)  # =>[0.95920148 0.96901762 0.97016309]
'''
##### 混淆矩阵
对分类器来说，一个好得多的性能评估指标是混淆矩阵。
大体思路是：输出类别A被分类成类别B的次数。
为了计算混淆矩阵，首先你需要有一系列的预测值，这样才能将预测值与真实值做比较。
你或许想在测试集上做预测。但是我们现在先不碰它。
**注意，只有当你处于项目的尾声，当你准备上线一个分类器的时候，你才应该使用测试集**  
应该使用cross_val_predict()函数。就像cross_val_score()，cross_val_predict()也使用K折交叉验证。
它不是返回一个评估分数，而是返回基于每一个测试折做出的一个预测值。
这意味着，对于每一个训练集的样例，你得到一个干净的预测（“干净”是说一个模型在训练过程当中没有用到测试集的数据）。
'''
# y_train_pred =cross_val_predict(sgd_clf, X_train,y_train_5,cv=3)
# 使用 confusion_matrix()函数，你将会得到一个混淆矩阵。传递目标类(y_train_5)和预测类（y_train_pred）给它。
# accuracy_matrix=confusion_matrix(y_train_5, y_train_pred)
# print(accuracy_matrix)
#=>[[49017   996]
# [  848  4139]]
#从上面的结果可以看到，混淆矩阵中的每一行表示一个实际的类, 而每一列表示一个预测的类。
#（真反例，true negatives=TN） 第一行认为“非 5”（反例）中的49017张被正确归类为 “非 5”,
#（假正例，false positives=FP）其余996被错误归类为"是 5" 。
#（假反例，false negatives=FN）第二行认为“是 5” （正例）中的848被错误地归类为“非 5”，
#（真正例，true positives=TP） 其余4139正确分类为 “是 5”类。
#可以发现，计算正确率precison其实就是真正例TP/(真正例TP+假正例FP)
#而召回率recall/敏感度（sensitivity）/真正例率（true positive rate， TPR）是正例被分类器正确探测出的比率，真正例TP/(真正例TP+假反例FN)。

'''
##### 准确率与召回率
Slearn提供函数计算分类器的指标，包括准确率和召回率。
可以发现，准确率为87.7%，召回率为68.1%，也就是说，这个二分类器在处理5的图片时只有87.7%的准确率，同时只能检测出68.1%为5的图片（**这两个区别有点微妙**）。  
引入一个指标F1，是准确率和召回率的调和平均值（普通的平均值平等地看待所有的值，而调和平均会给小的值更大的权重）。  
自然是F1越大说明模型效果越好，如何提高F1，就是要同时提升准确率和召回率。  

'''
# precision=precision_score(y_train_5,y_train_pred)
# recall=recall_score(y_train_5,y_train_pred)
# f1=f1_score(y_train_5, y_train_pred)
# print(precision) #=>0.8765814613994319
# print(recall) #=>0.6807700020052135
# print(f1) #=>0.7782628676470589

'''
##### 准确率/召回率之间的折衷
正如上面所说的，F1代表了这个模型训练结果的好坏，但是在现实中准确率和召回率两个指标往往会有侧重
- 鉴黄师，要高准确率（确保鉴别出来的都是好的，但有可能会把一些不是黄色的剔除掉）
- 罪犯鉴别，要高召回率（确保最大限度纳入所有可能的罪犯，而不是准确率很高（全都识别了），但是召回率低（遗漏掉））
教程开始分析SGDClassifier是如何做分类决策，使用![视图3.3](https://github.com/apachecn/hands-on-ml-zh/blob/dev/images/chapter_3/chapter3.3.jpeg?raw=true)进行解释.  
其实就是阈值在准确率和召回率之间的权衡，Sklearn不让用户直接设置阈值，而是提供了设置决策分数的方法。  
这个决策分数可以用来产生预测。通过调用decision_function()方法返回每一个样例的分数值，然后基于这个分数值，使用你想要的任何阈值做出预测。
~~~python
some_digit_5 = X_train[36000]
y_scores=sgd_clf.decision_function([some_digit_5])
print(y_scores)
threshold1=0 #设置一个0的阈值，返回的肯定与之前一致
y_some_digit_pred1 = (y_scores > threshold1)
print(y_some_digit_pred1)
threshold2=200000 #设置一个200000的阈值，返回预测结果是False，但是实际上图片是5，说明提高了召回率
y_some_digit_pred2 = (y_scores > threshold2)
print(y_some_digit_pred2)
~~~
**如何决定使用哪个阈值**。
- 首先，你需要再次使用cross_val_predict()得到每一个样例的分数值，但是这一次指定返回一个决策分数，而不是预测值。
- 有了这些分数值。对于任何可能的阈值，使用precision_recall_curve(),你都可以计算准确率和召回率。
- 最后，使用 Matplotlib 画出准确率和召回率，这里把准确率和召回率当作是阈值的一个函数。
'''
# y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function")
# precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
# def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
#     plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
#     plt.xlabel("Threshold")
#     plt.legend(loc="upper left")
#     plt.ylim([0, 1])
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()

# def plot_precision_vs_recall(precisions,recalls):
#     plt.plot(precisions, recalls, "g-", label="Recall")
#     plt.xlabel("Recall")
#     plt.ylim([0, 1])
# plot_precision_vs_recall(precisions,recalls)
# plt.show()
'''
可以看到，在召回率在 80% 左右的时候，准确率急剧下降。你可能会想选择在急剧下降之前选择出一个准确率/召回率折衷点。
我们假设决定达到90%的准确率。你查阅第一幅图（放大一些），在1附近找到一个阈值。为了作出预测（目前为止只在训练集上预测），你可以运行以下代码，而不是运行分类器的predict()方法。
'''
# y_train_pred_90 = (y_scores > 1)
# precision_score_90=precision_score(y_train_5, y_train_pred_90)
# print(precision_score_90) #=>0.9169570267131243
'''
##### ROC 曲线
受试者工作特征（ROC）曲线是另一个二分类器常用的工具。它和准确率/召回率曲线比较类似。  
ROC曲线是真正例率（true positive rate，另一个名字叫做召回率）/假正例率（false positive rate, FPR）的曲线。  
FPR是反例被错误分成正例的比率。它等于1-真反例率（true negative rate， TNR）。TNR是反例被正确分类的比率。TNR也叫做特异性。  
ROC曲线是画出召回率（Recall）/（1-TNR）的曲线。
- 使用roc_curve()函数计算各种不同阈值下的 TPR、FPR。
'''
# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.axis([0, 1, 0, 1])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
# plot_roc_curve(fpr, tpr)
# plt.show()
'''
这里同样存在折衷的问题：召回率（TPR）越高，分类器就会产生越多的假正例（FP）。图中的点线是一个完全随机的分类器生成的ROC曲线；  
一个好的分类器的 ROC 曲线应该尽可能远离这条线（即向左上角方向靠拢）。
一个比较分类器之间优劣的方法是：测量ROC曲线下的面积（AUC）。
- 完美的分类器的ROC AUC等于1
- 纯随机分类器的ROC AUC等于0.5。
Sklearn提供了一个**roc_auc_score函数**来计算ROC AUC：
'''
# roc_auc=roc_auc_score(y_train_5, y_scores)
# print(roc_auc) #=>0.9512979603045338
'''
因为ROC曲线跟准确率/召回率曲线（或者叫 PR）很类似，如何决定使用哪一个曲线。  
一个笨拙的规则是，当**真正例很少**或者当你**关注假正例多于假反例**时优先使用PR曲线。  
其他情况使用ROC曲线。举例子，回顾前面的ROC曲线和ROC AUC数值，你或许认为这个分类器很棒。但是这几乎全是因为只有少数正例（“是 5”），而大部分是反例（“非 5”）。相反，PR 曲线清楚显示出这个分类器还有很大的改善空间（PR 曲线应该尽可能地靠近右上角）。
让我们训练一个RandomForestClassifier，然后拿它的的ROC曲线和ROC AUC数值去跟SGDClassifier的比较。  
- 首先你需要得到训练集每个样例的数值。但是由于随机森林分类器的工作方式，RandomForestClassifier不提供decision_function()方法。相反它提供了predict_proba()方法。Skikit-Learn分类器通常二者中的一个。predict_proba()方法返回一个数组，数组的每一行代表一个样例，每一列代表一个类。数组当中的值的意思是：给定一个样例属于给定类的概率。比如，70%的概率这幅图是数字 5。
现在你即将得到 ROC 曲线。将前面一个分类器的 ROC 曲线一并画出来是很有用的，可以清楚地进行比较。
可以看到随机数森林0.99高于roc0.95
'''
# forest_clf = RandomForestClassifier(random_state=42)
# y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,method="predict_proba")
# y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
# fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
# plt.plot(fpr, tpr, "b:", label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.legend(loc="bottom right")
# plt.show()
# roc_auc_forest=roc_auc_score(y_train_5, y_scores_forest)
# print(roc_auc_forest) #=>0.9915297418700815

'''
#### 多类分类
二分类器只能区分两个类，而多类分类器（也被叫做多项式分类器）可以区分多于两个类。
- 直接处理多类分类问题：随机森林分类器、朴素贝叶斯分类器
- 严格的二分类器：SVM、线性分类器
有许多策略可以用二分类器去执行多类分类。
1. “一对所有”（OvA）策略，对mnist手写进行区分，可以做10个二分类器，然后对一张图进行分类，选取得分最高的一个结果
2. “一对一”（OvO）策略，对mnist手写进行区分，对每一对数据都做一个二分类器，0和1,0和2,0和3……如果有n个分类，就要做N*(N-1)/2个分类器，mnist数据就是10*（10-1）/2=45个。  
OvO可以在小的数据集上面可以更多地训练。但是，对于大部分的二分类器来.说，OvA 是更好的选择。  
教程在这里进行了验证，还是使用之前some_digit_test，在训练一个SGDClassifier分类器之后，调用decision_function方法，可以看到十个预测值里面5对应的得分是最高的7.13028292
'''
some_digit_test=X_test[134]
# sgd_clf.fit(X_train,y_train)
# pred=sgd_clf.predict([some_digit_test])
# print(pred)
# img(some_digit_test)
### 调用decision_function()方法。不是返回每个样例的一个数值，而是返回 10 个数值，一个数值对应于一个类。
# scores=sgd_clf.decision_function([some_digit_test])
# print(scores)# =>[[  2.08554369 -17.84337185 -10.45260132  -9.42023796 -14.45383437 7.13028292 -20.2338682  -28.85078879 -12.03563278 -18.1768617 ]]
'''
使用随机数森林来预测
很有意思的预测，这两次的图都比较难分辨，但是用随机树森林还是预测出来，可以看到对应的数值
'''
# forest_clf=RandomForestClassifier()
# forest_clf.fit(X_train,y_train)
# forest_pred=forest_clf.predict([some_digit_test])
# forest_scores=forest_clf.predict_proba([some_digit_test])
# print(forest_pred)
# print(forest_scores) #=>[[0.3 0.  0.7 0.  0.  0.  0.  0.  0.  0. ]]
# img(some_digit_test)
'''
最后，是对分类器的评估，使用交叉验证，可以看到随机梯度下降和随机树森林预测的评估中，准确率分别是86%和94%，同时教程中提到一个正则的方式** StandardScaler**进一步提升精度。
'''
# cross_sores=cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# print(cross_sores) #=>[0.85624693 0.88648885 0.87845063]

# cross_sores=cross_val_score(forest_clf, X_train, y_train, cv=3, scoring="accuracy")
# print(cross_sores) #=>[0.93766701 0.93972618 0.93846154]

'''
### 多标签分类和多输出分类
#### 多标签分类
就是针对多个输入进行判断。
- 人脸识别，在输入的一张图片中有A和C两个人，有一个分类器能够分类A，B，C，对输入的图片识别为[1,0,1]。这种输出多个二值标签的分类系统被叫做多标签分类系统。
教程中设置了两个目标标签（1.大于7的数 >=7；2.奇数 %2==1），然后使用KNN进行分类。最终用交叉验证进行评估。  
- 在这里依旧使用some_digit_test，可以看到数值为9，是一个大于7的奇数，所以结果是[[ True  True]]。  
- 有许多方法去评估一个多标签分类器。一个方法是对每个个体标签去量度 F1 值（或者前面讨论过的其他任意的二分类器的量度标准），然后计算平均值。教程中使用计算全部标签的平均 F1 值。  
然鹅，我的电脑一直没跑出来这个结果。不等了，代码就在下面。
'''
# label_large=(y_train>=7)
# label_odd=(y_train%2==1)
# y_multilabel=np.c_[label_large,label_odd]
# KNN_clf=KNeighborsClassifier()
# KNN_clf.fit(X_train,y_multilabel)
# pred=KNN_clf.predict([some_digit_test])
# print(pred) #=>[[ True  True]]
# img(some_digit_test)
# y_train_knn_pred = cross_val_predict(KNN_clf, X_train, y_train, cv=3)
# F1_score=f1_score(y_train, y_train_knn_pred, average="macro")
# print(F1_score)
'''
#### 多输出分类
最后一种分类任务被叫做“多输出-多类分类”（或者简称为多输出分类）。它是多标签分类的简单泛化，在这里每一个标签可以是多类别的（比如说，它可以有多于两个可能值）。  
教程中专门建立了一个增加图片噪声的系统，
'''
noise1 = np.random.randint(0, 100, (len(X_train), 784))
noise2 = np.random.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise1
X_test_mod = X_test + noise2
y_train_mod = X_train
y_test_mod = X_test
img(X_test_mod[5])
# knn_clf.fit(X_train_mod, y_train_mod)
# clean_digit = knn_clf.predict([X_test_mod[5]])
# plot_digit(clean_digit)