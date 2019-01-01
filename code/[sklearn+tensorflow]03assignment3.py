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
在这里完成第三个
'''
'''
Titanic数据集，目标是基于乘客属性（年龄、性别、舱室……）预测乘客能不能活下来。  
数据集需要再Kaggle上下载，我这里放在了Titanic_data文件夹下面
'''
### 1. 引入库
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


### 2. 读取数据
TITANIC_PATH = os.path.join(r".\Titanic_data")


def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")

### 3. 查看训练数据的基本信息。
# 从中可以看到丢失数据，891人，有年龄的714人，客舱204人，登记的889人。
# 可以看出，客舱数据缺失的较多，在这里就放弃使用了（网络上大多数教程都这么讲），年龄部分使用中间年龄来补齐。
# print(train_data.info())
'''
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
None
'''
# print(train_data.describe())
'''
平均年龄为29.7
	PassengerId	Survived	Pclass	Age	SibSp	Parch	Fare
count	891.000000	891.000000	891.000000	714.000000	891.000000	891.000000	891.000000
mean	446.000000	0.383838	2.308642	29.699118	0.523008	0.381594	32.204208
std	257.353842	0.486592	0.836071	14.526497	1.102743	0.806057	49.693429
min	1.000000	0.000000	1.000000	0.420000	0.000000	0.000000	0.000000
25%	223.500000	0.000000	2.000000	20.125000	0.000000	0.000000	7.910400
50%	446.000000	0.000000	3.000000	28.000000	0.000000	0.000000	14.454200
75%	668.500000	1.000000	3.000000	38.000000	1.000000	0.000000	31.000000
max	891.000000	1.000000	3.000000	80.000000	8.000000	6.000000	512.329200
接下来可以观测一下各类数据的分布情况
获救、舱室等级、性别、是否登记
'''


### 4. 构建预处理
# 由于sklearn不再解决DataFrame，在这里要建立一个类用来选择数字或分类列
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


#### 处理数据值，进行归一化处理
try:
    from sklearn.impute import SimpleImputer  # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

num_pipeline = Pipeline([("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
                         ("imputer", SimpleImputer(strategy="median")), ])
num_pipeline.fit_transform(train_data)
# print(num_pipeline)


#### 处理字符串分类，数字化处理
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
    ("imputer", MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse=False)),
])
cat_pipeline.fit_transform(train_data)
# print(cat_pipeline)


from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
# preprocess_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline), ("cat_pipeline", cat_pipeline),])
### 5. 训练模型，现在的数据是可以用模型来处理的了
X_train=preprocess_pipeline.fit_transform(train_data)
print(X_train)
y_train=train_data["Survived"]
print(y_train)




svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)

X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)


svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_final=svm_scores.mean()

forest_clf=RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores=cross_val_score(forest_clf,X_train,y_train,cv=10)
forest_final=forest_scores.mean()
print("svm_final:",svm_final)#=>0.7365250822835092
print("forest_final:",forest_final)#=>0.8149526160481217

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()