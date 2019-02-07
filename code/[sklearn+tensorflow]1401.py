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

### 1.RNN的结构
#### 循环神经元
如上文所述，每一个神经元其实是能够表达一个时间序列的，即是说神经元中包含了从t_0到t_n的输入和输出，是一个微小的神经网络；基于多个神经元，可以构建一个循环神经元层。  

每一个RNN有两组权重：一组用于本次时间步长输入x，一组用于前次时间步长输出y  
#### 记忆单元
由于时间t得循环神经元输出，是基于先前时间步骤计算所得，这就类似一种记忆形式。  
而一个RNN的一部分在跨越时间步长后保留的状态，就是储存单元。  

#### 输入和输出序列
RNN 可以同时进行一系列输入并产生一系列输出。  
- 左上角网络，这种类型的网络对于预测时间序列（如股票价格）非常有用：你在过去的N天内给出价格，并且它必须输出向未来一天移动的价格（即从N - 1天前到明天）。
- 右上角网络，可以向网络输入一系列输入，并忽略除最后一个之外的所有输出。 换句话说，这是一个向量网络的序列。 例如，你可以向网络提供与电影评论相对应的单词序列，并且网络将输出情感评分（例如，从-1 [恨]到+1 [爱]）。
- 左下角网络，可以在第一个时间步中为网络提供一个输入（而在其他所有时间步中为零），然后让它输出一个序列。 这是一个向量到序列的网络。 例如，输入可以是图像，输出可以是该图像的标题。
- 右下角网络，有一个序列到向量网络，称为编码器，后面跟着一个称为解码器的向量到序列网络。 可以用于将句子从一种语言翻译成另一种语言。 你会用一种语言给网络喂一个句子，编码器会把这个句子转换成单一的向量表示，然后解码器将这个向量解码成另一种语言的句子。 这种称为编码器-解码器的两步模型，比用单个序列到序列的 RNN（如左上方所示）快速地进行翻译要好得多，因为句子的最后一个单词可以 影响翻译的第一句话，所以你需要等到听完整个句子才能翻译。
#### 用Tensorflow实现基本RNN
构建一个 tanh 激活函数创建由 5 个循环神经元的循环层组成的 RNN。假设 RNN 只运行两个时间步，每个时间步输入大小为 3 的向量。
做了一些改动
- 两个层共享相同的权重和偏差项
- 在每一层都有输入，并从每个层获得输出。为了运行模型，我们需要在两个时间步中都有输入
~~~python
n_inputs = 3
n_neurons = 5
X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))
Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)
init = tf.global_variables_initializer()

X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])  # t=0时
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])  # t=1时
~~~
这个小批量包含四个实例，每个实例都有一个由两个输入组成的输入序列。 最后，Y0_val和Y1_val在所有神经元和小批量中的所有实例的两个时间步中包含网络的输出
~~~python
with tf.Session() as sess:
    init.run()
    Y0_val,Y1_val=sess.run([Y0,Y1],feed_dict={X0:X0_batch,X1:X1_batch})

    print(Y0_val)
    print(Y1_val)

[[-0.97274756  0.20005472  0.9992728   0.99134284 -0.9741845 ]
 [-0.9525191  -0.5095178   1.          1.         -0.97025913]
 [-0.9179007  -0.868501    1.          1.         -0.96574736]
 [ 0.99999833 -0.9946682   0.9998486   0.9999997   0.99942017]]
[[ 0.66469157 -0.99993795  1.          1.          0.36348128]
 [-0.9513445  -0.94794905 -0.9973859  -0.6554711  -0.9753817 ]
 [ 0.7856884  -0.99570477  1.          1.          0.04332094]
 [ 0.9773165  -0.38754973  0.99996173  0.9999522   0.689573  ]]
~~~



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

### 2.时间上的静态展开
static_rnn()函数通过链接单元创建一个展开的RNN网络。在这里构建一个与上一节完全相同的模型。  
创建输入占位符X0和X1
~~~python
n_inputs = 3
n_neurons = 5
X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])
~~~
创建BasicRNNCell，这个函数创建单元的副本来构建展开后的RNN（前面所说每一个单元都有若干时间步长，展开）。  
需要注意的是Tensorflow更新的问题，教程中使用的tf.contrib.rnn.BasicRNNCell已经变成了tf.nn.rnn_cell.BasicRNNCell，而我在运行的时候又变了，放在了Keras下的SimpleRNNCell。  
然后调用static_rnn()，向它提供单元工厂和输入张量，并告诉它输入的数据类型（用来创建初始状态矩阵，默认情况下是全零）。  
**static_rnn(basic_cell, [X0, X1], dtype=tf.float32)**，可以看到static_run()构建了单元的两个副本，每个单元包含有5个循环神经元的循环层。  
static_rnn()返回两个对象，一个是包含每个时间步的输出张量的 Python 列表。 一个是包含网络最终状态的张量
~~~python
# basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
basic_cell = tf.keras.layers.SimpleRNNCell(units=n_neurons)
output_seqs, states = tf.nn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
Y0, Y1 = output_seqs
~~~
如果有 50 个时间步长，则不得不定义 50 个输入占位符和 50 个输出张量。  
简化一下。下面的代码再次构建相同的 RNN，但是这次它需要一个形状为[None，n_steps，n_inputs]的单个输入占位符，其中第一个维度是最小批量大小。  
提取每个时间步的输入序列列表。 X_seqs是形状为n_steps的 Python 列表，包含形状为[None，n_inputs]的张量，其中第一个维度同样是最小批量大小。  
使用transpose()函数交换前两个维度。  
使用unstack()函数沿第一维（即每个时间步的一个张量）提取张量的 Python 列表。  
最后，使用stack()函数将所有输出张量合并成一个张量，然后我们交换前两个维度得到最终输出张量，形状为[None, n_steps，n_neurons]。  
~~~python
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
basic_cell=tf.keras.layers.SimpleRNNCell(units=n_neurons)
output_seqs, states=tf.nn.static_rnn(basic_cell,X_seqs,dtype=tf.float32)

outputs=tf.transpose(tf.stack(output_seqs),perm=[1,0,2])
~~~

通过提供一个包含所有小批量序列的张量来运行网络
~~~python
init = tf.global_variables_initializer()
X_batch = np.array([
        # t = 0      t = 1
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})
print(outputs_val)


[[[-0.71007144 -0.5149771  -0.4025803   0.84779686 -0.695326  ]
  [-0.99820256 -0.9999348   0.9861707   0.90765584 -0.9861588 ]]

 [[-0.9637711  -0.98081887  0.05506388  0.9819332  -0.96890193]
  [-0.906498   -0.2699622   0.8120271  -0.3782449   0.07480869]]

 [[-0.99599314 -0.9994144   0.49068618  0.9979843  -0.99722755]
  [-0.9924605  -0.9945629   0.99294806  0.62711215 -0.91487855]]

 [[ 0.9867179  -0.9999985   0.9999859  -0.95983976  0.9942305 ]
  [ 0.13565563  0.19667651  0.3582377   0.29364806 -0.9558775 ]]]
~~~
看起来挺美好的，因为节省了很多代码编写，但是教程认为这种静态展开的方式仍然会建立一个每个时间步包含一个单元的图。 如果有50个时间步，这个图看起来会非常难看。如果使用大图，在反向传播期间（特别是在 GPU 内存有限的情况下），你甚至可能会发生内存不足（OOM）错误，因为它必须在正向传递期间存储所有张量值，以便可以使用它们在反向传播期间计算梯度。  
这个问题就需要使用动态展开。  

### 3.时间上的动态展开
Tensorflow提供了dynamic_rnn()函数，使用while_loop()操作，在单元上运行适当次数，反向传播期间将GPU内存交换到CPU内存，避免了内存不足的错误。  
动态展开可以在每个时间步接受和输出所有单个张量，不需要堆叠、拆散、转置。
~~~python
if __name__ == '__main__':
    n_inputs = 3
    n_neurons = 5
    n_steps = 2

    X=tf.placeholder(tf.float32,[None, n_steps, n_inputs])
    basic_cell=tf.keras.layers.SimpleRNNCell(units=n_neurons)
    outputs, states=tf.nn.dynamic_rnn(basic_cell,X,dtype=tf.float32)

    init=tf.global_variables_initializer()
    X_batch=np.array([
        [[0, 1, 2], [9, 8, 7]],  # instance 1
        [[3, 4, 5], [0, 0, 0]],  # instance 2
        [[6, 7, 8], [6, 5, 4]],  # instance 3
        [[9, 0, 1], [3, 2, 1]],  # instance 4
    ])

    with tf.Session() as sess:
        init.run()
        outputs_val=outputs.eval(feed_dict={X:X_batch})

    print(outputs_val)

[[[-0.71723616  0.71103394 -0.9259252   0.35195237  0.97944766]
  [-1.          1.         -0.9999448   0.9999934   1.        ]]

 [[-0.99871033  0.99937254 -0.9987622   0.9802172   0.9999999 ]
  [-0.88610965  0.81995314 -0.6732962   0.7383372  -0.2807465 ]]

 [[-0.99999505  0.9999989  -0.99998003  0.9995836   1.        ]
  [-0.9999892   0.999996   -0.9973973   0.99972314  1.        ]]

 [[-0.9825746   0.9999997   0.99574494  0.99944705  0.9999318 ]
  [-0.96585083  0.9917968  -0.84697026  0.99012244  0.9625891 ]]]
~~~
在反向传播期间，while_loop()操作会执行相应的步骤：在正向传递期间存储每次迭代的张量值，以便在反向传递期间使用它们来计算梯度。
'''

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

### 4.训练RNN
在前面两节，是在处理如何构建一个RNN，现在进行训练。  
诀窍是在时间上展开，然后简单地使用常规反向传播。 这个策略被称为时间上的反向传播（BPTT）

#### 训练序列分类器
训练一个RNN做MNIST分类，使用 150 个循环神经元的单元，再加上一个全连接层，其中包含连接到上一个时间步的输出的 10 个神经元（每个类一个），然后是一个 softmax 层
图
参数设置
- n_steps = 28
- n_inputs = 28
- n_neurons = 150
- n_outputs = 10
建模阶段非常简单， 它和我们在第 10 章中建立的 MNIST 分类器几乎是一样的，只是展开的 RNN 替换了隐层。 注意，全连接层连接到状态张量，其仅包含 RNN 的最终状态。  
~~~python
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
learning_rate = 0.001
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.keras.layers.SimpleRNNCell(units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
~~~
加载MNIST数据和部分参数
~~~python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

X_test=X_test.reshape((-1,n_steps,n_inputs))

n_epochs=100
batch_size=150
~~~
现在训练 RNN 。 执行阶段与第 10 章中 MNIST 分类器的执行阶段完全相同，不同之处在于将每个训练的批量提供给网络之前要重新调整。  
~~~python
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch,y_batch in shuffle_batch(X_train,y_train,batch_size):
            X_batch=X_batch.reshape((-1,n_steps,n_inputs))
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        acc_batch=accuracy.eval(feed_dict={X:X_batch,y:y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)

0 Last batch accuracy: 0.93333334 Test accuracy: 0.9436
1 Last batch accuracy: 0.94666666 Test accuracy: 0.95
2 Last batch accuracy: 0.97333336 Test accuracy: 0.9624
3 Last batch accuracy: 0.9866667 Test accuracy: 0.9576
4 Last batch accuracy: 1.0 Test accuracy: 0.9689
5 Last batch accuracy: 0.97333336 Test accuracy: 0.9674
...
95 Last batch accuracy: 0.99333334 Test accuracy: 0.9768
96 Last batch accuracy: 1.0 Test accuracy: 0.9775
97 Last batch accuracy: 0.99333334 Test accuracy: 0.978
98 Last batch accuracy: 0.9866667 Test accuracy: 0.9763
99 Last batch accuracy: 1.0 Test accuracy: 0.9781
~~~
很惊艳啊，第一批次就是93%。


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

### 5.构建多层RNN
我们在这里构建一个三层的RNN。  
基本参数设定如下
~~~python
n_steps = 28
n_inputs = 28
n_neurons = 100
n_outputs = 10
learning_rate = 0.001
n_layers = 3
~~~
构建三层，可以发现是使用for循环，循环3次将之前一层的神经网络复制成三个压入一个layers得list中。
layers = [
    tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    for layer in range(n_layers)
]
然后使用tf.nn.rnn_cell.MultiRNNCell(layers)处理layers。
~~~python
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
# 原始一层的神经网络
# basic_cell = tf.keras.layers.SimpleRNNCell(units=n_neurons)
layers = [
    tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    for layer in range(n_layers)
]
# 这个地方应该也可以使用Keras，但是没有查到
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)

outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
# tf.concat是连接两个矩阵的操作，其中values应该是一个tensor的list或者tuple。axis则是我们想要连接的维度。tf.concat返回的是连接后的tensor。
states_concat = tf.concat(axis=1, values=states)

logits = tf.layers.dense(states_concat, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


X_test = X_test.reshape((-1, n_steps, n_inputs))

n_epochs = 10
batch_size = 150
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)


0 Last batch accuracy: 0.9533333 Test accuracy: 0.9516
1 Last batch accuracy: 0.97333336 Test accuracy: 0.9643
2 Last batch accuracy: 0.99333334 Test accuracy: 0.9742
3 Last batch accuracy: 0.98 Test accuracy: 0.9787
4 Last batch accuracy: 0.97333336 Test accuracy: 0.9762
5 Last batch accuracy: 0.98 Test accuracy: 0.9776
6 Last batch accuracy: 0.97333336 Test accuracy: 0.9804
7 Last batch accuracy: 0.99333334 Test accuracy: 0.984
8 Last batch accuracy: 0.99333334 Test accuracy: 0.9788
9 Last batch accuracy: 0.99333334 Test accuracy: 0.9841

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


# n_inputs = 3
# n_neurons = 5
# X0 = tf.placeholder(tf.float32, [None, n_inputs])
# X1 = tf.placeholder(tf.float32, [None, n_inputs])
#
# Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
# Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
# b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))
# Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
# Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)
# init = tf.global_variables_initializer()
#
# X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])  # t=0时
# X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])  # t=1时
# with tf.Session() as sess:
#     init.run()
#     Y0_val,Y1_val=sess.run([Y0,Y1],feed_dict={X0:X0_batch,X1:X1_batch})
#
#     print(Y0_val)
#     print(Y1_val)



# X0 = tf.placeholder(tf.float32, [None, n_inputs])
# X1 = tf.placeholder(tf.float32, [None, n_inputs])
## basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
# basic_cell = tf.keras.layers.SimpleRNNCell(units=n_neurons)
# output_seqs, states = tf.nn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
# Y0, Y1 = output_seqs
# print("Y0",Y0)
# print("Y1",Y1)
# print(states)

# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
# basic_cell=tf.keras.layers.SimpleRNNCell(units=n_neurons)
# output_seqs, states=tf.nn.static_rnn(basic_cell,X_seqs,dtype=tf.float32)
#
# outputs=tf.transpose(tf.stack(output_seqs),perm=[1,0,2])
#
#
# init = tf.global_variables_initializer()
# X_batch = np.array([
#         # t = 0      t = 1
#         [[0, 1, 2], [9, 8, 7]], # instance 1
#         [[3, 4, 5], [0, 0, 0]], # instance 2
#         [[6, 7, 8], [6, 5, 4]], # instance 3
#         [[9, 0, 1], [3, 2, 1]], # instance 4
#     ])
#
# with tf.Session() as sess:
#     init.run()
#     outputs_val = outputs.eval(feed_dict={X: X_batch})
#
# print(outputs_val)

import pandas as pd

# if __name__ == '__main__':
#     n_inputs = 3
#     n_neurons = 5
#     n_steps = 2
#
#     X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
#     basic_cell = tf.keras.layers.SimpleRNNCell(units=n_neurons)
#     seq_length = tf.placeholder(tf.int32, [None])
#     outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)
#
#     init = tf.global_variables_initializer()
#     X_batch = np.array([
#         [[0, 1, 2], [9, 8, 7]],  # instance 1
#         [[3, 4, 5], [0, 0, 0]],  # instance 2
#         [[6, 7, 8], [6, 5, 4]],  # instance 3
#         [[9, 0, 1], [3, 2, 1]],  # instance 4
#     ])
#     seq_length_batch = np.array([2, 1, 2, 2])
#
#     with tf.Session() as sess:
#         init.run()
#         # outputs_val=outputs.eval(feed_dict={X:X_batch})
#         outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})
#     print(outputs_val)
#     print(states_val)


n_steps = 28
n_inputs = 28
n_neurons = 100
n_outputs = 10
learning_rate = 0.001
n_layers = 3

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
# 原始一层的神经网络
# basic_cell = tf.keras.layers.SimpleRNNCell(units=n_neurons)
layers = [
    tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    for layer in range(n_layers)
]
# 这个地方应该也可以使用Keras，但是没有查到
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)

outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
# tf.concat是连接两个矩阵的操作，其中values应该是一个tensor的list或者tuple。axis则是我们想要连接的维度。tf.concat返回的是连接后的tensor。
states_concat = tf.concat(axis=1, values=states)

logits = tf.layers.dense(states_concat, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


X_test = X_test.reshape((-1, n_steps, n_inputs))

n_epochs = 10
batch_size = 150
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)
