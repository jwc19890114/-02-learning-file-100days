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

### 11.NLP初试

在这里使用的是http://mattmahoney.net/dc/text8.zip压缩文件，同样也可以下载下来，然后使用python进行读取。
~~~python
from six.moves import urllib
import errno
import os
import zipfile

words_path=r"./datasets/words"
words_url=r"http://mattmahoney.net/dc/text8.zip"

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

def fetch_words_data(words_url=words_url, words_path=words_path):
    os.makedirs(words_path,exist_ok=True)
    zip_path=os.path.join(words_path,"words.zip")
    # if not os.path.exists(zip_path):
    #     urllib.request.urlretrieve(words_url, zip_path)
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    return data.decode("ascii").split()

#因为用的是pycharm，不像jupyter那样，所以第二次调试的时候注销了fetch_words_data方法中关于压缩包生成的代码
words=fetch_words_data()
# print(words[:5])
~~~
构建字典
~~~python
words = fetch_words_data()
# print(words[:5])

vocabulary_size = 50000
vocabulary = [("UNK", None)] + Counter(words).most_common(vocabulary_size - 1)
vocabulary = np.array([word for word, _ in vocabulary])
dictionary={word: code for code, word in enumerate(vocabulary)}
data = np.array([dictionary.get(word, 0) for word in words])

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips==0
    assert num_skips<=2*skip_window
    batch=np.ndarray(shape=[batch_size], dtype=np.int32)
    labels=np.ndarray(shape=[batch_size,1],dtype=np.int32)
    span=2*skip_window+1
    buffer=deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = np.random.randint(0, span)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

np.random.seed(42)
~~~
构建模型
~~~python
batch_size = 128
embedding_size = 128  # 嵌入向量的维度
skip_window = 1
num_skips = 2  # 重复使用输入来生成标签的次数
# 在这里，教程使用了一个随机验证集作为最近临居进行抽样，限制了验证样本的ID数值

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

learning_rate = 0.01

train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

vocabulary_size=50000
embedding_size=150

init_embeds=tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0)
embeddings=tf.Variable(init_embeds)

train_inputs=tf.placeholder(tf.int32,shape=[None])
embed=tf.nn.embedding_lookup(embeddings,train_inputs)

# Construct the variables for the NCE loss
nce_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                        stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# 计算每一个batch的NCE loss
# tf.nce_loss：NCE是softmax的一种近似，但是为什么要做这种近似，而不直接用softmax呢？因为效果比softmax好……

loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                   num_sampled, vocabulary_size))

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# 使用cosine计算各个词汇之间的相似度
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

init = tf.global_variables_initializer()
~~~
训练模型
~~~python
num_steps = 10001

with tf.Session() as session:
    init.run()

    average_loss = 0
    for step in range(num_steps):
        print("\rIteration: {}".format(step), end="\t")
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the training op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([training_op, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = vocabulary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = vocabulary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)

    final_embeddings = normalized_embeddings.eval()

    np.save("./tf_logs/my_final_embeddings.npy", final_embeddings)

Iteration: 0	Average loss at step  0 :  284.065673828125
Nearest to people: gave, replaces, accept, shelves, pinnacle, grandfather, encompass, ctbt,
Nearest to can: ratites, spenser, aai, paine, neanderthal, jodie, fed, lewinsky,
Nearest to than: newborn, bolivian, harness, nineties, lpc, masochism, simplifying, cassady,
Nearest to its: austere, norsemen, cantigas, hermione, flockhart, quackery, apr, quaestor,
Nearest to their: dilemmas, holbach, anisotropic, imposes, slavic, buckley, chola, rivest,
Nearest to have: consumes, drogheda, resignation, knitted, traitors, sandro, bremer, azeotrope,
Nearest to six: cornelius, chagas, parrot, mckenzie, immunization, kombinate, literature, delgado,
Nearest to over: menachem, udf, chukotka, haste, kamal, edition, receiving, magnetism,
Nearest to two: songwriter, cultured, imitates, cheka, mpa, heracleidae, given, colon,
Nearest to UNK: conferring, hole, metrolink, zeno, macrovision, trash, maduro, sporadic,
Nearest to was: cruisers, and, boudinot, moabites, struggling, fractal, superstardom, kinship,
Nearest to his: alkalis, sellers, licensee, libertine, hackers, herbivorous, parthenon, breed,
Nearest to b: duval, cannabis, libretto, divider, blythe, bloemfontein, terry, preserved,
Nearest to this: vaccinations, alpinus, emancipated, toaster, gorilla, io, ther, undergoing,
Nearest to use: inertial, hypertension, devotions, brokered, crumbs, discrepancy, polyatomic, vor,
Nearest to one: preview, duo, bout, sets, etruscan, chaplin, jedi, cryonicists,
Iteration: 2000	Average loss at step  2000 :  131.6964364299774
Iteration: 4000	Average loss at step  4000 :  62.64465307974815
Iteration: 6000	Average loss at step  6000 :  41.10564466834068
Iteration: 8000	Average loss at step  8000 :  31.070515101790427
Iteration: 10000	Average loss at step  10000 :  26.000642070174216
Nearest to people: abbots, that, oath, dilation, stream, UNK, bellows, astatine,
Nearest to can: ginsberg, is, may, evolutionary, therefore, nonfiction, that, to,
Nearest to than: much, kleine, possess, the, or, a, irate, charges,
Nearest to its: the, carbonate, tarmac, rn, symbolic, secluded, seismology, nsu,
Nearest to their: the, propositional, atomists, decreed, antigua, gangsta, counterparts, astatine,
Nearest to have: and, has, hebrides, been, aberdeen, axon, tableland, be,
Nearest to six: seven, one, four, five, eight, zero, nine, two,
Nearest to over: utopian, beers, jabir, screens, plural, gangsta, originates, airships,
Nearest to two: zero, three, one, four, five, seven, nine, six,
Nearest to UNK: and, the, one, cosmonaut, altaic, of, astatine, bicycle,
Nearest to was: and, actinium, aberdeenshire, had, aggression, ceased, kierkegaard, overseas,
Nearest to his: the, absurd, orange, and, alhazred, atomism, plutonium, explosive,
Nearest to b: one, art, six, eight, seven, indelible, bicycle, dragster,
Nearest to this: the, that, asteraceae, morphism, willing, whale, a, aquarius,
Nearest to use: arrest, and, morphisms, winfield, aziz, quantum, ataxia, carrot,
Nearest to one: nine, seven, eight, four, five, six, two, three,
~~~
绘制词嵌入图
~~~python
def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')


from sklearn.manifold import TSNE

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [vocabulary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
~~~

'''

from six.moves import urllib
import errno
import os
import zipfile
from collections import Counter, deque
import numpy as np
import tensorflow as tf
import numpy as np
import os

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

words_path = r"./datasets/words"
words_url = r"http://mattmahoney.net/dc/text8.zip"


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def fetch_words_data(words_url=words_url, words_path=words_path):
    os.makedirs(words_path, exist_ok=True)
    zip_path = os.path.join(words_path, "words.zip")
    # if not os.path.exists(zip_path):
    #     urllib.request.urlretrieve(words_url, zip_path)
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    return data.decode("ascii").split()


# 因为用的是pycharm，不像jupyter那样，所以第二次调试的时候注销了fetch_words_data方法中关于压缩包生成的代码
words = fetch_words_data()
# print(words[:5])

vocabulary_size = 50000
vocabulary = [("UNK", None)] + Counter(words).most_common(vocabulary_size - 1)
vocabulary = np.array([word for word, _ in vocabulary])
dictionary = {word: code for code, word in enumerate(vocabulary)}
data = np.array([dictionary.get(word, 0) for word in words])


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=[batch_size], dtype=np.int32)
    labels = np.ndarray(shape=[batch_size, 1], dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = np.random.randint(0, span)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


np.random.seed(42)
data_index=0
# batch, labels=generate_batch(8,2,1)
# print(batch)
# print([vocabulary[word] for word in batch])


batch_size = 128
embedding_size = 128  # 嵌入向量的维度
skip_window = 1
num_skips = 2  # 重复使用输入来生成标签的次数
# 在这里，教程使用了一个随机验证集作为最近临居进行抽样，限制了验证样本的ID数值

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

learning_rate = 0.01

train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

vocabulary_size = 50000
embedding_size = 150

init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
embeddings = tf.Variable(init_embeds)

train_inputs = tf.placeholder(tf.int32, shape=[None])
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Construct the variables for the NCE loss
nce_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                        stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# 计算每一个batch的NCE loss
# tf.nce_loss：NCE是softmax的一种近似，但是为什么要做这种近似，而不直接用softmax呢？因为效果比softmax好……

loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                   num_sampled, vocabulary_size))

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# 使用cosine计算各个词汇之间的相似度
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

init = tf.global_variables_initializer()

# 训练该模型
num_steps = 10001

with tf.Session() as session:
    init.run()

    average_loss = 0
    for step in range(num_steps):
        print("\rIteration: {}".format(step), end="\t")
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the training op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([training_op, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = vocabulary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = vocabulary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)

    final_embeddings = normalized_embeddings.eval()

    np.save("./tf_logs/my_final_embeddings.npy", final_embeddings)


def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')


from sklearn.manifold import TSNE

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [vocabulary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
