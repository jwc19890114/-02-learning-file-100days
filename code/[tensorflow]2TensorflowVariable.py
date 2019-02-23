'''
对tensorflow变量的介绍，生成和初始化
定义变量很重要，因为变量控制参数。没有参数的话，训练、更新、保存、恢复以及其他操作都不能实现。
在tensorflow中定义变量是仅仅是有确定大小和类型的张量（tensor）。
张量必须使用数值初始化成为有效的。
'''
import tensorflow as tf
from tensorflow.python.framework import ops

'''
生成变量
对于变量的生成，使用tf.Variable()。
当我们定义变量时，就是传入一个张量和它的数值进入到图（graph）中
A 变量张量有一个数值传入途中
通过使用tf.assign，一个初始化设置变量的初始数值
'''
weights = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name='weights')
biases = tf.Variable(tf.zeros([3]), name='biases')
custom_variable = tf.Variable(tf.zeros([3]), name="custom")

# 获取所有变量的张量并存储在一个list中
all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

'''
初始化
必须在模型其他操作前运行初始化变量。这是自然的了，如果不进行变量初始化，其他操作无法进行。
变量能够全局初始化、指定初始化或从其他变量中初始化。
这里会对不同的操作调查。
'''
# 初始化特殊变量，使用tf.variables_initializer()，可以命令tensorflow仅仅初始化一个特定的比纳凉
# 需要注意的是custom initialization并不意味着不需要初始化其他变量。所有要使用到的变量都必须初始化或从已保存的便两种修改
all_variables_list = [weights, custom_variable]
init_custom_op = tf.variables_initializer(var_list=all_variables_list)

# 全局变量初始化，使用tf.global_variables_initializer(),所有变量能够一次性初始化，这一操作必须在模型建构完成后进行
# 两种初始化的方法
init_all_op = tf.global_variables_initializer()
init_all_op = tf.variables_initializer(var_list=all_variables_list)

# 使用其他已有变量进行初始化，使用initialized_value()，使用之前已经定义的变量数值初始化
WeightsNew = tf.Variable(weights.initialized_value(), name='WeightsNew')
init_WeightsNew_op = tf.variables_initializer(var_list=[WeightsNew])

# 运行会话session
with tf.Session() as sess:
    sess.run(init_all_op)
    sess.run(init_custom_op)
    sess.run(init_WeightsNew_op)
