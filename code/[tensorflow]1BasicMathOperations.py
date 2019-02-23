import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.app.flags.DEFINE_string(

#     'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
#     'Directory where event logs are written to.'
# )
#
# FLAGS = tf.app.flags.FLAGS

'''
tf的执行了一个特定的操作，输出的将是一个tensor（张量）
属性name定义能够更好地可视化
'''
a = tf.constant(5.0, name='a')
b = tf.constant(10.0, name='b')

x = tf.add(a, b, name='add')
y = tf.div(a, b, name='divide')

'''
session会话，是运行操作的环境，运行如下
'''
with tf.Session() as sess:
    # writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    print('output', sess.run([a, b, x, y]))
# writer.close()
sess.close()


