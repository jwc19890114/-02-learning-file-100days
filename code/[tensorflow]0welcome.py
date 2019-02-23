from __future__ import print_function
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.app.flags.DEFINE_string(
    'log_dir', os.path.dirname(os.path.abspath(__file__))+'/logs',
    'Directory where event logs are written to.'
)

FLAGS=tf.app.flags.FLAGS

if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
    raise ValueError('you must assign absolute path for --log_dir')

welcome=tf.constant('Welcome to Tensorflow world')

with tf.Session() as sess:
    writer=tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir),sess.graph)
    print('output: ',sess.run(welcome))

writer.close()
sess.close()



import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

welcome=tf.constant('Welcome to Tensorflow world')

with tf.Session() as sess:
    print('output: ',sess.run(welcome))

sess.close()
