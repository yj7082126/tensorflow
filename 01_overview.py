import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# a = tf.constant(2)
# b = tf.constant(3)

a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")
y = tf.multiply(a, b, name="mul")

#tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    #print sess.run(x)
    x, y = sess.run([x, y])
    print x, y
    
writer.close()

#To activate tensorflow: source ~/tensorflow/bin/activate
#To run python in tf: python (filename).py
#To run tensorboard: tensorboard --logdir="./graphs" --port 8081
#   Put (writer = tf.summary.FileWriter('./graphs', sess.graph)) in prog.
#   Then Open the URL.
#To deactivate tensorflow: deactivate