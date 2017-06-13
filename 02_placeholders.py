import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Tensorflow: Placeholders
# Can help assembling the graph first, without knowing the values
# needed for computation
# Something like the x and y in function f(x,y) = 2x + y
# Use so that data can come later
# tf.placeholder(dtype, shape=None, name=None)

# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])
# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)
# use the placeholder as you would a constant or a variable
c = a + b # Short for tf.add(a, b)
with tf.Session() as sess:
    # feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
    # fetch value of c
    print sess.run(c, {a: [1, 2, 3]})
    # >> [6, 7, 8]
 
# What if: feed multiple data points
# Feed all values in, one at a time
# with tf.Session() as sess:
    # for a_value in list_of_values_for_a:
    # print sess.run(c, {a: a_value})

# tf.Graph.is_feedable(tensor)
# True iff tensor is feedable

# Lazy Loading: defer creating/initializing object until it is needed

#Normal
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y) # you create the node for add node before executing the graph
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/l2', sess.graph)
    for _ in range(10):
        print sess2.run(z)
    writer.close()
    
#Lazy 
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess3:
    sess3.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/l2', sess.graph)
    for _ in range(10):
        print sess3.run(tf.add(x, y)) # someone decides to be clever to save one line of code
    writer.close()

#Both give same result, but graph misses the node Add.
#Lazy is bad for reading graphs, but not a bug. Should not do it for large input

#To counter consequences of computing operation thousands of times:
#1. Separate definition of ops from computing/running ops
#2. Use python property to ensure function is also loaded once the first time
#It is called.