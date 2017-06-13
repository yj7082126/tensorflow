import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Tensorflow: specific values 

input_tensor = tf.constant([[0,1], [2,3], [4,5]])

a = tf.zeros([2, 3], tf.int32)
# [[0 0 0]
#  [0 0 0]]

b = tf.zeros_like(input_tensor)
# [[0 0]
#  [0 0]
#  [0 0]]

c = tf.ones([2, 3], tf.int32)
# [[1 1 1]
#  [1 1 1]]

d = tf.ones_like(input_tensor)
# [[1 1]
#  [1 1]
#  [1 1]]

e = tf.fill([2, 3], 8)
# [[8 8 8]
#  [8 8 8]]
 
f = tf.linspace(10.0, 13.0, 4) #should be float
# [ 10.  11.  12.  13.]

g = tf.range(3, 18, 3)
# [ 3  6  9 12 15]

h = tf.range(4) #tensor objects are not iterable
# [0 1 2 3]

i = tf.random_normal([2, 3])
# [[-0.51910216 -0.21610695  0.89027923]
#  [ 0.47151601  2.0972178  -0.95941263]]

# also tf.truncated_normal, tf.random_uniform, tf.random_shuffle, tf.random_crop,
# tf.multinomial, tf.random_gamma

tf.set_random_seed(100)

# Only use constants for primitive types.
# If constants are big, loading graphs get expensive
# Use variables or readers for more data that requires more memory


# Tensorflow: Operations

A = tf.constant([[3,6],[3,2],[2,5]])

B = tf.constant([[2,2],[1,4],[3,7]])

C = tf.add(A, B) 
# [[ 5  8]
#  [ 4  6]
#  [ 5 12]]

D = tf.add_n([A, B, B])
# [[ 7 10]
#  [ 5 10]
#  [ 8 19]]

E = tf.multiply(A, B)
# [[ 6 12]
#  [ 3  8]
#  [ 6 35]]

#F = tf.matmul(A, B)
#vALUEeRROR

G = tf.matmul(tf.reshape(A, [2, 3]), tf.reshape(B, [3, 2]))
# [[21 51]
#  [21 47]]
 
H = tf.div(A, B)
# [[1 3]
#  [3 0]
#  [0 0]]

I = tf.mod(A, B)
# [[1 0]
#  [0 2]
#  [2 5]]

#Tensorflow: Data Types
#Uses Python native types: boolean, numeric, strings

#0 dimension tensor (scalar)
t_0 = 19
t_0_1 = tf.zeros_like(t_0) 
# 0
t_0_2 = tf.ones_like(t_0)
# 1

#1 dimension tensor (vector)
t_1 = ['apple', 'peach', 'banana']
t_1_1 = tf.zeros_like(t_1)
# ['' '' '']
#t_1_2 = tf.ones_like(t_1)
#TypeError

#2 dimension tensor (matrix)
t_2 = [[True, False, False],
        [False, False, True],
        [False, True, False]]
t_2_1 = tf.zeros_like(t_2)
# [[False False False]
#  [False False False]
#  [False False False]]

t_2_2 = tf.ones_like(t_2)
# [[True True True]
#  [True True True]
#  [True True True]]

# TensorFlow seems to integrate seemlessly with NumPy (might not be so
# compatible in the future)
# Do not use Python native types for tensors because TensorFlow 

with tf.Session() as sess:
    #print sess.run(i)
    print sess.run(t_2_1)
    