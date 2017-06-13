import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Tensorflow: Variables
a = tf.Variable(2, name="scalar") 

b = tf.Variable([2, 3], name="vector") 

c = tf.Variable([[0, 1], [2, 3]], name="matrix") 

W = tf.Variable(tf.zeros([784,10]))

# tf.Variable = class. tf.constant = operation
# Operations for Variables

init = tf.global_variables_initializer()
# Initialize all variables at once

init_ab = tf.variables_initializer([a, b], name="init_ab")
# Initialize subset of variables

init_w = W.initializer
# Initialize single variable

# Tensorflow: Assigning

#1.
#W = tf.Variable(10)
#W.assign(100)
#with tf.Session() as sess:
#    sess.run(W.initializer)
#    print W.eval() 
     # >> 10
# W.assign(100) not assign val to W 
# It creates an assign op, and that op needs to be run to take effect.
# Try this:

#2.
#W = tf.Variable(10)
#assign_op = W.assign(100)
#with tf.Session() as sess:
#    sess.run(W.initializer)
#    sess.run(assign_op)
#    print W.eval() 
     # >> 100
# You dont need to initialize variable because assign_op does it for you.
# initializer op is the assign op that assigns the variable initial value
# to the variable itself.

#3. Repetition
# create a variable whose original value is 2
#my_var = tf.Variable(2, name="my_var")
# assign a * 2 to a and call that op a_times_two
#my_var_times_two = my_var.assign(2 * my_var)
#with tf.Session() as sess:
#    sess.run(my_var.initializer)
#    sess.run(my_var_times_two) 
#    print my_var.eval()
    # >> 4
#    sess.run(my_var_times_two) 
#    print my_var.eval()
    # >> 8
#    sess.run(my_var_times_two) 
#    print my_var.eval()
    # >> 16
# Assigns 2 * my_var to a every time my_var_times_two is fetched.

#4. Assign_add() and Assign_sub()
#my_var = tf.Variable(10)
#with tf.Session() as sess:
#    sess.run(my_var.initializer)
#    print my_var.eval()
    # increment by 10
#    sess.run(my_var.assign_add(10)) 
#    print my_var.eval()
    # >> 20
    # decrement by 2
#    sess.run(my_var.assign_sub(2)) 
#    print my_var.eval()
    # >> 18
    
#5.
#Each session maintains own copy of variable
#W = tf.Variable(10)
#sess1 = tf.Session()
#sess2 = tf.Session()
#sess1.run(W.initializer)
#sess2.run(W.initializer)
#print sess1.run(W.assign_add(10)) # >> 20
#print sess2.run(W.assign_sub(2)) # >> 8
#print sess1.run(W.assign_add(100)) # >> 120
#print sess2.run(W.assign_sub(50)) # >> -42
#sess1.close()
#sess2.close()

#6. 
# Session vs Interactive Session
# InteractiveSession makes itself the default (x need to specify sess)
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# We can just use 'c.eval()' without specifying the context 'sess'
print(c.eval())
sess.close()

# Control Dependencies
# tf.Graph.control_dependencies(control_inputs)
# Define which ops should be run first
