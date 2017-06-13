import os
# Silence most warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import tensorflow as tf
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xlrd

# Step 1: read in data from the .xls file
from tensorflow.examples.tutorials.mnist import input_data
MNIST = input_data.read_data_sets("/data/mnist", one_hot=True)

learning_rate = 0.01
batch_size = 128
n_epochs = 25
# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, [batch_size, 784], name='X')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='Y')

# Step 3: create weight and bias, initialized to 0
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weight')
b = tf.Variable(tf.zeros([1,10]), name='bias')

# Step 4: build model to predict Y
logits = tf.matmul(X, w) + b 

# Step 5: use the square error as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = logits)
loss = tf.reduce_mean(entropy)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
	# Step 7: initialize the necessary variables, in this case, w and b
	start_time = time.time()
	sess.run(tf.global_variables_initializer()) 
	n_batches = int(MNIST.train.num_examples/batch_size)
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	
	# Step 8: train the model
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0
		for _ in range(n_batches):
			# Session runs train_op and fetch values of loss
			X_batch, Y_batch = MNIST.train.next_batch(batch_size)
			_, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
			
	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!') # should be around 0.35 after 25 epochs

	n_batches = int(MNIST.test.num_examples/batch_size)
	total_correct_preds = 0
	for i in range(n_batches):
		X_batch, Y_batch = MNIST.test.next_batch(batch_size)
		_, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y: Y_batch})
		preds = tf.nn.softmax(logits_batch)
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
		total_correct_preds += sess.run(accuracy)	
	
	print ('Accuracy {0}'.format(total_correct_preds/MNIST.test.num_examples))
	# close the writer when you're done using it
	writer.close() 
	