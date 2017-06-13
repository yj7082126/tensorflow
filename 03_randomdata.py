import os
# Silence most warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xlrd


# Step 1: read in data from the .xls file
X_input = np.linspace(-1, 1, 100)
Y_input = X_input * 3 + np.random.randn(X_input.shape[0]) * 0.5
n_samples = 100

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
w = tf.Variable(0.0, name='weight_1')
u = tf.Variable(0.0, name='weight_2')
b = tf.Variable(0.0, name='bias')

# Step 4: build model to predict Y
Y_predicted = X * X * w + X * u + b 

# Step 5: use the square error as the loss function

loss = tf.square(Y-Y_predicted)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
	# Step 7: initialize the necessary variables, in this case, w and b
	sess.run(tf.global_variables_initializer()) 
	
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	
	# Step 8: train the model
	for i in range(100): # train the model 100 times
		total_loss = 0
		for x , y in zip(X_input, Y_input):
			# Session runs train_op and fetch values of loss
			_, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y}) 
			total_loss += l
		print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

	# close the writer when you're done using it
	writer.close() 
	
	# Step 9: output the values of w and b
	w_value, u_value, b_value = sess.run([w, u, b]) 
	
# plot the results
X, Y = X_input, Y_input
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, (X * X * w_value) + (X * u_value) + b_value , 'r', label='Predicted data')
plt.legend()
plt.show()
plt.savefig('plot_3.png')