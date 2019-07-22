# Program for multivariate Linear Regression

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

# Data in ex1data2.txt is organized under Size, Bedrooms & Price
path = os.getcwd() + '/ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
print(data2.head())
print(data2.describe())

# Plot the data
fiq = plt.figure()
ax = Axes3D(fiq)
ax.scatter(data2['Size'], data2['Bedrooms'], data2['Price'], c='blue', marker='o', alpha=0.5)
ax.set_xlabel('size')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
plt.show()

# Normalize the Data
mean = data2.mean()
std = data2.std()
data_norm = (data2 - mean)/std
print(data_norm.head())


feature_names = ["Size", "Bedrooms"]
data_x= data_norm[feature_names]
data_y = data_norm["Price"]
print('input_shape:', data_x.shape)
print('output_shape:', data_y.shape)
data_x1 = data_norm["Size"]
data_x2 = data_norm["Bedrooms"]

# Plot Normalized Data
fiq = plt.figure()
ax = Axes3D(fiq)
ax.scatter(data_x1, data_x2, data_y, c='blue', marker='o', alpha=0.5)
ax.set_xlabel('size')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
plt.show()

# The hypothesis functions is Y = w1*X1 + w2*X2 + b
# This is a hyperplane in 3D space

with tf.name_scope('inputs'):
	X1 = tf.placeholder(tf.float32, name = 'input1')
	X2 = tf.placeholder(tf.float32, name = 'input2')
	Y  = tf.placeholder(tf.float32, name = 'output')

with tf.name_scope('parameters'):
	w1 = tf.Variable(0.0, name = 'weights_1')
	w2 = tf.Variable(0.0, name = 'weights_2')
	b  = tf.Variable(0.0, name = 'bias')

with tf.name_scope('regression_model'):
	Y_predicted = X1*w1 + X2*w2 + b

# define the loss function
with tf.name_scope('loss_function'):
	loss = tf.reduce_mean(tf.square(Y - Y_predicted, name = 'loss'))

# define the optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

# add summary ops to collect data
W1_hist = tf.summary.histogram('weights_1', w1)
W2_hist = tf.summary.histogram('weights_2', w2)
b_hist  = tf.summary.histogram('biases', b)
cost    = tf.summary.scalar('loss', loss)
merged_summaries = tf.summary.merge_all()

# create a saver object
saver = tf.train.Saver()

cost_history = np.empty(shape=[], dtype=float)

sess = tf.Session()
# create a summary writer
summary_writer =  tf.summary.FileWriter('./multivariate_lin_reg_summary', sess.graph)
sess.run(tf.global_variables_initializer())

# Train the model

for i in range(300):   
	# If only one data at a time is passed to Gradient descent (and cost), it is 
	# stochastic gradient descent i.e., feed_dict={X1:x1, X2:x2, Y:y} 
	# If full data is passed to Gradient descent(and cost) in one go
        # it is batch gradient descent.
	# i.e., feed_dict={X1:data_x1, X2:data_x2, Y:data_y}	
 	for x1, x2, y in zip(data_x1, data_x2, data_y):
  		_,loss_v, summary = sess.run([train_op, loss, merged_summaries], feed_dict={X1:data_x1, X2:data_x2, Y:data_y})
  	
	cost_history=np.append(cost_history, loss_v)
  	if i%20 == 0:
   		print("loss is:", loss_v)
   		# Add the summary to the event file 	   
   		summary_writer.add_summary(summary, i)
w1_value, w2_value, b_value = sess.run([w1, w2, b])	
saver.save(sess,'./multivariate')
summary_writer.close()

x1_test = np.array(data_x1[0:5])
x2_test = np.array(data_x2[0:5])
y_test  = np.array(data_y[0:5])
y_test_predicted = w1_value*x1_test + w2_value*x2_test + b_value
print('True value :', y_test) 
print('Predicted value :', y_test_predicted) 


# Plot Normalized Data and the Hyperplane
x1_surf, x2_surf = np.meshgrid(np.linspace(data_x1.min(), data_x1.max(), 100), np.linspace(data_x2.min(), data_x2.max(), 100))
Y_predicted_surf = x1_surf*w1_value + x2_surf*w2_value + b_value
fiq = plt.figure()
ax = Axes3D(fiq)
sct = ax.scatter(data_x1, data_x2, data_y, c='blue', marker='o', alpha=0.5)
plt_surf = ax.plot_surface(x1_surf, x2_surf, Y_predicted_surf, color='red', alpha = 0.2)
ax.set_xlabel('size')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
plt.show()

sess.close()
