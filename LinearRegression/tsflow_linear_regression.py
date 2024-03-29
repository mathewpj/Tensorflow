# Tensor flow program for stochastic gradient descent.
# In stocastic gradient descent only one data point at a time is passed
# to the Gradient descent algorithm    
# i.e., 
#
#  for x1, y1 in zip(x_data, y_data):
#   _, loss_val,W_, b_ = sess.run([train, loss, W, b],  feed_dict={x: x1, y: y1})
#
# as opposed to batch processed Gradient descent algo where the full data
# is passed (Refer tsflow_multivariate_linear_regression.py)   



import random
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

tf.reset_default_graph()
np.random.seed(1)
# Create some random data
x_data = np.atleast_2d(np.linspace(-1, 1, 101)).T
xdata = x_data.reshape(101, 1)
y_data = 12 * x_data + 6 + np.random.randn(*x_data.shape) * 1.33


W = tf.Variable(tf.random_uniform([1, 1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
y_hat = tf.multiply(x, W) + b

n = 100 # number of iterations
alpha = 0.01 # learning rate
loss = tf.reduce_mean(tf.square(y - y_hat))

# Setting up the Optimizer
 
# The monolithic optimizer.minimize(loss) can be split into  
# more intutive compute_gradients & apply_gradients steps 
# i.e., the steps below
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
# grads = optimizer.compute_gradients(loss, [W, b])
# train = optimizer.apply_gradients(grads)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
train = optimizer.minimize(loss)


# Initialize variables
init = tf.global_variables_initializer()

# Run session
sess = tf.Session()
sess.run(init)

params = np.zeros((3,n))
# Iteratively fit the line
for step in range(n):   
    # Evaluate loss, W, and b for each training step
    # so that we can see its progression as we train
    for x1, y1 in zip(x_data, y_data):
     _, loss_val,W_, b_ = sess.run([train, loss, W, b],  feed_dict={x: x1, y: y1})
	
    # the following is also valid		    
    #  sess.run(train, feed_dict={x: x1, y: y1})	    
    #  loss_val = sess.run(loss, feed_dict={x: x1, y: y1})	    
    #  W_ = sess.run(W)	
    #  b_ = sess.run(b)
    params[:,step] = np.array([W_, 
                              b_,
                              loss_val])

# Plot the best fit line 
x_start = -1.0
# use .item() to convert from a tensor to a plain float/integer  	
# the [0] since np.multiply retuens an array data type
y_start = (np.multiply(W_.item(), x_start) + b_)[0]  	
x_end = 1.0
y_end = (np.multiply(W_.item(), x_end) + b_)[0] 	
x_start_end = np.array([x_start, x_end])
y_start_end = np.array([y_start, y_end])


# Plot the data
fiq = plt.figure()
plt.scatter(x_data, y_data)
plt.plot(x_start_end, y_start_end, 'r-')
plt.show()

param_titles = ['Weight ($W$)', 'Bias ($b$)', 'Loss ($MSE$)']
plt.figure(figsize=(15,10))
for i in range(params.shape[0]):
    plt.subplot(3, 1, i+1)
    plt.plot(params[i])
    plt.title(param_titles[i])
plt.show()

sess.close()
