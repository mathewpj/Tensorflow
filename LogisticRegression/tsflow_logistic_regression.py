# Author : mathew.p.joseph@gmail.com

# Import dependencies
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split

# Get data from the csv file
data = np.genfromtxt('./ex2data1.csv', delimiter=',')

# Get the 2 features (hours slept & hours studied)
X_frm_csv = data[:, 0:2]
# Get the result (0 suspended - 1 approved)
Y_frm_csv = data[:, 2]


'''
# Plotting the original data
pos = np.where(Y_frm_csv == 1)
neg = np.where(Y_frm_csv == 0)
plt.scatter(X_frm_csv[pos, 0], X_frm_csv[pos, 1], marker='o', c='b')
plt.scatter(X_frm_csv[neg, 0], X_frm_csv[neg, 1], marker='x', c='r')
plt.xlabel('Hours studied')
plt.ylabel('Hours slept')
plt.legend(['Approved', 'Suspended'])
plt.show()
'''

# Split the data in train & test in the ratio of test_size
# i.e., if test_size  = 0.2, the data is split as train:test::80:20
# setting shuffle to False so that the data is not shuffled 
Y_reshape = data[:,2].reshape(data[:,2].shape[0], 1)
x_train, x_test, y_train, y_test = train_test_split(data[:, 0:2], Y_reshape, test_size=0.2, shuffle=False)

# Normalize the data
x_train[:,0] = (x_train[:,0] - x_train[:,0].mean())/x_train[:,0].std()
x_train[:,1] = (x_train[:,1] - x_train[:,1].mean())/x_train[:,1].std()

'''
# Plot the data points 
pos = np.where(y_train == 1)
neg = np.where(y_train == 0)
plt.scatter(x_train[pos, 0], x_train[pos, 1], marker='o', c='b')
plt.scatter(x_train[neg, 0], x_train[neg, 1], marker='x', c='r')
plt.xlabel('Hours studied')
plt.ylabel('Hours slept')
plt.legend(['Approved', 'Suspended'])
plt.show()
'''

# Incomming data should be of the form X_0, X_1, X_2
# so append a column of ones i.e., X_0 = 1
  
column_of_ones = np.ones((80, 1))
x_train = np.column_stack((column_of_ones, x_train))

column_of_ones = np.ones((20, 1))
x_test = np.column_stack((column_of_ones, x_test))

print ("x_train shape: " + str(x_train.shape))
print ("y_train shape: " + str(y_train.shape))
print ("x_test shape: " + str(x_test.shape))
print ("y_test shape: " + str(y_test.shape))


#tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape = [None, 3]) 
Y = tf.placeholder(tf.float32, [None, 1])

# The Weights are W_0, W_1, W_2
W = tf.Variable(tf.random_normal([3, 1]), name='weight')

# For data set of 2 variables, i.e, X_1, X_2 
# The decision boundary is a line of the form 
# W_0*X_0 + W_1*X_1 + W_2*X_2 = 0
# rearranging the terms , W_2*X_2 = -W_1*X_1 - W_0*X_0
# or X_2 = -(W_1/W_2)*X_1 - (W_0/W_2)*X_0
# => y = mx + c  	  

# Hypothesis
# Since X = [N X 3] and W = [3 X 1],
# tf.matmul o/p is a column vector of N X 1
# Hence the  o/p of tf.sigmoid () is a column 
# of also N X 1  
hypothesis = tf.sigmoid(tf.matmul(X, W))

# Cost Function
# Using mean squared error from linear regression is not a good idea here,
# as the resulting cost function is not convex and so is not well-suited
# for gradient descent. Y * tf.log(hypothesis) + (1- Y)*tf.log(1 - hypothesis)
# is also called the Binary Cross Entropy
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1- Y)*tf.log(1 - hypothesis))

# Initialize the Step Size 
step_size = 0.01 
# Optimize Cost Function using Gradient Descent
train = tf.train.GradientDescentOptimizer(learning_rate=step_size).minimize(cost)
# Initialize number of iterationa for Gradient Descent Algorithm
n = 3000

# Declare a variable to hold the cost values for every iteration of 
# Gradient Descent Algorithm
cost_val = []   

# A bit of tensorflow magic to calculate the accuracy of prediction
z = tf.sigmoid(tf.matmul(X, W))
predicted = tf.cast(z  > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session() 
sess.run(init)

# Train the model
for epoch in range(n):
       		cost_, _, = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
		cost_val.append(cost_)
		"""
		 W_arr = W.eval(sess)
		 X2_left = -1*(W_arr[1]/W_arr[2])*(-2.0) - (W_arr[0]/W_arr[2])
		 X2_right = -1*(W_arr[1]/W_arr[2])*(2.0) - (W_arr[0]/W_arr[2])
		 x_start_end = np.array([-2.0, 2.0])
		 y_start_end = np.array([X2_left, X2_right])

		 # Plotting
		 pos = np.where(y_train == 1)
		 neg = np.where(y_train == 0)
		 plt.scatter(x_train[pos, 1], x_train[pos, 2], marker='o', c='b')
		 plt.scatter(x_train[neg, 1], x_train[neg, 2], marker='x', c='r')
		 plt.plot(x_start_end, y_start_end, 'g-')
		 plt.xlabel('Hours studied')
		 plt.ylabel('Hours slept')
		 plt.legend(['Approved', 'Suspended'])
		 plt.show()
		"""
# Plotting the decision boundary
# X_2 = -(W_1/W_2)*X_1 - (W_0/W_2)*X_0
# Plot the decison boundary from [-2.0 to 2.0]  
# Setting up the exterme values on X-axis
x_start_end = np.array([-2.0, 2.0])

# The Weights calculated by the Gradient Descent algorithm
W_arr = W.eval(sess)

#X2_left = -1*(W_arr[1]/W_arr[2])*(-2.0) - (W_arr[0]/W_arr[2])
#X2_right = -1*(W_arr[1]/W_arr[2])*(2.0) - (W_arr[0]/W_arr[2])

X2_left  = -1*(W_arr[1]/W_arr[2])*(x_start_end[0]) - (W_arr[0]/W_arr[2])
X2_right = -1*(W_arr[1]/W_arr[2])*(x_start_end[1]) - (W_arr[0]/W_arr[2])

y_start_end = np.array([X2_left, X2_right])

# Plot the decision boundary over the data points 
pos = np.where(y_train == 1)
neg = np.where(y_train == 0)
plt.scatter(x_train[pos, 1], x_train[pos, 2], marker='o', c='b')
plt.scatter(x_train[neg, 1], x_train[neg, 2], marker='x', c='r')
plt.plot(x_start_end, y_start_end, 'g-')
plt.xlabel('Hours studied')
plt.ylabel('Hours slept')
plt.legend(['Approved', 'Suspended'])
plt.show()

# plot the error after each iteration of  Gradient Decent 
plt.plot(cost_val)
plt.show()

# Evaluate the prediction accuracy
# the bit of magix is that you provide X: x_test, Y: y_test 
# and TF knows it needs to pass x_test to z and y_test to 
# to accuracy 
print(sess.run(accuracy, feed_dict = {X: x_test, Y: y_test}))

sess.close()
