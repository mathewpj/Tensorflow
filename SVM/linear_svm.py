# Author mathew.p.joseph@gmail.com
# reworked the  example from Tensor Flow Cookbook

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Set random seeds
np.random.seed(7)
tf.set_random_seed(7)


# Load the data
# The iris data sets consists of 3 different types of irises (Setosa, Versicolour, and Virginica).
# Sepal & Petal datta is stored in a 150x4 numpy.ndarray
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
# iris.target is classified as 0 = setosa, 1 = versicolor, 2 = virginica
iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y==0 else -1 for y in iris.target])
# y_vals will now have 1 for entries if type setosa and -1 for entries
# that are versicolor or virginica
# x_vals will be of dimension 150 X 2 and y_vals 150 X 1


# Pick 135 random entries out of the data set for traininbg and reminder 
# 15 for testing 
train_indices = np.random.choice(len(x_vals), int(len(x_vals)*0.9), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

# Assign the 135 X 2 entries to x_vals_train and corresponding "y" i.e., 
# 15 X 1 to y_vals_train. the reminder 15 X 2 to x_vals_test etc  
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

batch_size = 135

# Setting up the data variables
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[2,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Hypothesis
# is of the form y = Ax - b 
# for 2D data this is a line, for 3D data it is a plane
# Generalizing for nD data it is hyperpane 
hypothesis = tf.subtract(tf.matmul(x_data, A), b)

# Cost Function
# 1/n*SUM(max(0, 1 - y*(Ax - b))) + alpha*MOD(A)^2

alpha = tf.constant([0.01])
# y*(Ax - b)
term1 = tf.multiply(hypothesis, y_target)
# 1 - y*(Ax - b)
term2 = tf.subtract(1.0, term1)
# max(0, 1 - y*(Ax - b))
term3 = tf.maximum(0.0, term2)
# 1/n*SUM(max(0, 1 - y*(Ax - b)))
term4 = tf.reduce_mean(term3)
# 1/n*SUM(max(0, 1 - y*(Ax - b))) + alpha*MOD(A)^2
loss = term4 + tf.multiply(alpha, tf.reduce_sum(tf.square(A)))


'''
l2_norm = tf.reduce_sum(tf.square(A))
alpha = tf.constant([0.01])
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
'''

# The predistion whether a point is setosa or not, 
# once you have discovered the (A,b) from the train data,
# you plug in the test set and if its >=+1 , it is setosa. If is it <= -1 it is NOT setosa 
prediction = tf.sign(hypothesis)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

# Set up the optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = opt.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

loss_vec = []
train_accuracy = []
test_accuracy = []

for i in range(1000):
    # for every step of the Gradient descent, yvou are sending the same set of data points
    # but jumbling the order differently. Not sure if this is the ideal way
    # however since the data set is small have to be satisfied with doing that  	
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    train_acc_temp = sess.run(accuracy, feed_dict={
        x_data: x_vals_train,
        y_target: np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)

    test_acc_temp = sess.run(accuracy, feed_dict={
        x_data: x_vals_test,
        y_target: np.transpose([y_vals_test])})
    test_accuracy.append(test_acc_temp)

    if (i + 1) % 100 == 0:
        print('Step #{} A = {}, b = {}'.format(
            str(i+1),
            str(sess.run(A)),
            str(sess.run(b))
        ))
        print('Loss = ' + str(temp_loss))
		
# Get the coefficients of the line
[[a1], [a2]] = sess.run(A)
[[b]] = sess.run(b)
slope = -a2/a1
y_intercept = b/a1

# Extract x1 
x1_vals = [d[1] for d in x_vals]

# Get best fit line
best_fit = []
for i in x1_vals:
  best_fit.append(slope*i+y_intercept)


# Separate I. setosa
setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==1]
setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==1]
not_setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==-1]
not_setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==-1]

# Plot data and line
plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()

# Plot train/test accuracies
plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

sess.close()
