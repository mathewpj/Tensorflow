# Program that demonstrates multinomial logistic regression
# i.e., it classifies data into one of 'n' possibilities.
# here the data is to classify a data point into either 
# of of three types of irises based on 
# sepal length
# sepal_width
# petal_length
# petal_width
# the Hypothesis is a sigmoid function (as opposed to a softmax function)
# Author : mathew.p.joseph@gmail.com
 
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.model_selection import train_test_split

str = 'Prediction Accuracy'

data = pd.read_csv("iris.data", sep=",",
                   names=["sepal_length", "sepal_width", "petal_length", "petal_width", "iris_class"])


# Randomise the input data
# frac keyword argument specifies the fraction of rows to 
# return in the random sample, so frac=1 means return all 
# rows (in random order)
# drop=True prevents .reset_index from creating a column 
# containing the old index entries.
np.random.seed(0)
data = data.sample(frac=1).reset_index(drop=True)
print(data.head())

all_x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# Generate  one hot encoding for nominal label data.iris_class
# i.e., 
#	Iris-setosa  Iris-versicolor  Iris-virginica
#            0                0               1
#            0                1               0
#            1                0               0
all_y = pd.get_dummies(data.iris_class)


# Split the data in the ratio 20/80 
train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(all_x, all_y, test_size= 0.2)


# Insert a colum of 1's to the input data such that
# X_0 = 1, X_1 = sepal_length, X_2 = sepal_width, X_3 = petal_length, X_4 = petal_width
column_of_ones = np.ones((train_x.shape[0], 1))
train_x = np.column_stack((column_of_ones, train_x))
column_of_ones = np.ones((test_x.shape[0], 1))
test_x = np.column_stack((column_of_ones, test_x))
'''
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
'''
n_x = train_x.shape[1]
n_y = train_y.shape[1]

# Placeholder for data that wil be passed to the optimizer later 
X = tf.placeholder(tf.float32, shape=[None, n_x])
Y = tf.placeholder(tf.float32, shape=[None, n_y])

# Weights and Bias
W = tf.Variable(tf.zeros(shape=[n_x, n_y]))

# Hypothesis
# if X = 100 X 5, then W = 5 X 3 (W is a 5X3 matric and not a 5X1 because 
# one hot encoding is used for the Nominal label y (data.iris_class)
# O/P is 100 X 3
# compared to the sigmoid Hypothesis function I'm using a softmax hypothesis
# function. 
# The softmax function ensures that the probabilities of the o/p will all 
# add up to 1 
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# Cost Function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1- Y)*tf.log(1 - hypothesis)) 

# The Gradient Descent Algorithm
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Calculate the prediction accuracy
'''
tf.argmax returns the index with the largest value across axes of a tensor
pred = np.array([[31, 23,  4, 24, 27, 34],
                 [18,  3, 25,  0,  6, 35],
                 [28, 14, 33, 22, 20,  8],
                 [13, 30, 21, 19,  7,  9],
                 [16,  1, 26, 32,  2, 29],
                 [17, 12,  5, 11, 10, 15]])
tf.argmax(pred) equivalent to tf.argmax(pred, 0) gives [0 3 2 4 0 1]
 i.e., the index of maximum element in each column 
tf.argmax(pred, 1) gives [5, 5, 2, 1, 3, 0]
 i.e., the index of maximum element in each row 
'''
# If test_x is the reminder 50 samples, then the input to hypothesis
# are two arrays of 50X5 and 5X3 resulting in an op of 50X3
# Y will be a 50X3 array (because of one hot encoding
# so tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)) will return 1 in a case 
# hypothesis = [1.62678957e-03, 2.15468258e-01, 8.13180685e-01] and Y = [ 0 0 1]

prediction_is_correct = tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), tf.float32)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(10000):
    sess.run([train], feed_dict={X: train_x, Y: train_y})
#W_hat, b_hat = sess.run([W, b])
W_hat = sess.run(W)

#print(test_x[:10])
#print(test_y[:10])

#print(sess.run([hypothesis], feed_dict={X: test_x[:10], W: W_hat}))

print(str, sess.run(tf.reduce_mean(prediction_is_correct), feed_dict={W: W_hat, X: test_x, Y: test_y}))

sess.close()
