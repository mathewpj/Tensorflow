#	Author : mathew.p.joseph@gmail.com

#import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data

#
# Load the mnist data
#
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

#
# Define the neural network topology and processing parameters
#
batch_size = 100
image_height = 28 # MNIST images are 28x28 square
flatenned_input_data_size = image_height * image_height
n_classes = 10 # 1 for each digit in a one-hot configuration

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#
# Create hidden layers based on the 
# input list of node sizes and produce
# the neural network model.
#
def model_neural_network(data):

	hidden_1_layer_weights = tf.Variable(tf.random_normal([784, n_nodes_hl1]))
	hidden_1_layer_biases  = tf.random_normal([n_nodes_hl1])

        hidden_2_layer_weights = tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]))
	hidden_2_layer_biases  = tf.random_normal([n_nodes_hl2])

        hidden_3_layer_weights = tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]))
	hidden_3_layer_biases  = tf.random_normal([n_nodes_hl3])

        output_layer_weights   = tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])) 
	output_layer_biases    = tf.random_normal([n_classes])

        # (input_data X weights) + biases
        l1 = tf.add(tf.matmul(data, hidden_1_layer_weights), hidden_1_layer_biases)
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer_weights), hidden_2_layer_biases)
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer_weights), hidden_3_layer_biases)
        l3 = tf.nn.relu(l3)

        output = tf.matmul(l3, output_layer_weights)

        return(output)

x = tf.placeholder('float', [None, flatenned_input_data_size]) # Enforce the shape of the input)
y = tf.placeholder('float')

# For some unimaginable reason, the model_neural_network 
# is the equivalent of the hypotesis function 
prediction = model_neural_network(x)

# Calculates the difference between the prediction that we got and the tagged data 
cost_func = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
cost = tf.reduce_mean(cost_func)


# if you want to break down the monolithic "minimize" operation,
# you can explicitly calculate the gradients for the Optimizer
# & apply the gradients (then uncomment the following lines)
# & the  _, c = sess.run([train, cost], feed_dict={x: epoch_x, y: epoch_y}) in 
# the for loop   
#optimizer= tf.train.AdamOptimizer(learning_rate=0.001)
#grads = optimizer.compute_gradients(cost)
#train = optimizer.apply_gradients(grads)

# default learning_rate=0.001
optimizer= tf.train.AdamOptimizer().minimize(cost)

# You could use a Gradient Descent Optimizer, however the accuracy comes down
#optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

# cycles of FF + BP
hm_epochs = 10
sess = tf.Session()

sess.run(tf.global_variables_initializer())

#
# Train the network
#
for epoch in range(hm_epochs):
    epoch_loss = 0
    print('Starting epoch', epoch)
    for _ in range(int(mnist.train.num_examples / batch_size)):
        epoch_x, epoch_y = mnist.train.next_batch(batch_size)
	# By running the optimizer, it adjusts the "weights" and "biases" in the nn
	# structure in order to minimize the loss between predicted and labelled data 
        _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
        #_, c = sess.run([train, cost], feed_dict={x: epoch_x, y: epoch_y})
        epoch_loss += c
    
    print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)

#
# Evaluate the model
#
# return index of maximum value and should match
correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
print('accuracy', accuracy.eval(session=sess, feed_dict = {x: mnist.test.images, y: mnist.test.labels }))

sess.close()

