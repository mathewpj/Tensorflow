# # Example code to create a summary to monitor a histogram 

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

x_scalar = tf.compat.v1.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
x_matrix = tf.compat.v1.get_variable('x_matrix', shape=[1000, 1000], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

#Step1 : Create a scalar summary
summary_scalar = tf.compat.v1.summary.scalar('My_scalar_summary', x_scalar)
summary_histogram = tf.compat.v1.summary.histogram('My_histogram_summary', x_matrix)

init = tf.initialize_all_variables()

sess = tf.Session()

#Step2 : Create the writer inside the session
writer = tf.compat.v1.summary.FileWriter('./graphs', sess.graph)


for step in range(100):
	sess.run(init)
#Step3 : Evaluate the scalar summary
	summary1, summary2 = sess.run([summary_scalar, summary_histogram])
#Step4 : Add the summary to the event file
	writer.add_summary(summary1, step)
	writer.add_summary(summary2, step)

sess.close()





