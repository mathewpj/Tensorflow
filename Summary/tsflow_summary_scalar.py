# Example code to create a summary to monitor a scalar variable 

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

x_scalar = tf.compat.v1.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
#Step1 : Create a scalar summary
first_summary = tf.summary.scalar(name = 'My_first_scalar_summary', tensor=x_scalar)

init = tf.initialize_all_variables()

sess = tf.Session()

#Step2 : Create the writer inside the session
writer = tf.compat.v1.summary.FileWriter('./graphs', sess.graph)


sess.run(init)
for step in range(100):
#Step3 : Evaluate the scalar summary
	summary = sess.run(first_summary)
#Step4 : Add the summary to the event file
	writer.add_summary(summary, step)

sess.close()





