# Good example to show summary histogram 

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np

raw_data = np.random.normal(10, 1, 100)
alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.0)

update_avg = alpha*curr_value + (1 - alpha)*prev_avg

#Step1 : Create a scalar summary
avg_hist = tf.compat.v1.summary.scalar('running_average', update_avg)
value_hist = tf.compat.v1.summary.scalar('incomming value', curr_value)

# A new step is added to merge all the summaries
merged = tf.summary.merge_all()

init = tf.initialize_all_variables()

sess = tf.Session()

#Step2 : Create the writer inside the session
writer = tf.compat.v1.summary.FileWriter('./graphs', sess.graph)

sess.run(init)

for i in range(len(raw_data)):
#Step3 : Evaluate the scalar summary
	summary, curr_avg  = sess.run([merged, update_avg], feed_dict={curr_value: raw_data[i]})
	sess.run(tf.assign(prev_avg, curr_avg))

#Step4 : Add the summary to the event file
	writer.add_summary(summary, i)

sess.close()





