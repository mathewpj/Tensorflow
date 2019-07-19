# Example code to create a summary of type image  

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

w_gs = tf.compat.v1.get_variable('W_Grayscale', shape=[30, 10], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
w_c = tf.compat.v1.get_variable('W_Color', shape=[50, 30], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

#Step1 : Reshape it into 4D-tensors
w_gs_reshaped  = tf.reshape(w_gs, (3, 10, 10, 1))
w_c_reshaped   = tf.reshape(w_c, (5, 10, 10, 3))

#Step2 : Create the summaries  
gs_summary = tf.summary.image('Grayscale', w_gs_reshaped)
c_summary  = tf.summary.image('Color', w_c_reshaped, max_outputs=4)

#Step3 : Merge all summaries
merged = tf.summary.merge_all()

init = tf.initialize_all_variables()

sess = tf.Session()

#Step4 : Create the writer inside the session
writer = tf.compat.v1.summary.FileWriter('./graphs', sess.graph)

sess.run(init)

#Step5 : Evaluate the merged op to get the summaries
summary = sess.run(merged)

#Step6 : Add the summary to the event file
writer.add_summary(summary)

sess.close()





