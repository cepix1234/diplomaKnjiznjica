import tensorflow as tf

import os
import numpy as np
import tensorflow_hub as hub
import gc

#Example of how to RUN on GPU no 1

#TODO---------------------------------------------
try:
	with tf.device('/device:GPU:0'):
		a = tf.constant([[1.0,2.0,3.0], [4.0, 5.0, 6.0]])
		b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
		print ("Rezultat: ", tf.matmul(a, b))
except RuntimeError as e:
	print(e)