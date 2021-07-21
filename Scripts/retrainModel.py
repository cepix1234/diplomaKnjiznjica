from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


import os
import numpy as np
#import matplotlib.pyplot as plt
#from Pillow import Image
#import cv2
#import re  
import tensorflow_hub as hub
import gc

try:
	with tf.device('/device:GPU:0'):
		#get all directories of generated images
		pathToDiplomaDir = os.path.join("/storage/user/blazo-workspace/Diploma/Pictures/SlikeZaUcenje/diplomaKnjiznjica")
		pathGeneratedImages = os.path.join(pathToDiplomaDir, "224 new generator")
		allBooks = os.listdir(pathGeneratedImages)


		#filter out files inside the generated images directory 
		CLASS_NAMES  = []
		for book in allBooks:
			dir = os.path.join(pathGeneratedImages, book)
			if os.path.isdir(dir) and book != ".git":
				CLASS_NAMES .append(book)

		number_of_books = sum([len(files) for r, d, files in os.walk(pathGeneratedImages)]) - 3

		validation_size = 0.3
		# start the model training and classification
		batch_size = 5
		shuffleSize= 1000
		epochs = 10
		IMG_HEIGHT = 224
		IMG_WIDTH = 224


		#modul_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_large/classification/4", trainable= True)
		#modul_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4", trainable= True)
		#modul_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_large/classification/5", trainable= True)
		modul_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/4", trainable= True)
						
		
		# image_size = tuple(
		#       modul_layer._func.__call__  # pylint:disable=protected-access
		#       .concrete_functions[0].structured_input_signature[0][0].shape[1:3])


		dataflow_kwargs = dict(target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=batch_size,
				         interpolation="bilinear")

		image_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split = validation_size)

		train_data_gen = image_generator.flow_from_directory(directory=str(pathGeneratedImages),
				                                     shuffle=True,
				                                     subset = "training",
				                                     **dataflow_kwargs)


		validation_data_gen = image_generator.flow_from_directory(directory=str(pathGeneratedImages),
				                                     shuffle=True,
				                                     subset = "validation",
				                                     **dataflow_kwargs)


		data_augmentation = tf.keras.Sequential([
			tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
			tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
		])
		print("Class names: "+ ', '.join(CLASS_NAMES))

		model = tf.keras.Sequential([
			modul_layer,
			tf.keras.layers.Dropout(rate=0.2),
			tf.keras.layers.Dense(
			    len(CLASS_NAMES),
			    activation="softmax",
			    kernel_regularizer=tf.keras.regularizers.l2(0.0001))
		])

		#model = tf.keras.Sequential([
		#	#tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
		#	tf.keras.layers.Conv2D(16, 3, padding = "same", input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
		#	tf.keras.layers.MaxPooling2D(),
		#	tf.keras.layers.Conv2D(32, 3, padding = "same"),
		#	tf.keras.layers.MaxPooling2D(),
		#	tf.keras.layers.Conv2D(64, 3, padding = "same"),
		#	tf.keras.layers.MaxPooling2D(),
		#	tf.keras.layers.Dropout(rate=0.2),
		#	tf.keras.layers.Flatten(),	
		#	tf.keras.layers.Dense(128, activation = "relu"),
		#	tf.keras.layers.Dense(
		#	    len(CLASS_NAMES))
		#])

		# model = tf.keras.Sequential([
		#         tf.keras.layers.Conv2D(64, 32, strides=3 ,padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
		#         tf.keras.layers.MaxPooling2D(),
		#         tf.keras.layers.Conv2D(128, 32, strides=3 ,padding='same', activation='relu'),
		#         tf.keras.layers.MaxPooling2D(),
		#         tf.keras.layers.Flatten(),
		#         tf.keras.layers.Dense(530, activation='relu'),
		#         tf.keras.layers.Dropout(rate=0.2),
		#         tf.keras.layers.Dense(
		#             len(CLASS_NAMES),
		#             activation="softmax",
		#             kernel_regularizer=tf.keras.regularizers.l2(0.0001))
		#   ])


		model.build((None, IMG_HEIGHT, IMG_WIDTH, 3))
		print(model.summary())

		loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
		model.compile(
		    optimizer=tf.keras.optimizers.SGD(
			lr=0.00082, momentum=0.9),
		    loss=loss,
		    metrics=["accuracy"])
		model.summary()
		steps_per_epoch = train_data_gen.samples // batch_size
		validation_steps = validation_data_gen.samples // batch_size
		

		#at the end of every epoch clear garbage
		class MyCustomCallback(tf.keras.callbacks.Callback):
		  def on_epoch_end(self, epoch, logs=None):
		    gc.collect()


		model.fit(
		    train_data_gen,
		    shuffle=True,
		    epochs=epochs,
		    steps_per_epoch=steps_per_epoch,
		    validation_data=validation_data_gen,
		    validation_steps=validation_steps,
		    batch_size = 5,
		    callbacks=[MyCustomCallback()])

		model.save("InveptionV3-picturesWithoutHands.h5")

		
		
except RuntimeError as e:
	print(e)
