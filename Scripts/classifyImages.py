from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


import os
import numpy as np
from PIL import Image
import cv2
import re  
import tensorflow_hub as hub


try:
	with tf.device('/device:GPU:3'):
		#get all directories of generated images
		pathToDiplomaDir = os.path.join("/opt/workspace/host_storage_hdd", "slike")
		#pathGeneratedImages = os.path.join(pathToDiplomaDir, "TestSlike224", "TestPictures2RanTrueUnity")
		pathGeneratedImages = os.path.join(pathToDiplomaDir, "SlikeZaTestiranje")
		allBooks = os.listdir(pathGeneratedImages)

		#filter out files inside the generated images directory 
		CLASS_NAMES  = []
		for book in allBooks:
			dir = os.path.join(pathGeneratedImages, book)
			if os.path.isdir(dir):
				CLASS_NAMES .append(book)

		number_of_books = sum([len(files) for r, d, files in os.walk(pathGeneratedImages)])

		validation_size = 0.2
		# start the model training and classification
		batch_size = 16
		shuffleSize= 1000
		epochs = 5
		IMG_HEIGHT = 224
		IMG_WIDTH = 224


		# modul_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_large/classification/4", trainable= True)
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
				                             shuffle=False,
				                             subset = "validation",
				                             **dataflow_kwargs)

		test_data_gen = image_generator.flow_from_directory(directory=str(pathGeneratedImages))


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


		model.build((None, IMG_HEIGHT, IMG_WIDTH, 3))
		print(model.summary())

		loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
		model.compile(
		optimizer=tf.keras.optimizers.SGD(
		lr=0.005, momentum=0.9),
		loss=loss,
		metrics=["accuracy"])
		steps_per_epoch = train_data_gen.samples // batch_size
		validation_steps = validation_data_gen.samples // batch_size

		print(model.summary())

		model.load_weights("InveptionV3-picturesWithoutHands.h5")
		model.summary()

		test_images_dict = test_data_gen.class_indices
		rev_test_images_dict = {v: k for k, v in test_images_dict.items()}
		predictions = []
		real_values = []

		def classifyImage (image_name, imagePath, model):
			#print("Predicting for image: " + image_name)
			image = tf.keras.preprocessing.image.load_img(imagePath, target_size=(IMG_HEIGHT,IMG_WIDTH))
			img_array = tf.keras.preprocessing.image.img_to_array(image)
			img_array = tf.expand_dims(img_array, 0)

			#image = cv2.imread(imagePath)
			#image = image.reshape(1,IMG_HEIGHT,IMG_WIDTH,3)
			#image = image.astype(float)
			predicted = model.predict(img_array)
			prediction = tf.nn.softmax(predicted[0])
			predictions.append(np.argmax(prediction))
			index = np.argmax(prediction)
			
			label = CLASS_NAMES[index]
			return label

		def countBookInMatrix(book_name, label):
			for book in bookMatrix:
				if(book[0].lower() == book_name.lower()):
					for bookLabel in book[1]:
						if(bookLabel[0].lower() == label.lower()):
							bookLabel[1] = bookLabel[1] + 1


		def getAccuracyOfBook(book_name, directory, model):
			allImages = os.listdir(directory)    
			classified = 0
			for image in allImages:
				imagePath = os.path.join(directory,image)
				label = classifyImage(book_name+"/"+image, imagePath, model)
				#print("Book: " + book_name + ", classified as: "+ label)

				real_values.append(test_images_dict[book_name])
				countBookInMatrix(book_name, label)
				if(book_name.lower() == label.lower()):
				    classified = classified + 1
			return (classified/ len(allImages))

		bookAccuracies = []
		# Create Matrix
		maxBookChar = 0
		bookMatrix = []

		for book_name in CLASS_NAMES:
			table = []
			for book_name_inner in CLASS_NAMES:
				table.append([book_name_inner, 0])
			bookMatrix.append((book_name, table))
			if(maxBookChar < len(book_name)):
				maxBookChar = len(book_name)



		for book_name in CLASS_NAMES:
			print("Predicting for Book: " + book_name)
			directory = os.path.join(pathGeneratedImages, book_name)
			accuracy = getAccuracyOfBook(book_name, directory, model)
			bookAccuracies.append((book_name, accuracy))


		def formatOutput(addComma):
			result = ""
			if(addComma):
				result = ","
			return result


		model.evaluate(test_data_gen)

		
		matrix = tf.math.confusion_matrix(real_values,predictions)

		#Print Book names
		print(",", end=formatOutput(False))
		for i in range(26):
			print(rev_test_images_dict[i], end=formatOutput(True))
		print("")

		#Print matrix
		row = 0
		unstacked = tf.unstack(matrix)
		for line in unstacked:
			print(rev_test_images_dict[row], end=formatOutput(True))
			row = row + 1
			unstacked_line = tf.unstack(line)
			for column in unstacked_line:
				print(tf.keras.backend.get_value(column), end=formatOutput(True))
			print("")
		

except RuntimeError as e:
	print(e)