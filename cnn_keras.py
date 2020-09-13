import numpy as np
import pickle
import cv2, os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils import plot_model
K.set_image_dim_ordering('tf') 			#Sets the value of the image dimension ordering convention ('th' or 'tf')
#It specifies the shape of the input as (1, img_rows, img_cols,batch) - i.e there's one color channel (gray scale) and it comes first.
#That's why requires Theano's dimension ordering "th".
# for tensorflow (batch,image_widht,image_height,channel)
#if you wish to use theano backend and have to use channels first configuration for image dimension ordering.



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'			#TF_CPP_MIN_LOG_LEVEL:3 = INFO, WARNING, and ERROR messages are not printed

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

def get_num_of_classes():
	return len(os.listdir('gestures/'))			#To get number of classes

image_x, image_y = get_image_size()			#Reading image size from gestures O label 100th image

def cnn_model():
	num_of_classes = get_num_of_classes()
	model = Sequential()
	model.add(Conv2D(32, (5,5), input_shape=(image_x, image_y, 1), activation='sigmoid'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
	model.add(Conv2D(64, (5,5), activation='sigmoid'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.6))
	model.add(Dense(num_of_classes, activation='softmax'))
	sgd = optimizers.SGD(lr=1e-2)			#lr=learning rate

	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	filepath="cnn_model_keras2.h5"

	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')  #Save the model after every epoch.
	#The ModelCheckpoint callback class allows you to define where to checkpoint the model weights,
	#how the file should named and under what circumstances to make a checkpoint of the model.

	#By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
	#verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
	#save_best_only: if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.

	callbacks_list = [checkpoint]  #Converting to list
	return model, callbacks_list

	#To understand model architecture, please go to model.png in main folder

def train():
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("test_images", "rb") as f:
		test_images = np.array(pickle.load(f))
	with open("test_labels", "rb") as f:
		test_labels = np.array(pickle.load(f), dtype=np.int32)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
	test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
	train_labels = np_utils.to_categorical(train_labels)     #np.utils.to_categorical is used to convert array of labeled data(from 0 to nb_classes-1) to one-hot vector.
	test_labels = np_utils.to_categorical(test_labels)       #one-hot vector = binary class matrix.

	model, callbacks_list = cnn_model()
	#plot_model(model, to_file='model.png')

	model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=50, batch_size=100, callbacks=callbacks_list)
	#One epoch consists of one full training cycle on the training set.
	# Once every sample in the set is seen, you start again - marking the beginning of the 2nd epoch
	#Batch size defines number of samples that going to be propagated through the network.
	scores = model.evaluate(test_images, test_labels, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))
	#model.save('cnn_model_keras2.h5')

train()
