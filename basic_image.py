
# Learning From: https://www.tensorflow.org/beta/tutorials/keras/basic_classification

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow: ", tf.__version__)
print("Keras: ", keras.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

EPOCHS = 20

tbCallback_BASE = keras.callbacks.TensorBoard(log_dir="./logs/size/base", histogram_freq=0, write_graph=True, write_images=True)
tbCallback_SMALL = keras.callbacks.TensorBoard(log_dir="./logs/size/smaller", histogram_freq=0, write_graph=True, write_images=True)
tbCallback_LARGE = keras.callbacks.TensorBoard(log_dir="./logs/size/larger", histogram_freq=0, write_graph=True, write_images=True)
tbCallback_ADDITIONAL = keras.callbacks.TensorBoard(log_dir="./logs/size/additional", histogram_freq=0, write_graph=True, write_images=True)
#BASELINE MODEL ------------------------------------------------------------------------------
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation='relu'),
	keras.layers.Dense(128, activation='relu'),
	keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy', 'sparse_categorical_crossentropy'])

baseline_history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels), callbacks=[tbCallback_BASE])

#SMALLER MODEL ------------------------------------------------------------------------------
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(64, activation='relu'),
	keras.layers.Dense(64, activation='relu'),
	keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy', 'sparse_categorical_crossentropy'])

smaller_history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels), callbacks=[tbCallback_SMALL])

#LARGER MODEL ------------------------------------------------------------------------------
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(254, activation='relu'),
	keras.layers.Dense(254, activation='relu'),
	keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy', 'sparse_categorical_crossentropy'])

larger_history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels), callbacks=[tbCallback_LARGE])

#ADDITIONAL LAYER MODEL ------------------------------------------------------------------------------
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation='relu'),
	keras.layers.Dense(128, activation='relu'),
	keras.layers.Dense(128, activation='relu'),
	keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy', 'sparse_categorical_crossentropy'])

baseline_history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels), callbacks=[tbCallback_ADDITIONAL])


