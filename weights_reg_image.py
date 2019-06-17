
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

tbCallback_BASE = keras.callbacks.TensorBoard(log_dir="./logs/weights/base", histogram_freq=0, write_graph=True, write_images=True)
tbCallback_WEIGHT = keras.callbacks.TensorBoard(log_dir="./logs/weights/weights", histogram_freq=0, write_graph=True, write_images=True)

train_images = train_images / 255.0

test_images = test_images / 255.0

EPOCHS = 20

#BASELINE MODEL ------------------------------------------------------------------------------
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation='relu'),
	keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy', 'sparse_categorical_crossentropy'])

baseline_history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels), callbacks=[tbCallback_BASE])

#WEIGHT REG MODEL ------------------------------------------------------------------------------
weight_reg_model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
	keras.layers.Dense(10, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.001))
])

weight_reg_model.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy', 'sparse_categorical_crossentropy'])

weight_reg_history = weight_reg_model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels), callbacks=[tbCallback_WEIGHT])



def plot_history(histories, key='sparse_categorical_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.show()


plot_history([('baseline', baseline_history),
				('weight', weight_reg_history)

              ])



