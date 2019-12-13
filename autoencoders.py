import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_text = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))


input_size = x_train.shape[1]
hidden_size = 128
code_size = 32


# SHALLOW AUTOENCODER
input_shallow = Input(shape=(input_size,))
code_shallow = Dense(code_size, activation='relu')(input_shallow)
output_shallow = Dense(input_size)(code_shallow)
model_shallow = Model(input_shallow, output_shallow)
model_shallow.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_shallow.fit(x_train, x_train, epochs=5)

# Deep Autoencoder

input_img = Input(shape=(input_size,))
hidden_1 = Dense(hidden_size, activation='relu')(input_img)
code = Dense(code_size, activation='relu')(hidden_1)
hidden_2 = Dense(hidden_size, activation='relu')(code)
output_img = Dense(input_size, activation='relu')(hidden_2)

autoencoder = Model(input_img, output_img)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(x_train, x_train, epochs=3)