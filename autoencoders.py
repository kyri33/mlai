import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import matplotlib.pyplot as plt


def plot_autoencoder_outputs(autoencoder, n, dims):
    decoded_imgs = autoencoder.predict(x_test)

    n = 5
    plt.figure(figsize = (10, 4.5))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(*dims))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n / 2:
            ax.set_title('Original Images')
        
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(*dims))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n / 2:
            ax.set_title("Reconstructed Images")
    plt.show()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))


input_size = x_train.shape[1]
hidden_size = 128
code_size = 32

'''
# SHALLOW AUTOENCODER
input_shallow = Input(shape=(input_size,))
code_shallow = Dense(code_size, activation='relu')(input_shallow)
output_shallow = Dense(input_size, activation='sigmoid')(code_shallow)
model_shallow = Model(input_shallow, output_shallow)
model_shallow.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_shallow.fit(x_train, x_train, epochs=5)

plot_autoencoder_outputs(model_shallow, 5, (28, 28))

# Deep Autoencoder

input_deep = Input(shape=(input_size,))
hidden_1_deep = Dense(hidden_size, activation='relu')(input_deep)
code_deep = Dense(code_size, activation='relu')(hidden_1_deep)
hidden_2_deep = Dense(hidden_size, activation='relu')(code_deep)
output_deep = Dense(input_size, activation='sigmoid')(hidden_2_deep)

autoencoder = Model(input_deep, output_deep)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(x_train, x_train, epochs=5)

plot_autoencoder_outputs(autoencoder, 5, (28, 28))
'''
# Denoising Autoencoder

noise_factor = 0.4
x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

n = 5
plt.figure(figsize=(10, 4.5))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n / 2:
        ax.set_title("Original Images")

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n / 2:
        ax.set_title("Noisy Input")
plt.show()

input_noisy = Input(shape=(input_size,))
hidden_1_noisy = Dense(hidden_size, activation='relu')(input_noisy)
code_noisy = Dense(code_size, activation='relu')(hidden_1_noisy)
hidden_2_noisy = Dense(hidden_size, activation='relu')(code_noisy)
output_noisy = Dense(input_size, activation='sigmoid')(hidden_2_noisy)
model_noisy = Model(input_noisy, output_noisy)
model_noisy.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_noisy.fit(x_train_noisy, x_train, epochs=10)

n = 5
plt.figure(figsize=(10, 7))
images = model_noisy.predict(x_test_noisy)

for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n / 2:
        ax.set_title("Original Images")

    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n / 2:
        ax.set_title('Noisy Input')

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n / 2:
        ax.set_title('Autoencoder Output')

plt.show()