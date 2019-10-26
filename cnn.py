import tensorflow as tf
from tensorflow import keras

root_dir = '/Users/kyriakosioulianou/'

model = keras.Sequential()
model.add(keras.layers.Convolution2D(32, 3, input_shape=(64, 64, 3), activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2)))
model.add(keras.layers.Convolution2D(32, 3, activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# Augmenting the images so that we can have batches of the same data but augmented in order to enrich the dataset without adding more images

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, 
                    shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(root_dir + 'Downloads/catdog/dataset/training_set', 
                        target_size=(64, 64),
                        batch_size = 32, class_mode='binary')

test_set = test_datagen.flow_from_directory(root_dir + 'Downloads/catdog/dataset/test_set',
                        target_size=(64, 64),
                        batch_size=32,
                        class_mode='binary')
model.fit_generator(training_set, steps_per_epoch=4000, epochs=1,
                    validation_data = test_set, validation_steps=1000)
