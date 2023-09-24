import os
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import  ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

#include_top refers to including / or not the
# densely connected classifier on the top of the network
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))
print(conv_base.summary())

base_dir  = '/home/deeplearning/PycharmProjects/fchallote/cats_and_dogs_small/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
epoch = 5
batch_size = 16
image_width = 150
image_height = 150

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory=directory,
                                            target_size=(image_width, image_height),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(directory=train_dir, sample_count=2000)
validation_features, validation_labels = extract_features(directory=validation_dir, sample_count=1000)
test_features, test_labels = extract_features(directory=test_dir, sample_count=1000)

train_features = np.reshape(train_features, (2000,  4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000,  4 * 4 * 512))
test_features = np.reshape(test_features, (1000,  4 * 4 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim= 4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=epoch, batch_size=batch_size,
                    validation_data=(validation_features, validation_labels))

train_acc  = history.history['acc']
train_loss = history.history['loss']
val_acc  = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1, len(train_acc) + 1)

plt.plot(epochs, train_acc, 'go', label='Training Accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, train_loss, 'go', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()