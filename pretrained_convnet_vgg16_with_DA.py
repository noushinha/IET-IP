import os
import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras.optimizers import *
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator


#variable definition
epoch = 50
batch_size = 16
image_width  = 112
image_height = 112

#data lodaing and preparation
base_dir  = '/media/deepmind/Data/Datasets/CKPlus_AllFrames/'
train_dir = os.path.join(base_dir, 'train')
vald_dir  = os.path.join(base_dir, 'validation')
test_dir  = os.path.join(base_dir, 'test')

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(image_width, image_height, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(7, activation='sigmoid'))

print(model.summary())
print('Number of trainable Weights before freezing the conv base: ', len(model.trainable_weights))
conv_base.trainable = False
print('Number of trainable Weights after freezing the conv base: ', len(model.trainable_weights))

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        directory=vald_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='categorical')
optimizer = Adam(lr=1e-05, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=epoch,
    validation_data=validation_generator,
    validation_steps=50)

train_acc = history.history['acc']
train_loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1, len(train_acc) + 1)

plt.plot(epochs, train_acc, 'k-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, train_loss, 'k-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

test_generator = test_datagen.flow_from_directory(
				test_dir,
				target_size=(image_width, image_height),
				batch_size=16,
				class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test_acc: ', test_acc)
