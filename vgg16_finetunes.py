import os
import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
# from keras import optimizers
from keras.optimizers import *
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

# Variable Definition
base_dir  = '/media/deepmind/Data/Datasets/CKPlus_AllFrames/'
train_dir = os.path.join(base_dir, 'train')
vald_dir  = os.path.join(base_dir, 'validation')
test_dir  = os.path.join(base_dir, 'test')

base_dir2  = '/media/deepmind/Data/IET IP/Data/CKPlus_AllFrames/'
classes_num = 7
history_train_acc  = [] #np.zeros(shape= [k, epoch])
history_vald_acc   = [] #np.zeros(shape= [k, epoch])
history_train_loss = [] #np.zeros(shape= [k, epoch])
history_vald_loss  = [] #np.zeros(shape= [k, epoch])
history_lrr  = [] #np.zeros(shape= [k, epoch])
epoch_history_train_acc =[]
epoch_history_vald_acc =[]
epoch_history_train_loss =[]
epoch_history_vald_loss =[]
epoch_history_lrr = []

#load data
train_data = np.load(os.path.join(base_dir2, "Face_train_data_ordered.npy"))
vald_data  = np.load(os.path.join(base_dir2, "Face_validation_data_ordered.npy"))
test_data  = np.load(os.path.join(base_dir2, "Face_test_data_ordered.npy"))
print(train_data.shape)
print(vald_data.shape)
print(test_data.shape)

train_labels = np.load(os.path.join(base_dir2, "Face_train_label_ordered.npy"))
vald_labels  = np.load(os.path.join(base_dir2, "Face_validation_label_ordered.npy"))
test_labels  = np.load(os.path.join(base_dir2, "Face_test_label_ordered.npy"))

# convert to one hot coded
train_labels_one_hot_coded = to_categorical(y=train_labels, num_classes=classes_num)
vald_labels_one_hot_coded  = to_categorical(y=vald_labels, num_classes=classes_num)
test_labels_one_hot_coded  = to_categorical(y=test_labels, num_classes=classes_num)
print(train_labels_one_hot_coded.shape)
print(vald_labels_one_hot_coded.shape)
print(test_labels_one_hot_coded.shape)

# changing data type to avoid type errors
train_data = train_data.astype('float32')
vald_data  = vald_data.astype('float32')
test_data  = test_data.astype('float32')

epoch = 30
channel = 3
batch_size = 20
image_height = 200
image_width  = 200
steps_per_epoch = 100
validation_steps = 50

# Function Definition
# Smoothing the plots
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return  smoothed_points

# Convolutional Base Model
conv_base = VGG16(include_top=False,
                  weights='imagenet',
                  input_shape=(image_width, image_height, channel))


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(7, activation='sigmoid'))

print(model.summary())

# Freez Conv Base part of the model and train the Classifier part
conv_base.trainable = False

# Data Preprocess and Data Augmentation
train_datagen = ImageDataGenerator(
                     rotation_range=40,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     rescale=1./255,
                     fill_mode='nearest')
vald_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                     directory=train_dir,
                     target_size=(image_width,image_height),
                     class_mode='categorical',
                     batch_size=batch_size)

vald_generator = test_datagen.flow_from_directory(
                     directory=vald_dir,
                     target_size=(image_width, image_height),
                     class_mode='categorical',
                     batch_size=batch_size)

optimizer = Adam(lr=2e-05, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=0.0)
# Training the newly added part of the model (Top of the model, i.e Classifier)
model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=epoch,
            validation_data=vald_generator,
            validation_steps=50)


conv_base.trainable = True

set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

optimizer = Adam(lr=1e-05, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=epoch,
    validation_data=vald_generator,
    validation_steps=50)


train_acc  = history.history['acc']
train_loss = history.history['loss']
val_acc  = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1, len(train_acc) + 1)

test_generator = test_datagen.flow_from_directory(
				test_dir,
				target_size=(image_width, image_height),
				batch_size=1,
				class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test_acc: ', test_acc)


plt.plot(epochs, smooth_curve(train_acc), 'k-', label=('Training Accuracy'))
plt.plot(epochs, smooth_curve(val_acc), 'r-', label=('Validation Accuracy'))
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, smooth_curve(train_loss), 'k-', label=('Training Loss'))
plt.plot(epochs, smooth_curve(val_loss), 'r-', label=('Validation Loss'))
plt.title('Training and Validation Loss')
plt.legend()

plt.show()