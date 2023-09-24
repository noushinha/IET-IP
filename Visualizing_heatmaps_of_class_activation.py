import numpy as np
# from matplotlib import pyplot
import cv2
import os
from keras.applications import VGG16, ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import backend as K
from keras import layers
# from keras.models import Model
from keras import models
from keras import initializers
from keras.regularizers import l2


# modeltype = 'EigenFaceNet'
modeltype = 'VGG-16'
# modeltype = 'ResNet50'

save_dir = "/media/Data/IET IP/Materials/Heatmaps/" + modeltype + "/Fear/"
base_dir = "/media/Data/IET IP/Materials/Heatmaps/" + modeltype + "/Fear/"

model = VGG16()
print(model.summary())
model.load_weights('/media/Data/IET IP/Results/RML/EigenFaces/20/VGG16_20/weights-improvement-020-0.93-0.16-0.92-0.19.hdf5', by_name=True)

# model = ResNet50()
# print(model.summary())
# model.load_weights('/media/Data/IET IP/Results/eNTERFACE/KMEANS/RESNET50_KMEANS/weights-improvement-094-0.94-0.17-0.60-1.60.hdf5', by_name=True)

# model = models.Sequential()
# model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1),
#                                input_shape=(150, 150, 3),
#                                kernel_initializer='random_uniform', activation="relu", name="conv_1",
#                                kernel_regularizer=l2(0.0), bias_initializer=initializers.Zeros(), padding='SAME'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(3, 3), padding='VALID', name="maxpool_1"))
# model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1),
#                                kernel_initializer='random_uniform', activation="relu", name="conv_2",
#                                kernel_regularizer=l2(0.0), bias_initializer=initializers.Zeros(), padding='SAME'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(3, 3), padding='VALID', name="maxpool_2"))
# model.add(layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1),
#                                kernel_initializer='random_uniform', activation="relu", name="conv_3",
#                                kernel_regularizer=l2(0.0), bias_initializer=initializers.Zeros(), padding='SAME'))
# model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(3, 3), padding='VALID', name="maxpool_3"))
# model.add(layers.Dropout(0.5, name="dropout"))
# model.add(layers.Flatten(name="flatten"))
# model.add(layers.Dense(64, activation='relu', name="fc4"))
# model.add(layers.Dense(6, activation='softmax'))
# print(model.summary())
# # model.load_weights('/media/Data/IET IP/Results/RML/EigenFaces/30/eigenFaceNet_30/weights-improvement-197-1.00-0.01-0.89-0.67.hdf5', by_name=True)
# model.load_weights('/media/Data/IET IP/Results/RML/EigenFaces/ALL/eigenFaceNet_ALL/weights-improvement-176-1.00-0.01-0.85-0.83.hdf5', by_name=True)

# Local path to the target image
img_path = os.path.join(base_dir,'F_042_01.png')

# Python Image Library (PIL) image of size 224 x 224 for VGGT16
img = image.load_img(img_path, target_size=(224, 224))

# Converting to float32 nNumpy Array of shape (224, 224, 3)
x = image.img_to_array(img)

# Adds a dimension to transform the array into a batch of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Preprocesses the batch(this does channel-wise color normalization)
x = preprocess_input(x)

preds = model.predict(x)
# print('predicted: ', decode_predictions(preds, top=3)[0])
# square = 8
# ix = 1
# for _ in range(square):
# 	for _ in range(square):
# 		# specify subplot and turn of axis
# 		ax = pyplot.subplot(square, square, ix)
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		# plot filter channel in grayscale
# 		pyplot.imshow(preds[0, :, :, ix-1], cmap='bone')
# 		ix += 1
# # show the figure
# pyplot.show()
# finding the index of african elephant
# print(np.argmax(preds[0]))

# African Elephant entry in the prediction vector
african_elephant_output = model.output[:, np.argmax(preds[0])]

# Output feature map of the blockj5_conv3 layer, the last convolutional layer in VGG16
mylayer = 'block5_conv2'
# mylayer = 'res5c_branch2b'
# mylayer = "conv_3"
last_conv_layer = model.get_layer(mylayer)

# Gradient of the "African elephant" class with regard to the output feature map of block5_conv3
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# Vector of shape (512,), where each entry is the mean intensity of the gradient over a specific feature-map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# Lets me access the values of the quantities you just defined:
# pooled_grads and the output feature map of block5_conv3, givena  sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# Values of these two quantities, as Numpy arrays, given the sample image of two elephants
pooled_grad_value, conv_layer_output_value = iterate([x])

layerfilter = 512
for i in range(layerfilter):
    conv_layer_output_value[:, :, i] *= pooled_grad_value[i]

heatmap = np.mean(conv_layer_output_value, axis =-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# pyplot.matshow(heatmap)
# pyplot.show()

# Loading the original image using cv2
img = cv2.imread(img_path)

# Resizes the heatmap to be the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# Converts the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# Applies the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# heatmap intensity factor is 0.4
superimposed_img = heatmap * 0.4 + img

# Saving the image into the disk
cv2.imwrite(os.path.join(save_dir, modeltype + '_F_042_01_heatmap_' + mylayer + '_' + str(layerfilter) + '.png'), superimposed_img)
