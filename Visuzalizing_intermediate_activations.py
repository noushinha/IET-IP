import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras import models
from keras.applications import VGG16
import os

model = VGG16(include_top=False,
                      # weights='imagenet',
                      weights = '/media/Data/IET IP/Results/eNTERFACE/KMEANS/RESNET50_KMEANS/weights-improvement-094-0.94-0.17-0.60-1.60.hdf5',
                      input_shape=(150, 150, 3))

model.load_weights('/media/Data/IET IP/Results/RML/EigenFaces/20/VGG16_20/weights-improvement-020-0.93-0.16-0.92-0.19.hdf5', by_name=True)
print(model.summary())

img_path = '/media/Data/IET IP/Code/H_010_01.png'

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

plt.imshow(img_tensor[0])

#extracts the outputs of the top eight layer
layer_outputs = [layer.output for layer in model.layers][1:]
#creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
#returns a list of five numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)

#plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
#plt.show()

layer_names = []
for layer in model.layers:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    if 'conv' not in layer_name:
        continue
    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            if channel_image.std() != 0:
                channel_image /= channel_image.std()
            else:
                channel_image /= 1.
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.tight_layout()
    plt.imshow(display_grid, aspect='auto', cmap='bone')
    filename = os.path.join('/media/Data/IET IP/Materials', layer_name + ".eps")
    plt.savefig(filename, format='eps', dpi=80, bbox_inches="tight")

# plt.show()