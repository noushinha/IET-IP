import cv2
import os
from glob import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# Dataset = 'SAVEE'
# Dataset = 'eNTERFACE'
Dataset = 'RML'
Dataset = 'AFEW'
# Dataset = 'CKPlus_AllFrames'
# Dataset="Vox2"
base_dir = '/media/Data/Datasets/' + Dataset
save_dir = '/media/Data/Conferance/Data/' + Dataset
class_list = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise'] #'Contempt',

flag = 'train'

train = []
test = []
validation = []

for i in range(0,7):

    PATH = os.path.abspath(os.path.join(base_dir, flag, class_list[i]))
    #print(PATH)
    images = glob(os.path.join(PATH, "*.png"))
    images.sort(key=lambda f: int(filter(str.isdigit, f)))
    print(len(images))
    # print(images[1071:1080])
    label = class_list[i]
    categorical_label = i

    WIDTH = 200
    HEIGHT = 200

    seqIdx = 1

    for img in images:
        # get image name
        base = os.path.basename(img)
        SubjIdx = base.split("_")
        print(SubjIdx)

        # Read the original RGB image
        full_size_image = cv2.imread(img)
        #print(full_size_image.shape)

        # Resize the image to 96x96
        gray_image = cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)

        if flag == 'test':
            test.append([np.array(gray_image), categorical_label])
        elif flag == 'validation':
            validation.append([np.array(gray_image), categorical_label])
        else:
            train.append([np.array(gray_image), categorical_label])

    if flag == 'test':
        print("number of images in test: ", len(test))
    elif flag == 'validation':
        print("number of images in validation: ", len(validation))
    else:
        print("number of images in train: ", len(train))


# ################## TEST ####################
if flag == 'test':
    np.save(os.path.join(save_dir, "Face_test_DL_ordered.npy"), test)
    # np.save(os.path.join(save_dir, "Face_test_DL_ordered.npy"), random.sample(test, len(test)))  # for shuffle data
    # npytest = np.load(os.path.join(save_dir, "Face_test_DL_ordered.npy"))
    npytest = np.load(os.path.join(save_dir, "Face_test_DL_ordered.npy"))
    print(npytest.shape)

    test_data = np.array([i[0] for i in npytest])
    # np.save(os.path.join(save_dir, "Face_test_data_ordered.npy"), test_data)
    np.save(os.path.join(save_dir, "Face_test_data_ordered.npy"), test_data)
    print(test_data.shape)

    test_label = np.array([i[1] for i in npytest])
    test_label = np.expand_dims(test_label, axis=1)
    # np.save(os.path.join(save_dir, "Face_test_label_ordered.npy"), test_label)
    np.save(os.path.join(save_dir, "Face_test_label_ordered.npy"), test_label)
    print(test_label.shape)
################## TEST ####################

################# VALIDATION ##############
if flag == 'validation':
    np.save(os.path.join(save_dir, "Face_validation_DL_ordered.npy"), validation)
    # npyvald = np.load(os.path.join(save_dir, "Face_validation_DL_ordered.npy"))
    npyvald = np.load(os.path.join(save_dir, "Face_validation_DL_ordered.npy"))
    print(npyvald.shape)

    vald_data = np.array([i[0] for i in npyvald])
    # np.save(os.path.join(save_dir, "Face_validation_data_ordered.npy"), vald_data)
    np.save(os.path.join(save_dir, "Face_validation_data_ordered.npy"), vald_data)
    print(vald_data.shape)

    vald_label = np.array([i[1] for i in npyvald])
    vald_label = np.expand_dims(vald_label, axis=1)
    # np.save(os.path.join(save_dir, "Face_validation_label_ordered.npy"), vald_label)
    np.save(os.path.join(save_dir, "Face_validation_label_ordered.npy"), vald_label)
    print(vald_label.shape)
# ################## VALIDATION ##############

################### TRAIN ###################
if flag == 'train':
    # np.save(os.path.join(save_dir, "Face_train_DL_ordered.npy"), random.sample(train, len(train)))  # for shuffle data
    np.save(os.path.join(save_dir, "Face_train_DL_ordered.npy"), train)
    npytrain = np.load(os.path.join(save_dir, "Face_train_DL_ordered.npy"))
    #npytrain = np.load(os.path.join(save_dir, "Face_train_DL_ordered.npy"))
    print(npytrain.shape)

    train_data = np.array([i[0] for i in npytrain])
    np.save(os.path.join(save_dir, "Face_train_data_ordered.npy"), train_data)
    # np.save(os.path.join(save_dir, "Face_train_data_ordered.npy"), train_data)
    print(train_data.shape)

    train_label = np.array([i[1] for i in npytrain])
    train_label = np.expand_dims(train_label, axis=1)
    np.save(os.path.join(save_dir, "Face_train_label_ordered.npy"), train_label)
    # np.save(os.path.join(save_dir, "Face_train_label_ordered.npy"), train_label)
    print(train_label.shape)
################# TRAIN ###################



# augmented_data = []
# batches = 0
# train_datagen.fit(train_data)
# i = 0
# for X_batch, y_batch in train_datagen.flow(train_data, train_labels, batch_size=100, shuffle=False):
#     augmented_data.append(X_batch)
#     batches += 1
#     if batches > len(train_data) / 100:
#         plt.subplot(330 + 1 + i)
#         plt.imshow(augmented_data[i][0].reshape(200, 200, 3), cmap=plt.get_cmap('gray'))
#         # show the plot
#         plt.show()
#         break
#
# augmented_data = np.concatenate(augmented_data)
# np.save(os.path.join(base_dir, "Face_test_data_augmented_ordered.npy"), augmented_data)