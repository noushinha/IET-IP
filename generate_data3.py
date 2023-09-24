import cv2
import os
from glob import glob
import numpy as np
import random
from PIL import Image
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

# Dataset = 'SAVEE'
# Dataset = 'eNTERFACE'
Dataset = 'RML'
# Dataset="Vox2"
base_dir = '/media/Data/Datasets/' + Dataset + '/Eigenfaces10/'
class_list = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

train = []  # images as arrays
train_seq = []  # image sequence for one subject

test = []
test_seq = []

validation = []
vald_seq = []

for i in range(0,6):

    PATH1 = os.path.abspath(os.path.join(base_dir, 'test', class_list[i]))
    # PATH1 = os.path.abspath(os.path.join(base_dir, 'train', class_list[i]))
    # PATH2 = os.path.abspath(os.path.join(base_dir, 'validation', class_list[i]))

    # #print(PATH)
    images = glob(os.path.join(PATH1, "*.png"))
    # images1 = glob(os.path.join(PATH1, "*.png"))
    # images2 = glob(os.path.join(PATH2, "*.png"))
    # images = images1 + images2

    images.sort(key=lambda f: int(filter(str.isdigit, f)))
    print(len(images))
    # print(images[1071:1080])
    label = class_list[i]
    categorical_label = i

    WIDTH = 96
    HEIGHT = 96

    seqIdx = 1

    for img in images:
        # get image name
        base = os.path.basename(img)
        SubjIdx = base.split("_")
        print(SubjIdx)

        # Read the original RGB image
        full_size_image = cv2.imread(img)

        # Resize the image to 96x96
        gray_image = cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)

        # RML Indices
        test_indices = [10, 15, 34, 42, 43, 58, 63, 71, 93, 94, 109, 112]


        # test_indices = []
        vald_indices = []

        # each subject has 9 frames, make sure all 9 frames are saved within one array
        if seqIdx < 11:
            test_seq.append(np.array(gray_image))
            # train_seq.append(np.array(gray_image))
        if seqIdx == 10:
            test.append([test_seq, categorical_label])
            test_seq = []
            # train.append([train_seq, categorical_label])
            # train_seq = []
            seqIdx = 0

        # go for next image of current/new subject
        seqIdx = seqIdx + 1


################### TRAIN ###################



###################### Spectrograms #################
###################### Spectrograms #################
###################### Spectrograms #################
################### TRAIN ###################
save_dir = '/media/Data/IET IP/Data/' + Dataset + '/Eigenfaces/10/'
# np.save(os.path.join(save_dir, "Face_train_DL_ordered.npy"), random.sample(train, len(train)))  # for shuffle data
# np.save(os.path.join(save_dir, "Face_train_DL_ordered.npy"), train)
# npytrain = np.load(os.path.join(save_dir, "Face_train_DL_ordered.npy"))
# #npytrain = np.load(os.path.join(save_dir, "Face_train_DL_ordered.npy"))
# print(npytrain.shape)
#
# train_data = np.array([i[0] for i in npytrain])
# np.save(os.path.join(save_dir, "Face_train_data_ordered.npy"), train_data)
# # np.save(os.path.join(save_dir, "Face_train_data_ordered.npy"), train_data)
# print(train_data.shape)
#
# train_label = np.array([i[1] for i in npytrain])
# train_label = np.expand_dims(train_label, axis=1)
# np.save(os.path.join(save_dir, "Face_train_label_ordered.npy"), train_label)
# # np.save(os.path.join(save_dir, "Face_train_label_ordered.npy"), train_label)
# print(train_label.shape)
################# TRAIN ###################

################# VALIDATION ##############
# np.save(os.path.join(save_dir, "Face_validation_DL_ordered.npy"), validation)
# # npyvald = np.load(os.path.join(save_dir, "Face_validation_DL_ordered.npy"))
# npyvald = np.load(os.path.join(save_dir, "Face_validation_DL_ordered.npy"))
# print(npyvald.shape)
#
# vald_data = np.array([i[0] for i in npyvald])
# # np.save(os.path.join(save_dir, "Face_validation_data_ordered.npy"), vald_data)
# np.save(os.path.join(save_dir, "Face_validation_data_ordered.npy"), vald_data)
# print(vald_data.shape)
#
# vald_label = np.array([i[1] for i in npyvald])
# vald_label = np.expand_dims(vald_label, axis=1)
# # np.save(os.path.join(save_dir, "Face_validation_label_ordered.npy"), vald_label)
# np.save(os.path.join(save_dir, "Face_validation_label_ordered.npy"), vald_label)
# print(vald_label.shape)
# ################## VALIDATION ##############
#
# # ################## TEST ####################
np.save(os.path.join(save_dir, "Face_test_DL_ordered.npy"), test)
#np.save(os.path.join(save_dir, "Face_test_DL_ordered.npy"), random.sample(test, len(test)))  # for shuffle data
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

###################### Spectrograms ################################
###################### Spectrograms ################################
###################### Spectrograms ################################