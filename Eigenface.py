from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.misc import imread
from itertools import chain
import numpy as np
# import scipy as sp
from glob import glob
import cv2
import os
# import pandas as pd
import time
import matplotlib.gridspec as gridspec
# import random

numsbj = 120
NUM_PCA_COMP = 20
IM_SIZE = 150


# Dataset = 'eNTERFACE'
# Dataset = 'RML'
Dataset = 'AFEW'

# base_dir = '/media/Data/Datasets/' + Dataset + '/Extracted_Frames_AU'
# save_dir = '/media/Data/Datasets/' + Dataset + '/Eigenfaces' + str(NUM_PCA_COMP) + '_AU/'
# save_dir = '/media/Data/Datasets/' + Dataset + '/ALLEigenfaces/'

# base_dir = '/media/Data/Datasets/' + Dataset + '/Extracted_Frames/Test'
# save_dir = '/media/Data/Datasets/' + Dataset + '/Eigenfaces' + str(NUM_PCA_COMP)

base_dir = '/media/Data/Datasets/' + Dataset + '/Extraction/train'
save_dir = '/media/Data/Datasets/' + Dataset + '/EigenFaces' + str(NUM_PCA_COMP)

class_list = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']  # 'Contempt',
class_list2 = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']  # 'Contempt',
abbr = ['A', 'D', 'F', 'H', 'SA', 'SU']

# nums = [x for x in range(210)]
# random.shuffle(nums)
# print(nums)
testindices = []
valdindices = []

if Dataset == 'RML':
    testindices = [10, 15, 34, 42, 43, 58, 63, 71, 93, 94, 109, 112]
    valdindices = [21, 32, 38, 51, 46, 59, 69, 77, 88, 91, 113, 119]
elif Dataset == 'eNTERFACE':
    testindices = np.sort([73, 22, 194, 12, 86, 31, 113, 25, 56, 16, 65, 147, 127, 185, 156, 197, 3, 205, 175, 27, 189])
    valdindices = np.sort([105, 157, 148, 34, 136, 4, 91, 78, 37, 117, 75, 72, 203, 69, 160, 130, 19, 114, 150, 96, 21])

# ncomp = [1, 3, 5, 7, 10, 13, 15, 17, 20, 25, 30]  # 1, 3, 5, 7, 10, 13, 15, 17, 20, 25, 30

n_components = 0
# fig, axs = plt.subplots(6,5, figsize=(15, 6), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace = .5, wspace=.001)
fig = plt.figure(constrained_layout=False, figsize=(8, 6))
gs1 = gridspec.GridSpec(nrows=6, ncols=5, wspace=0.04, hspace=0.04)
axs = []
counter = 0
cols = ['EF {}'.format(col) for col in range(1, 6)]
comp_times = []
# for c in range(len(ncomp)):
# numsbjs_afew = [133, 74, 80, 147, 113, 71]
# numsbjs_afew = [59, 39, 44, 63, 59, 46]

numsbjs_afew = [118, 73, 70, 145, 104, 69]
# numsbjs_afew = [60, 37, 39, 60, 57, 42]
mayub = []
indexmayub = []
start = time.clock()
for i in range(0, 6):
    # only for AFEW
    numsbj = numsbjs_afew[i]
    print("-------------------------------------------------------------------")
    folders = os.path.abspath(os.path.join(base_dir, class_list[i]))
    folders = glob(os.path.join(folders, "*"))
    folders.sort(key=lambda f: int(filter(str.isdigit, f)))
    for j in range(0, numsbj):
        NUM_PCA_COMP = 1
        # print(folders[j])
        # print(j)
        sbjstr = folders[j]
        aaa = folders[j].split('/')[8]  # for AFEW
        PATH = os.path.abspath(os.path.join(sbjstr, aaa + '_aligned'))
        # PATH = os.path.abspath(os.path.join(base_dir, class_list[i], sbjstr, sbjstr+'_aligned'))
        # sbjstr = abbr[i] + '_' + str(j+1) + '_aligned'
        # PATH = os.path.abspath(os.path.join(base_dir, class_list[i], str(j), sbjstr))

        images = glob(os.path.join(PATH, "*.png"))
        print(PATH, len(images))
        # if(len(images) == 0):
        #      print(PATH)
        #      continue
        images.sort(key=lambda f: int(filter(str.isdigit, f)))
        # print(images[1071:1080])
        label = class_list[i]
        categorical_label = i

        # filenames = [img for img in glob.glob(os.path.join(PATH, "*.png"))]
        m = [[] for d in range(len(images))]
        for k in range(len(images)):
            im = imread(images[k])
            im = cv2.resize(im, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_CUBIC)
            # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            m[k] = list(chain.from_iterable(im))

        m = np.matrix(m)
        # NUM_PCA_COMP = ncomp[c]
        model = PCA(n_components=NUM_PCA_COMP)
        # model = PCA()
        pts = normalize(m)
        model.fit(pts)
        # if len(model.components_) < NUM_PCA_COMP:
        #     NUM_PCA_COMP = len(model.components_)
        #     mayub.append(PATH)
        # else:
        #     NUM_PCA_COMP = 10
#         # for the purpose of visualization of eigen faces
#         for b in range(0,5):
#             ord = model.components_[b]
#             img = ord.reshape(IM_SIZE,IM_SIZE)
#             cursp = fig.add_subplot(gs1[counter])
#             axs.append(fig.add_subplot(gs1[counter]))
#             # cursp.axis('off')
#             if b == 0:
#                 cursp.annotate(class_list2[i], xy=(0, 0.5), xytext=(-cursp.yaxis.labelpad - 2, 0),
#                             xycoords=cursp.yaxis.label, textcoords='offset points',
#                             size=11, ha='right', va='center', rotation=90)
#             if i == 0:
#                 cursp.annotate(cols[b], xy=(0.5, 1), xytext=(0, 10),
#                             xycoords='axes fraction', textcoords='offset points',
#                             size=10, ha='center', va='baseline',  rotation=0)
#             cursp.set_xticks([])
#             cursp.set_yticks([])
#             axs[-1].imshow(img, cmap="bone")
#             counter = counter + 1
# plt.show()
        if(NUM_PCA_COMP > len(model.components_)):
            NUM_PCA_COMP = len(model.components_)
            mayub.append(PATH)
            indexmayub.append(j)

        # print(NUM_PCA_COMP)
        pts2 = model.transform(pts)
        for l in range(NUM_PCA_COMP):
            compsord = model.components_[l]
            gray = compsord.reshape(IM_SIZE, IM_SIZE)
            # gray = projected[l].reshape(150, 150)
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            # cv2.imwrite("projected.png", gray)
            img = np.zeros((IM_SIZE, IM_SIZE, 3))
            img[:, :, 0] = gray
            img[:, :, 1] = gray
            img[:, :, 2] = gray
            # cv2.imwrite("projected2.png", gray)
            # if j < 10:
            #     subject = "00" + str(j)
            # elif j <100:
            #     subject = "0" + str(j)
            # else:
            #     subject = str(j)
            #
            # if l < 10:
            #     eigenfacestr = "0" + str(l+1)
            # else:
            #     eigenfacestr = str(l+1)
            #
            # imgname = abbr[i] + "_" + subject + "_"  + eigenfacestr + ".png"
            #
            # if j in testindices:
            #     path1 = "test"
            # elif j in valdindices:
            #     path1 = "validation"
            # else:
            #     path1 = "train"
            #
            # thispath = os.path.join(save_dir, path1, class_list[i], imgname)
            # plt.imsave(thispath, img, cmap="gray")
            ef_num = str(l)
            if l < 10:
                ef_num = '0' + str(l)

            sbj_num = str(j)
            if j < 10:
                sbj_num = '00' + str(j)
            elif 10 <= j < 100:
                sbj_num = '0' + str(j)
            filename = abbr[i] + '_' + sbj_num + '_' + ef_num + '.png'

            # if j in testindices:
            #     savingpath = os.path.join(save_dir, 'test', class_list[i], filename)
            #     # cv2.imwrite(savingpath, gray)
            #     # test.append([np.array(img), categorical_label])
            # elif j in valdindices:
            #     savingpath = os.path.join(save_dir, 'validation', class_list[i], filename)
            #     # cv2.imwrite(savingpath, gray)
            #     # validation.append([np.array(img), categorical_label])
            # else:
            savingpath = os.path.join(save_dir, 'train', class_list[i], filename)
            cv2.imwrite(savingpath, gray)
            # train.append([np.array(img), categorical_label])
stop = time.clock()
comptime = stop - start
print(comptime)
comp_times.append(comptime)
print(comp_times)
print(mayub)
print(indexmayub)

# pd.DataFrame(variances).to_csv(os.path.join(base_dir, "pca_variance.csv"), header=False,index=False)
# datavariance = np.cumsum(model.explained_variance_ratio_)
# for b in range(len(datavariance)):
#     if(datavariance[b] >= 0.95):
#         n_components = n_components + b + 1
#         break

# stop = time.clock()
# process_time = (stop - start)
# avg = n_components / 720
# print("process time: ", process_time)
# print("average number of components: ", avg)


# # for the purpose of visualization of eigen faces
# fig, axes = plt.subplots(1,5, figsize=(12, 4))
# for i, ax  in enumerate(axes.flat):
#     ord = model.components_[i]
#     img = ord.reshape(IM_SIZE,IM_SIZE)
#     ax.imshow(img, cmap="bone")
#     ax.axis('off')
# plt.show()
# plt.figure()
# plt.plot(np.cumsum(model.explained_variance_ratio_))
# # plt.annotate('max inflation', xy=(year_max, np.cumsum(model.explained_variance_ratio_)))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance (%)')  # for each component
# plt.title('Pulsar Dataset Explained Variance')
# plt.savefig("VAriance_Number_of_Components", format='eps', dpi=800, bbox_inches="tight")
# plt.show()


# variances = model.explained_variance_
# # variances = np.transpose(variances)
# # pd.DataFrame(variances).to_csv("/media/Data/IET IP/Results/RML/
# EigenFaces/pca_variance.csv", header=False, index=False)
# # projected = model.inverse_transform(pts2)

# for the purpose of visualization of reconstructed faces
# faces_pca = PCA()
# faces_pca.fit(m)
# components = faces_pca.transform(m)
# projected = faces_pca.inverse_transform(components)
# fig, axes = plt.subplots(2, 5, figsize=(12, 4))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(projected[i].reshape(150, 150 ), cmap="gray")
#
# plt.show()
