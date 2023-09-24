from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.misc import imread
from itertools import chain
import numpy as np
import scipy as sp
from glob import glob
import cv2
import os
import pandas as pd
import time
import matplotlib.gridspec as gridspec
import random
from PIL import Image
import scipy.misc
import seaborn as sns
import operator

NUM_PCA_COMP = 20
IM_SIZE = 150

# Dataset = 'eNTERFACE'
# Dataset = 'RML'
Dataset = 'AFEW'
base_dir = '/media/Data/Datasets/' + Dataset + '/Extracted_Frames'
save_dir = '/media/Data/Datasets/' + Dataset + '/Eigenfaces' + str(NUM_PCA_COMP) + '_Average/'
class_list = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise'] #'Contempt',
class_list2 = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'] #'Contempt',
abbr = ['A', 'D', 'F', 'H', 'SA', 'SU']

# nums = [x for x in range(210)]
# random.shuffle(nums)
# print(nums)

if Dataset == 'RML':
    testindices = [10, 15, 34, 42, 43, 58, 63, 71, 93, 94, 109, 112]
    valdindices = [21, 32, 38, 51, 46, 59, 69, 77, 88, 91, 113, 119]
elif Dataset == 'eNTERFACE':
    testindices = np.sort([73, 22, 194, 12, 86, 31, 113, 25, 56, 16, 65, 147, 127, 185, 156, 197, 3, 205, 175, 27, 189])
    valdindices = np.sort([105, 157, 148, 34, 136, 4, 91, 78, 37, 117, 75, 72, 203, 69, 160, 130, 19, 114, 150, 96, 21,])

train = []
test = []
validation = []
n_components = 0
# avg_face = np.zeros((150,150,3),np.float)
# start = time.clock()

fig = plt.figure(constrained_layout=False, figsize=(6, 4))
gs1 = gridspec.GridSpec(nrows=4, ncols=6, wspace=0.02, hspace=0.04)
axs = []
counter = 0
cols = ['EF {}'.format(col-1) for col in range(1, 7)]
cols[0] = "Angry"
cols[1] = "Disgust"
cols[2] = "Fear"
cols[3] = "Happy"
cols[4] = "Sad"
cols[5] = "Surprise"
# cols [0] = "EF 1"
# cols [1] = "EF 2"
# cols [2] = "EF 3"
# cols [3] = "EF 4"
# cols [4] = "EF 5"


# class_list = ["Angry","Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
# class_list2 = ["Angry","Disgust", "Fear", "Happy", "Sad", "Surprise"]

# for i in range(0,6):
#     PATH = os.path.join('/media/Data/Datasets/RML/Eigenfaces20/test/', class_list[i])
#     print(PATH)
#     images = glob(os.path.join(PATH, "*.png"))
#     images.sort(key=lambda f: int(filter(str.isdigit, f)))
#     print(images)
#     for b in range(0,5):
#         img = imread(images[b])
#         cursp = fig.add_subplot(gs1[counter])
#         axs.append(fig.add_subplot(gs1[counter]))
#         # cursp.axis('off')
#         if b == 0:
#             cursp.annotate(class_list2[i], xy=(0, 0), xytext=(0, 0),
#                            xycoords=cursp.yaxis.label, textcoords='offset points',
#                            size=11, ha='right', va='center', rotation=90)
#         if counter < 5:
#             cursp.annotate(cols[b], xy=(0.5, 1), xytext=(0, 5),
#                            xycoords='axes fraction', textcoords='offset points',
#                            size=11, ha='center', va='baseline',  rotation=0)
#         cursp.set_xticks([])
#         cursp.set_yticks([])
#         # axs[-1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         axs[-1].imshow(img, cmap="bone")
#         counter = counter + 1
# # plt.tight_layout()
# plt.show()
# # plt.savefig(os.path.join('/media/Data/IET IP/Materials/Heatmaps/', "Heatmap_Class_Activation.eps"), format='eps', dpi=800, bbox_inches="tight")
# # plt.savefig("Heatmap_Class_Activation.png", format='png', dpi=800, bbox_inches="tight")



# class_list2 = ["eigenFaceNet","VGG-16", "ResNet-50"]
class_list2 = ["RML","eNTERFACE", "AFEW", "SAVEE"]
PATH = os.path.abspath('/media/Data/IET IP/Materials/Datasets/')
print(PATH)
images = glob(os.path.join(PATH, "*.png"))
images.sort(key=lambda f: int(filter(str.isdigit, f)))
for i in range(0,1):
    for b in range(0,24):
        img = imread(images[b])
        cursp = fig.add_subplot(gs1[counter])
        axs.append(fig.add_subplot(gs1[counter]))
        # cursp.axis('off')
        if b == 0:
            cursp.annotate(class_list2[0], xy=(0, 0), xytext=(0, 0),
                        xycoords=cursp.yaxis.label, textcoords='offset points',
                        size=11, ha='right', va='center', rotation=90)
        if b == 6:
            cursp.annotate(class_list2[1], xy=(0, 0), xytext=(0, 0),
                        xycoords=cursp.yaxis.label, textcoords='offset points',
                        size=11, ha='right', va='center', rotation=90)
        if b == 12:
            cursp.annotate(class_list2[2], xy=(0, 0), xytext=(0, 0),
                        xycoords=cursp.yaxis.label, textcoords='offset points',
                        size=11, ha='right', va='center', rotation=90)
        if b == 18:
            cursp.annotate(class_list2[3], xy=(0, 0), xytext=(0, 0),
                        xycoords=cursp.yaxis.label, textcoords='offset points',
                        size=11, ha='right', va='center', rotation=90)
        if i == 0 and b < 6:
            cursp.annotate(cols[b], xy=(0.5, 1), xytext=(0, 5),
                        xycoords='axes fraction', textcoords='offset points',
                        size=11, ha='center', va='baseline',  rotation=0)
        cursp.set_xticks([])
        cursp.set_yticks([])
        # axs[-1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[-1].imshow(img, cmap="gray")
        counter = counter + 1
# plt.tight_layout()

filename = "Frame_Samples.eps"
plt.savefig(os.path.join('/media/Data/Thesis/Figure/', filename), format='eps', dpi=250, bbox_inches="tight")
plt.show()
# plt.savefig("Heatmap_Class_Activation.png", format='png', dpi=800, bbox_inches="tight")


# cols[0] = "Avg. Face"
# cols[1] = "Ef 1"
# cols[2] = "EF 1 + Avg"
# cols[3] = "Ef 2"
# cols[4] = "EF 2 + Avg"
# cols[5] = "Ef 3"
# cols[6] = "EF 3 + Avg"
# cols[7] = "Ef 4"
# cols[8] = "EF 4 + Avg"
# cols[9] = "Ef 5"
# cols[10] = "EF 5 + Avg"
h = 0
# # m = [[] for d in range(106626)] #106626
# sum = 0
# for i in range(0,1):
#     imid = 0
#     for j in range(0,1):
#         h = 0
#         sbjstr = abbr[i] + '_' + str(j+1) + '_aligned'
#         # PATH = os.path.abspath(os.path.join(base_dir, class_list[i], str(j), sbjstr))
#         PATH = "/media/Data/Datasets/AFEW/Extracted_Frames/Train/Surprise/003707080"
#         # PATH = "/media/Data/Datasets/eNTERFACE/Extracted_Frames/Angry/35/A_36_aligned"
#         print(PATH)
#         images = glob(os.path.join(PATH, "*.jpg"))
#         images.sort(key=lambda f: int(filter(str.isdigit, f)))
#
#         label = class_list[i]
#         categorical_label = i
#         # sum = sum + len(images)
#         m = [[] for d in range(len(images))]  # 106626
#         for k in range(len(images)):
#             im = imread(images[k])
#             im = cv2.resize(im, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_CUBIC)
#             # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#             m[h] = list(chain.from_iterable(im))
#             h += 1
# # print(sum)
# #
#         m = np.matrix(m)
#         # np.save("RML_All_images.npy", m)
#         # m = np.load("RML_All_images.npy")
#         model = PCA()
#         pts = normalize(m)
#         model.fit(pts)
#         pts2 = model.transform(pts)
#
#         plt.figure()
#         variances = np.cumsum(model.explained_variance_ratio_)
#         meanpoint = np.mean(variances)
#         dists = np.abs(np.subtract(variances, 0.90))
#         # dists = np.abs([x - meanpoint for x in variances])
#         index, value = min(enumerate(dists), key=operator.itemgetter(1))
#         markers_on = [index]
#         plt.plot(np.cumsum(model.explained_variance_ratio_), '-o', color='black', markevery=markers_on, markerfacecolor='red', markersize=7, markeredgecolor="red")
#         plt.xlabel('Number of Principal Components', fontsize=20)
#         # plt.ylabel('Explained Variance (%)', fontsize=20)  # for each component
#         plt.xticks(fontsize=18)
#         plt.yticks(fontsize=18)
#         # plt.title(PATH)
#         plt.savefig("/media/Data/IET IP/Materials/" + Dataset + "_Data_Variation.eps", format='eps', dpi=200, bbox_inches="tight")
#         # plt.draw()
# plt.show()

# for l in range(NUM_PCA_COMP):
#
#     ord = model.components_[l]
#     gray = ord.reshape(IM_SIZE,IM_SIZE)
#     # gray = projected[l].reshape(150, 150)
#     gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
#
#     # cv2.imwrite("projected.png", gray)
#     img = np.zeros((IM_SIZE,IM_SIZE,3))
#     img[:, :, 0] = gray
#     img[:, :, 1] = gray
#     img[:, :, 2] = gray
#     img2 = img + avg_face
#     # scipy.misc.imsave('myimg2.png', img)
#
#     ef_num = str(l)
#     if l < 10:
#         ef_num = '0' + str(l)
#
#     sbj_num = str(j)
#     if j < 10:
#         sbj_num = '00' + str(j)
#     elif (j >= 10 and j < 100):
#         sbj_num = '0' + str(j)
#
#
#     # filename1 = abbr[i] + '_' + sbj_num + '_' + ef_num + '_0.png'
#     # filename2 = abbr[i] + '_' + sbj_num + '_' + ef_num + '.png'
#     # savingpath = ''
# if j in testindices:
#     # savingpath1 = os.path.join(save_dir, 'test', class_list[i], filename1)
#     savingpath2 = os.path.join(save_dir, 'test', class_list[i], filename2)
# elif j in valdindices:
#     # savingpath1 = os.path.join(save_dir, 'validation', class_list[i], filename1)
#     savingpath2 = os.path.join(save_dir, 'validation', class_list[i], filename2)
# else:
#     # savingpath1 = os.path.join(save_dir, 'train', class_list[i], filename1)
#     savingpath2 = os.path.join(save_dir, 'train', class_list[i], filename2)
# # scipy.misc.imsave(savingpath1, img)
# scipy.misc.imsave(savingpath2, img2)
