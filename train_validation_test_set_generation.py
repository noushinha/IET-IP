import os,shutil
import numpy as np
from PIL import Image
import cv2

original_dataset_dir = '/media/Data/Datasets/CK+/Surprise/'
base_dir = '/media/Data/Datasets/CKPlus/'

if not os.path.exists(base_dir): os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir): os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir): os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir): os.mkdir(test_dir)


######################train################################
train_angry_dir = os.path.join(train_dir, 'Angry')
if not os.path.exists(train_angry_dir): os.mkdir(train_angry_dir)
validation_angry_dir = os.path.join(validation_dir, 'Angry')
if not os.path.exists(validation_angry_dir): os.mkdir(validation_angry_dir)
test_angry_dir = os.path.join(test_dir, 'Angry')
if not os.path.exists(test_angry_dir): os.mkdir(test_angry_dir)

train_contempt_dir = os.path.join(train_dir, 'Contempt')
if not os.path.exists(train_contempt_dir): os.mkdir(train_contempt_dir)
validation_contempt_dir = os.path.join(validation_dir, 'Contempt')
if not os.path.exists(validation_contempt_dir): os.mkdir(validation_contempt_dir)
test_contempt_dir = os.path.join(test_dir, 'Contempt')
if not os.path.exists(test_contempt_dir): os.mkdir(test_contempt_dir)

train_disgust_dir = os.path.join(train_dir, 'Disgust')
if not os.path.exists(train_disgust_dir): os.mkdir(train_disgust_dir)
validation_disgust_dir = os.path.join(validation_dir, 'Disgust')
if not os.path.exists(validation_disgust_dir): os.mkdir(validation_disgust_dir)
test_disgust_dir = os.path.join(test_dir, 'Disgust')
if not os.path.exists(test_disgust_dir): os.mkdir(test_disgust_dir)

train_fear_dir = os.path.join(train_dir, 'Fear')
if not os.path.exists(train_fear_dir): os.mkdir(train_fear_dir)
validation_fear_dir = os.path.join(validation_dir, 'Fear')
if not os.path.exists(validation_fear_dir): os.mkdir(validation_fear_dir)
test_fear_dir = os.path.join(test_dir, 'Fear')
if not os.path.exists(test_fear_dir): os.mkdir(test_fear_dir)

train_happiness_dir = os.path.join(train_dir, 'Happiness')
if not os.path.exists(train_happiness_dir): os.mkdir(train_happiness_dir)
validation_happiness_dir = os.path.join(validation_dir, 'Happiness')
if not os.path.exists(validation_happiness_dir): os.mkdir(validation_happiness_dir)
test_happiness_dir = os.path.join(test_dir, 'Happiness')
if not os.path.exists(test_happiness_dir): os.mkdir(test_happiness_dir)


train_sadness_dir = os.path.join(train_dir, 'Sadness')
if not os.path.exists(train_sadness_dir): os.mkdir(train_sadness_dir)
validation_sadness_dir = os.path.join(validation_dir, 'Sadness')
if not os.path.exists(validation_sadness_dir): os.mkdir(validation_sadness_dir)
test_sadness_dir = os.path.join(test_dir, 'Sadness')
if not os.path.exists(test_sadness_dir): os.mkdir(test_sadness_dir)

train_surprise_dir = os.path.join(train_dir, 'Surprise')
if not os.path.exists(train_surprise_dir): os.mkdir(train_surprise_dir)
validation_surprise_dir = os.path.join(validation_dir, 'Surprise')
if not os.path.exists(validation_surprise_dir): os.mkdir(validation_surprise_dir)
test_surprise_dir = os.path.join(test_dir, 'Surprise')
if not os.path.exists(test_surprise_dir): os.mkdir(test_surprise_dir)

Catabbr = 'SU_'
test__dst = test_surprise_dir
valid_dst = validation_surprise_dir
train_dst = train_surprise_dir


sbjlist = os.listdir(original_dataset_dir)
num_test_samples = int(np.ceil(.1 * len(sbjlist)))
num_validation_samples = int(np.ceil(.1 * len(sbjlist)))
num_train_samples = len(sbjlist) - (num_test_samples+num_validation_samples)

print(sbjlist)

############# TEST ###################3
for sbj in range(0,num_test_samples):
    sbjIndex = int(sbjlist[sbj])
    sbjstr = str(sbjIndex)
    if sbjIndex < 10:
        sbjstr = '00' + str(sbjIndex)
    elif sbjIndex < 100:
        sbjstr = '0' + str(sbjIndex)

    folderstr = Catabbr + sbjstr + '_aligned'
    src = os.path.join(original_dataset_dir, str(sbjIndex), folderstr)
    print(src)

    bmpfiles = os.listdir(src)
    for fname in bmpfiles:
        srcfile = os.path.join(src, fname)
        oldname = fname.split('_')
        framnum = oldname[3].split('.')
        newname = Catabbr + sbjstr + '_' + framnum[0] + '.png'
        dstfile = os.path.join(test__dst, newname)
        img = cv2.imread(srcfile)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(img)
        img2[:, :, 0] = gray_image
        img2[:, :, 1] = gray_image
        img2[:, :, 2] = gray_image
        if img2.shape[2] != 3:
            raise Exception('x should should have 3 channels. x has: {} channels'.format(img2.shape))

        cv2.imwrite(dstfile, img2)
        # Image.open(gray_image).save(dstfile)



############# VALIDATION ###################3
for sbj in range(num_test_samples,num_test_samples+num_validation_samples):
    sbjIndex = int(sbjlist[sbj])
    sbjstr = str(sbjIndex)
    if sbjIndex < 10:
        sbjstr = '00' + str(sbjIndex)
    elif sbjIndex < 100:
        sbjstr = '0' + str(sbjIndex)

    folderstr = Catabbr + sbjstr + '_aligned'
    src = os.path.join(original_dataset_dir, str(sbjIndex), folderstr)
    print(src)

    bmpfiles = os.listdir(src)
    for fname in bmpfiles:
        srcfile = os.path.join(src, fname)
        oldname = fname.split('_')
        framnum = oldname[3].split('.')
        newname = Catabbr + sbjstr + '_' + framnum[0] + '.png'
        dstfile = os.path.join(valid_dst, newname)
        img = cv2.imread(srcfile)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(img)
        img2[:, :, 0] = gray_image
        img2[:, :, 1] = gray_image
        img2[:, :, 2] = gray_image
        if img2.shape[2] != 3:
            raise Exception('x should should have 3 channels. x has: {} channels'.format(img2.shape))

        cv2.imwrite(dstfile, img2)



############ TRAIN ###################
for sbj in range(num_test_samples+num_validation_samples, len(sbjlist)):
    sbjIndex = int(sbjlist[sbj])
    sbjstr = str(sbjIndex)
    if sbjIndex < 10:
        sbjstr = '00' + str(sbjIndex)
    elif sbjIndex < 100:
        sbjstr = '0' + str(sbjIndex)

    folderstr = Catabbr + sbjstr + '_aligned'
    src = os.path.join(original_dataset_dir, str(sbjIndex), folderstr)
    print(src)

    bmpfiles = os.listdir(src)
    for fname in bmpfiles:
        srcfile = os.path.join(src, fname)
        oldname = fname.split('_')
        framnum = oldname[3].split('.')
        newname = Catabbr + sbjstr + '_' + framnum[0] + '.png'
        dstfile = os.path.join(train_dst, newname)
        img = cv2.imread(srcfile)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(img)
        img2[:, :, 0] = gray_image
        img2[:, :, 1] = gray_image
        img2[:, :, 2] = gray_image
        if img2.shape[2] != 3:
            raise Exception('x should should have 3 channels. x has: {} channels'.format(img2.shape))

        cv2.imwrite(dstfile, img2)



# print('total training angry images', len(os.listdir(train_angry_dir)))
# print('total training contempt images', len(os.listdir(train_contempt_dir)))
# print('total training disgust images', len(os.listdir(train_disgust_dir)))
# print('total training fear images', len(os.listdir(train_fear_dir)))
# print('total training happiness images', len(os.listdir(train_happiness_dir)))
# print('total training sadness images', len(os.listdir(train_sadness_dir)))
# print('total training surprise images', len(os.listdir(train_surprise_dir)))


# print('total validation angry images', len(os.listdir(train_angry_dir)))
# print('total validation contempt images', len(os.listdir(train_contempt_dir)))
# print('total validation disgust images', len(os.listdir(train_disgust_dir)))
# print('total validation fear images', len(os.listdir(train_fear_dir)))
# print('total validation happiness images', len(os.listdir(train_happiness_dir)))
# print('total validation sadness images', len(os.listdir(train_sadness_dir)))
# print('total validation surprise images', len(os.listdir(train_surprise_dir)))


# print('total testing angry images', len(os.listdir(train_angry_dir)))
# print('total testing contempt images', len(os.listdir(train_contempt_dir)))
# print('total testing disgust images', len(os.listdir(train_disgust_dir)))
# print('total testing fear images', len(os.listdir(train_fear_dir)))
# print('total testing happiness images', len(os.listdir(train_happiness_dir)))
# print('total testing sadness images', len(os.listdir(train_sadness_dir)))
# print('total testing surprise images', len(os.listdir(train_surprise_dir)))
