import os
import math
import time
import random
import string
import shutil
import itertools
import numpy as np
import pandas as pd
# import seaborn as sns
from scipy import interp
import matplotlib.pyplot as plt
from clr_callback import CyclicLR

from keras import models
from keras.callbacks import CSVLogger
from keras.models import Model
from keras import layers
from keras.optimizers import *
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
# from keras import backend as bk
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from keras.callbacks import LearningRateScheduler
from keras import initializers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# from sklearn.model_selection import StratifiedKFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# In Sake of Reproducible Results
seednum = 521
np.random.seed(seednum)
random_str = ''.join([random.choice(string.ascii_uppercase + string.digits) for n in xrange(16)])
print(random_str)

# Variable Definition
lrr = []
finetuneepoch = 50
trainepoch = 20
train_batchsize = 32
val_batchsize = 16
test_batchsize = 1
channel = 3
image_height = 197
image_width = 197
patience = 10
patience2 = 10
max_lr = 1e-03
fbase_lr = 1e-04
base_lr = 1e-04
min_lr = 1e-05
lr_factor = 0.5
dropout_rate = 0.5
classes_num = 6
smooth_factor = 0.8
start_point = 0
# lr_typ = "cyclic"
lr_typ = "reduce_lr"
# lr_typ2 = "reduce_lr2"
layer_name = "dense_1"
# layer_name = "fc4"

# Dataset = 'RML'
# Dataset = 'eNTERFACE'
Dataset = 'AFEW'
class_names = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
ModelType = "VGG16"
# ModelType = "ResNET50"
# ModelType = "eigenFaceNet"

# # this variable is only for VGG-16 and RESNET-50
layername = "block5_conv1"
# layername = "res5c_branch2a"
# layername = ""

NUM_COMP = 20

file_path = "/media/Data/IET IP/Code/"
# base_dir = "/media/Data/IET IP/Data/" + Dataset + "/ALL/"
# res_dir = "/media/Data/IET IP/Results/" + Dataset + "/Random/"
# res_dir = "/media/Data/IET IP/Results/" + Dataset + "/Combined_AU_Eigenfaces9/"
# res_dir = "/media/Data/IET IP/Results/" + Dataset + "/KMEANS/"
# res_dir = "/media/Data/IET IP/Results/" + Dataset + "/Extracted_Frames_AU/"
# res_dir = "/media/Data/IET IP/Results/" + Dataset + "/ALLFrames/"
# res_dir = "/media/Data/IET IP/Results/" + Dataset + "/EigenFaces/ALL/"
res_dir = "/media/Data/IET IP/Results/" + Dataset + "/EigenFaces/" + str(NUM_COMP) + "/"
# res_dir = "/media/Data/IET IP/Results/" + Dataset + "/EigenFaces/" + str(NUM_COMP) + "_AU/"


os.makedirs(os.path.join(res_dir, random_str))
cur_res_dir = os.path.join(res_dir, random_str)

os.makedirs(os.path.join(cur_res_dir, "csv"))
os.makedirs(os.path.join(cur_res_dir, "npy"))
os.makedirs(os.path.join(cur_res_dir, "eps"))
os.makedirs(os.path.join(cur_res_dir, "finetune"))
os.makedirs(os.path.join(cur_res_dir, "feature"))

csv_dir = os.path.join(cur_res_dir, "csv")
npy_dir = os.path.join(cur_res_dir, "npy")
eps_dir = os.path.join(cur_res_dir, "eps")
fit_dir = os.path.join(cur_res_dir, "finetune")
feature_dir = os.path.join(cur_res_dir, "feature")
checkpoint_dir = os.path.join(cur_res_dir,
                              'weights-improvement-{epoch:03d}-{acc:.2f}-{loss:.2f}-{val_acc:.2f}-{val_loss:.2f}.hdf5')
checkpoint_dir2 = os.path.join(fit_dir,
                               'weights-improvement-{epoch:03d}-{acc:.2f}-{loss:.2f}-{val_acc:.2f}-{val_loss:.2f}.hdf5')


# img_dir = '/media/Data/Datasets/' + Dataset + '/Extracted_Keyframes/Train_Test_Validation/'
# img_dir = '/media/Data/Datasets/' + Dataset + '/ALLFrames/'
# img_dir = '/media/Data/Datasets/' + Dataset + '/ALLEigenfaces/'
# img_dir = '/media/Data/Datasets/' + Dataset + "/Combined_AU_Eigenfaces9/"
# img_dir = '/media/Data/Datasets/' + Dataset + '/Extracted_Frames_AU/'
# img_dir = '/media/Data/Datasets/' + Dataset + '/Random/'
img_dir = '/media/Data/Datasets/' + Dataset + '/Eigenfaces' + str(NUM_COMP) + '/'
# img_dir = '/media/Data/Datasets/' + Dataset + '/Eigenfaces' + str(NUM_COMP) + '_AU/'
# img_dir = '/media/Data/Datasets/' + Dataset +   '/Eigenfaces' + str(NUM_COMP) + '_Average/'
# img_dir = '/media/Data/Datasets/' + Dataset +   '/Eigenframes' + str(NUM_COMP) + '/'


train_dir = os.path.join(img_dir, 'train')
vald_dir = os.path.join(img_dir, 'validation')
test_dir = os.path.join(img_dir, 'test')


# load data
# train_data = np.load(os.path.join(base_dir, "Face_train_data_ordered.npy"))
# vald_data = np.load(os.path.join(base_dir, "Face_validation_data_ordered.npy"))
# test_data = np.load(os.path.join(base_dir, "Face_test_data_ordered.npy"))
# print(train_data.shape)
# print(vald_data.shape)
# print(test_data.shape)
#
# train_labels = np.load(os.path.join(base_dir, "Face_train_label_ordered.npy"))
# vald_labels = np.load(os.path.join(base_dir, "Face_validation_label_ordered.npy"))
# test_labels = np.load(os.path.join(base_dir, "Face_test_label_ordered.npy"))
#
# # convert to one hot coded
# train_labels_one_hot_coded = to_categorical(y=train_labels, num_classes=classes_num)
# vald_labels_one_hot_coded = to_categorical(y=vald_labels, num_classes=classes_num)
# test_labels_one_hot_coded = to_categorical(y=test_labels, num_classes=classes_num)
# print(train_labels_one_hot_coded.shape)
# print(vald_labels_one_hot_coded.shape)
# print(test_labels_one_hot_coded.shape)
#
# # changing data type to avoid type errors
# train_data = train_data.astype('float32')
# vald_data = vald_data.astype('float32')
# test_data = test_data.astype('float32')

# rotation_range=40,
# width_shift_range=0.2,
# height_shift_range=0.2,
# shear_range=0.2,
# zoom_range=0.2,
# horizontal_flip=True,
# vertical_flip=True,
# rescale=1. / 255,
# fill_mode='nearest'

# Data Preprocess and Data Augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255)

vald_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# train_datagen.fit(train_data, augment=True)
train_generator = train_datagen.flow_from_directory(
                     directory=train_dir,
                     target_size=(image_width, image_height),
                     class_mode='categorical',
                     batch_size=train_batchsize)
#
vald_generator = vald_datagen.flow_from_directory(
                     directory=vald_dir,
                     target_size=(image_width, image_height),
                     class_mode='categorical',
                     batch_size=val_batchsize)
#
test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size=(image_width, image_height),
                    batch_size=test_batchsize,
                    shuffle=False,
                    class_mode='categorical')


# Saving Hyperparamters of the model

def save_settings():
    setting_info = "Dataset = " + str(Dataset)
    setting_info = setting_info + "\nSaving folder Path =" + random_str
    setting_info = setting_info + "\nSeed for Random Numbers = " + str(seednum)
    setting_info = setting_info + "\nNumber of Epochs In Finetune = " + str(finetuneepoch)
    setting_info = setting_info + "\nNumber of Epochs In Training = " + str(trainepoch)
    setting_info = setting_info + "\nTrain Batchsize = " + str(train_batchsize)
    setting_info = setting_info + "\nValidation Batchsize = " + str(val_batchsize)
    setting_info = setting_info + "\nTest Batchsize = " + str(test_batchsize)
    setting_info = setting_info + "\nMinimum Learning Rate = " + str(min_lr)
    setting_info = setting_info + "\nLearning Rate = " + str(base_lr)
    setting_info = setting_info + "\nMaximum Learning Rate = " + str(max_lr)
    setting_info = setting_info + "\nLearning Rate decay factor = " + str(lr_factor)
    setting_info = setting_info + "\nLearning Rate Patience = " + str(patience)
    setting_info = setting_info + "\nDropout Rate = " + str(dropout_rate)
    setting_info = setting_info + "\nSmoothing Factor = " + str(smooth_factor)
    setting_info = setting_info + "\nFeatures Saved For Layer = " + str(layer_name)
    setting_info = setting_info + "\nStarting Point = " + str(start_point)
    # setting_info = setting_info + "\nData Path = " + str(base_dir)
    setting_info = setting_info + "\nCallbacks = " + callbacks_list_str
    setting_info = setting_info + "\nTrain accuracy = " + str(train_acc)
    setting_info = setting_info + "\nTrain loss = " + str(train_loss)
    setting_info = setting_info + "\nFinetuning Train accuracy = " + str(ftrain_acc)
    setting_info = setting_info + "\nFinetuning Train loss = " + str(ftrain_loss)
    setting_info = setting_info + "\nValidation accuracy = " + str(np.mean(val_acc))
    setting_info = setting_info + "\nValidation loss = " + str(np.mean(val_loss))
    setting_info = setting_info + "\nTest accuracy = " + str(test_acc)
    setting_info = setting_info + "\nTest loss = " + str(test_loss)
    setting_info = setting_info + "\nProcess Time in seconds = " + str(process_time)
    return setting_info


# Function Definition

def normalizing_data(traindata, testdata):
    mean = traindata.mean(axis=0)
    train_data2 = traindata - mean
    std = train_data2.std(axis=0)

    if std.any != 0:
        traindata -= mean
        traindata /= std
        testdata -= mean
        testdata /= std

    return traindata, testdata


# saving labels or predicted probablities as a npy file

def save_npy(data, flag="Train", name="Probabilities", path=npy_dir, epoch=trainepoch):
    np.save(os.path.join(path, flag + "_Face_" + name + "_" + str(epoch) + "_Epochs.npy"), data)


# saving labels or predicted probablities as a csv file
def save_csv(data, flag="Train", name="Probabilities", path=csv_dir, epoch=trainepoch):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path, flag + "_Face_" + name + "_" + str(epoch) + "_Epochs.csv"), header=False, index=False)


def save_layer_output(x, path=feature_dir, name="Train"):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    # intermediate_output = intermediate_layer_model.predict(x)
    intermediate_output = intermediate_layer_model.predict_generator(x)
    filename = name + "_" + layer_name + "_layer_features"
    np.save(os.path.join(path, filename), intermediate_output)


# Smoothing the plots
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# learning rate schedule
def step_decay(epoch):
    initial_lrate = base_lr
    drop = 0.5
    epochs_drop = 50
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    lrr.append(lrate)
    return lrate


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        # lrr.append(float(K.get_value(optimizer.lr)))
        return optimizer.lr
    return lr


def set_lr(typ="reduce_lr"):
    # learning schedule callback
    lrate = ReduceLROnPlateau(monitor='val_acc', factor=lr_factor,
                              patience=patience, min_lr=min_lr, mode='max', verbose=1)

    if typ == "reduce_lr2":
        lrate = ReduceLROnPlateau(monitor='val_acc', factor=lr_factor,
                                  patience=patience2, min_lr=min_lr, mode='max', verbose=1)
    elif typ == "step_decay":
        lrate = LearningRateScheduler(step_decay)
    elif typ == "lambda":
        lrate = LearningRateScheduler(lambda epoch: base_lr * 0.99 ** epoch)
    elif typ == "cyclic":
        lrate = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=500., mode='exp_range', gamma=0.99994)

    return lrate


def set_optimizer(typ="Adam"):
    # choosing between different optimizers
    optimizer = Adam(lr=base_lr, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=0.0)

    if typ == "Adam2":
        optimizer = Adam(lr=fbase_lr, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=0.75)
    elif typ == "SGD":
        optimizer = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    elif typ == "Adagrad":
        optimizer = Adagrad(lr=base_lr, epsilon=1e-08, decay=0.0)
    elif typ == "Adadelta":
        optimizer = Adadelta(lr=base_lr, rho=0.95, epsilon=1e-08, decay=0.0)
    elif typ == "Adamax":
        optimizer = Adamax(lr=base_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    elif typ == "Nadam":
        optimizer = Nadam(lr=base_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    elif typ == "RMSprop":
        optimizer = RMSprop(lr=base_lr, rho=0.9, epsilon=1e-08, decay=0.0)
    return optimizer


def set_checkpoint():
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    return checkpoint


def set_checkpoint2():
    checkpoint = ModelCheckpoint(checkpoint_dir2, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    return checkpoint


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues, epoch=trainepoch):  # title='Confusion matrix',
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if normalize:
        cf_nonnormalized_filename = os.path.join(eps_dir, Dataset + "_Face_Normalized_" + str(epoch) + "_Epoch.eps")
        plt.savefig(cf_nonnormalized_filename, format='eps', dpi=800, bbox_inches="tight")
    else:
        cf_normalized_filename = os.path.join(eps_dir, Dataset + "_Face_nonNormalized_" + str(epoch) + "_Epoch.eps")
        plt.savefig(cf_normalized_filename, format='eps', dpi=800, bbox_inches="tight")


def plot_train_vs_vald(train_points, vald_points, isloss=False, epoch=trainepoch, isfinetune=False):
    plot_label = 'Accuracy'
    plot_name = ''
    if isloss:
        plot_label = 'Loss'

    if isfinetune:
        plot_name = 'Finetuned_'

    epochs = range(1, len(train_points) + 1)

    lines1 = plt.plot(epochs, smooth_curve(train_points), label='Training ' + plot_label)
    plt.setp(lines1, color='red', linewidth=1.0)
    lines2 = plt.plot(epochs, smooth_curve(vald_points), 'b-', label='Validation ' + plot_label)
    plt.setp(lines2, color='black', linewidth=1.0)
    # plt.title('Training and Validation ' + plot_label)
    plt.xlabel('Epochs')
    plt.ylabel(plot_label)
    plt.legend()
    filename = os.path.join(eps_dir, plot_name + Dataset + "_Face_Training_Validation_" + plot_label + "_" + str(
        epoch) + "_Epoch.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


# Compute and plot ROC curve and ROC area for each class
def plot_roc(y_test, y_score, epoch=trainepoch):
    # plot line width
    # lw =1

    # variable definition
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculating auc, false positive rate and true positive rate for each class
    for i in range(classes_num):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes_num)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(classes_num):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= classes_num

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # smoothing the micro roc curve
    # micro_poly = np.polyfit(fpr["micro"], tpr["micro"], 5)
    # micro_poly_y = np.poly1d(micro_poly)(fpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='magenta', linestyle=':', linewidth=1)

    # smoothing the macro roc curve
    # macro_poly = np.polyfit(fpr["macro"], tpr["macro"], 5)
    # macro_poly_y = np.poly1d(macro_poly)(fpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='black', linestyle='-.', linewidth=1)
    # colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # colors = itertools.cycle(['pink', 'purpule', 'aqua', 'red', 'yellow', 'green'])
    colors = itertools.cycle(['blue', 'magenta', 'cyan', 'red', 'yellow', 'green'])
    for i, color in zip(range(classes_num), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label=' {0} (area = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='silver', linestyle='--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic for each class')
    plt.legend(loc="lower right")
    filename = os.path.join(eps_dir, Dataset + "_Face_Micro_Macro_Avg_ROC_Curve_" + str(epoch) + "_Epoch.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


# plotting learning rate
def plot_lr(lr_points, epoch=finetuneepoch, isfinetune=False):

    plot_name = ''
    if isfinetune:
        plot_name = 'Finetuned_'
    epochs = range(1, len(lr_points) + 1)

    lines1 = plt.plot(epochs, lr_points, label='learning rate')
    plt.setp(lines1, color='black', linewidth=1.0)

    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    filename = os.path.join(eps_dir, plot_name + Dataset + "_Face_Learning_Rate_" + str(epoch) + "_Epoch.eps")
    plt.savefig(filename, format='eps', dpi=800, bbox_inches="tight")


def custom_model():
    eigenFaceNet = models.Sequential()

    eigenFaceNet.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1),
                                   input_shape=(image_width, image_height, 3),
                                   kernel_initializer='random_uniform', activation="relu", name="conv_1",
                                   kernel_regularizer=l2(0.0), bias_initializer=initializers.Zeros(), padding='SAME'))
    eigenFaceNet.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(3, 3), padding='VALID', name="maxpool_1"))

    eigenFaceNet.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1),
                                   kernel_initializer='random_uniform', activation="relu", name="conv_21",
                                   kernel_regularizer=l2(0.0), bias_initializer=initializers.Zeros(), padding='SAME'))
    # eigenFaceNet.add(layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1),
    #                                kernel_initializer='random_uniform', activation="relu", name="conv_22",
    #                                kernel_regularizer=l2(0.0), bias_initializer=initializers.Zeros(), padding='SAME'))
    eigenFaceNet.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(3, 3), padding='VALID', name="maxpool_2"))

    eigenFaceNet.add(layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1),
                                   kernel_initializer='random_uniform', activation="relu", name="conv_31",
                                   kernel_regularizer=l2(0.0), bias_initializer=initializers.Zeros(), padding='SAME'))
    # eigenFaceNet.add(layers.Conv2D(filters=128, kernel_size=(7, 7), strides=(1, 1),
    #                                kernel_initializer='random_uniform', activation="relu", name="conv_32",
    #                                kernel_regularizer=l2(0.0), bias_initializer=initializers.Zeros(), padding='SAME'))
    eigenFaceNet.add(layers.MaxPool2D(pool_size=(2, 2), strides=(3, 3), padding='VALID', name="maxpool_3"))
    #
    # eigenFaceNet.add(layers.Conv2D(filters=256, kernel_size=(7, 7), strides=(1, 1),
    #                                kernel_initializer='random_uniform', activation="relu", name="conv_4",
    #                                kernel_regularizer=l2(0.0), bias_initializer=initializers.Zeros(), padding='SAME'))
    # eigenFaceNet.add(layers.MaxPool2D(pool_size=(2, 2), strides=(3, 3), padding='VALID', name="maxpool_4"))
    #
    # eigenFaceNet.add(layers.Conv2D(filters=512, kernel_size=(7, 7), strides=(1, 1),
    #                                kernel_initializer='random_uniform', activation="relu", name="conv_5",
    #                                kernel_regularizer=l2(0.0), bias_initializer=initializers.Zeros(), padding='SAME'))
    # eigenFaceNet.add(layers.MaxPool2D(pool_size=(2, 2), strides=(3, 3), padding='VALID', name="maxpool_5"))

    eigenFaceNet.add(layers.Dropout(dropout_rate, name="dropout"))
    eigenFaceNet.add(layers.Flatten(name="flatten"))
    eigenFaceNet.add(layers.Dense(64, activation='relu', name="fc4"))
    eigenFaceNet.add(layers.Dense(classes_num, activation='softmax'))
    print(eigenFaceNet.summary())
    # eigenFaceNet.load_weights("/media/Data/IET IP/Results/RML/KMEANS/XLKGYQEFMDK5RJIJ/weights-improvement-285-0.81-0.51-0.85-0.48.hdf5")
    # checkpoint results
    checkpoint = set_checkpoint()

    # set learning schedule
    lrate = set_lr(typ="reduce_lr")
    # set the optimizer
    optimizer = set_optimizer(typ="Adam")

    # set callbacks
    callbacks_list = [checkpoint, lrate]
    callbacks_list_str = "[checkpoint, reduce_lr]"
    lr_metric = get_lr_metric(optimizer)
    eigenFaceNet.compile(loss='categorical_crossentropy',  # binary_crossentropy
                         optimizer=optimizer,
                         metrics=['accuracy', lr_metric])

    # fit the model to the generated data
    hhistory = eigenFaceNet.fit_generator(train_generator,
                                          steps_per_epoch=train_generator.samples // train_generator.batch_size,
                                          epochs=trainepoch, verbose=1,
                                          validation_data=vald_generator,
                                          validation_steps=vald_generator.samples // vald_generator.batch_size,
                                          callbacks=callbacks_list)
    tr_acc = hhistory.history['acc']
    tr_loss = hhistory.history['loss']
    vl_acc = hhistory.history['val_acc']
    vl_loss = hhistory.history['val_loss']
    tr_lrr = hhistory.history['lr']

    # saving the average results from folds
    # save_csv(train_labels, flag="Train", name="True_Labels")
    save_csv(tr_acc, flag="Train", name="Averaged_Accuracy")
    save_csv(vl_acc, flag="Validation", name="Averaged_Accuracy")
    save_csv(tr_loss, flag="Train", name="Averaged_Loss")
    save_csv(vl_loss, flag="Validation", name="Averaged_Loss")
    save_csv(tr_lrr, flag="Train", name="Averaged_LearningRate")

    plt.figure(num=1, figsize=(8, 6), dpi=80)
    plot_train_vs_vald(tr_loss[start_point:], vl_loss[start_point:], isloss=True)

    plt.figure(num=2, figsize=(8, 6), dpi=80)
    plot_train_vs_vald(tr_acc[start_point:], vl_acc[start_point:])

    plt.figure(num=3, figsize=(8, 6), dpi=80)
    plot_lr(tr_lrr[start_point:])

    return callbacks_list_str, eigenFaceNet, hhistory, tr_acc, tr_loss, vl_acc, vl_loss


def set_model(typ):
    if typ == "VGG16":
        conv_base = VGG16(include_top=False,
                          weights='imagenet',
                          input_shape=(image_width, image_height, channel))
    if typ == "VGG19":
        conv_base = VGG19(include_top=False,
                          weights='imagenet',
                          input_shape=(image_width, image_height, channel))
    elif typ == "ResNET50":
        conv_base = ResNet50(include_top=False,
                             weights='imagenet',
                             input_shape=(image_width, image_height, channel))

    return conv_base


# 3D model
def create_finetune_train_model():
    """ Return the Keras model of the network
    """
    conv_base = set_model(ModelType)
    print(conv_base.summary())
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))
    print(model.summary())

    # Freez Conv Base part of the model and train the Classifier part
    conv_base.trainable = False


    fcheckpoint = set_checkpoint2()
    flrate = set_lr(typ=lr_typ)
    foptimizer = set_optimizer(typ="Adam")
    fcallbacks_list = [fcheckpoint, flrate]
    lr_metric = get_lr_metric(foptimizer)

    model.compile(loss='categorical_crossentropy',
                  optimizer=foptimizer,
                  metrics=['accuracy', lr_metric])

    fhhistory = model.fit_generator(train_generator,
                                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                                    epochs=finetuneepoch, verbose=1,
                                    validation_data=vald_generator,
                                    validation_steps=vald_generator.samples // vald_generator.batch_size,
                                    callbacks=fcallbacks_list)
    ftr_acc = fhhistory.history['acc']
    ftr_loss = fhhistory.history['loss']
    fvl_acc = fhhistory.history['val_acc']
    fvl_loss = fhhistory.history['val_loss']
    ftr_lrr = fhhistory.history['lr']

    # saving the average results from folds
    # save_csv(train_labels, flag="Train", name="True_Labels")
    save_csv(ftr_acc, flag="Train", name="Finetuned_Averaged_Accuracy")
    save_csv(fvl_acc, flag="Validation", name="Finetuned_Averaged_Accuracy")
    save_csv(ftr_loss, flag="Train", name="Finetuned_Averaged_Loss")
    save_csv(fvl_loss, flag="Validation", name="Finetuned_Averaged_Loss")
    save_csv(ftr_lrr, flag="Train", name="Finetuned_Averaged_LearningRate")

    plt.figure(num=1, figsize=(8, 6), dpi=80)
    plot_train_vs_vald(ftr_loss[start_point:], fvl_loss[start_point:], isloss=True, isfinetune=True)

    plt.figure(num=2, figsize=(8, 6), dpi=80)
    plot_train_vs_vald(ftr_acc[start_point:], fvl_acc[start_point:], isfinetune=True)

    plt.figure(num=3, figsize=(8, 6), dpi=80)
    plot_lr(ftr_lrr[start_point:], isfinetune=True)

    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == layername:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False


    tcheckpoint = set_checkpoint()
    tlrate = set_lr(typ=lr_typ)
    toptimizer = set_optimizer(typ="Adam")
    tcallbacks_list = [tcheckpoint, tlrate]
    callbacks_list_str = "[checkpoint, " + lr_typ + "]"

    model.compile(loss='categorical_crossentropy',  # binary_crossentropy
                  optimizer=toptimizer,
                  metrics=['accuracy', lr_metric])

    # fit the model to the generated data
    thistory = model.fit_generator(train_generator,
                                   steps_per_epoch=train_generator.samples//train_generator.batch_size,
                                   epochs=trainepoch, verbose=1,
                                   validation_data=vald_generator,
                                   validation_steps=vald_generator.samples // vald_generator.batch_size,
                                   callbacks=tcallbacks_list)
    tr_acc = thistory.history['acc']
    tr_loss = thistory.history['loss']
    vl_acc = thistory.history['val_acc']
    vl_loss = thistory.history['val_loss']
    tr_lrr = thistory.history['lr']

    # saving the average results from folds
    # save_csv(train_labels, flag="Train", name="True_Labels")
    save_csv(tr_acc, flag="Train", name="Averaged_Accuracy")
    save_csv(vl_acc, flag="Validation", name="Averaged_Accuracy")
    save_csv(tr_loss, flag="Train", name="Averaged_Loss")
    save_csv(vl_loss, flag="Validation", name="Averaged_Loss")
    save_csv(tr_lrr, flag="Train", name="Averaged_LearningRate")

    plt.figure(num=4, figsize=(8, 6), dpi=80)
    plot_train_vs_vald(tr_loss[start_point:], vl_loss[start_point:], isloss=True)

    plt.figure(num=5, figsize=(8, 6), dpi=80)
    plot_train_vs_vald(tr_acc[start_point:], vl_acc[start_point:])

    plt.figure(num=6, figsize=(8, 6), dpi=80)
    plot_lr(tr_lrr[start_point:], epoch=trainepoch)

    return callbacks_list_str, model, thistory, fhhistory, ftr_acc, ftr_loss, fvl_acc, fvl_loss, tr_acc, tr_loss, vl_acc, vl_loss


test_true_labels = test_generator.classes
save_csv(test_true_labels, flag="Test", name="TrueClassLabels")
start = time.clock()
if ModelType == "eigenFaceNet":
    callbacks_list_str, model, history, train_acc, train_loss, val_acc, val_loss = custom_model()
else:
    callbacks_list_str, model, history, fhistory, ftrain_acc, ftrain_loss, fval_acc, fval_loss, train_acc, train_loss, val_acc, val_loss = create_finetune_train_model()


train_acc = np.mean(train_acc)
train_loss = np.mean(train_loss)
# ftrain_acc = ""
# ftrain_loss = ""
if ModelType != "eigenFaceNet":
    ftrain_acc = np.mean(ftrain_acc)
    ftrain_loss = np.mean(ftrain_loss)
end = time.clock()
process_time = (end - start)

# serialize model to JSON
model_json = model.to_json()
with open(os.path.join(cur_res_dir, Dataset + "_Face_model.json"), "w") as json_file:
    json_file.write(model_json)

# evaluation on train
# train_loss, train_acc = model.evaluate_generator(train_generator,
# steps=train_generator.samples//test_generator.batch_size)
# print('train_acc: ',train_acc,'train_loss: ',train_loss)
#
# train_predicted_probs = model.predict_generator(train_generator)
# train_predicted_labels = train_predicted_probs.argmax(axis=1)
#
# # Saving Train results
# save_npy(train_predicted_probs, flag="Train", name="Probabilities")
# save_npy(train_predicted_labels, flag="Train", name="ClassLabels")
# save_csv(train_predicted_probs, flag="Train", name="Probabilities")
# save_csv(train_predicted_labels, flag="Train", name="ClassLabels")

test_loss, test_acc, learnrate = model.evaluate_generator(test_generator,
                                                          steps=test_generator.samples//test_generator.batch_size)
print('test_acc: ', test_acc, 'test_loss: ', test_loss)

# Confution Matrix and Classification Report
filenames = test_generator.filenames
nb_samples = len(filenames)

test_predicted_probs = model.predict_generator(test_generator, steps=nb_samples)
test_predicted_labels = test_predicted_probs.argmax(axis=-1)

# saving features
save_layer_output(train_generator, path=feature_dir, name="Train")
save_layer_output(test_generator, path=feature_dir, name="Test")

# Plot all ROC curves

test_true_labels = test_generator.classes
test_labels_one_hot_coded = to_categorical(y=test_true_labels, num_classes=classes_num)

# fnames = test_generator.filenames
# label_map = (test_generator.class_indices)
# label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
# predictions = [label_map[k] for k in test_predicted_labels]

plt.figure(num=7, figsize=(8, 6), dpi=80)
plot_roc(test_labels_one_hot_coded, test_predicted_probs)

# Saving Test Results
save_npy(test_predicted_probs, flag="Test", name="Probabilities")
save_npy(test_predicted_labels, flag="Test", name="ClassLabels")
save_csv(test_predicted_probs, flag="Test", name="Probabilities")
save_csv(test_predicted_labels, flag="Test", name="ClassLabels")


# Compute confusion matrix
cnf_matrix = confusion_matrix(test_generator.classes, test_predicted_labels)
np.set_printoptions(precision=2)


# Plot and save non-normalized confusion matrix
plt.figure(num=8, figsize=(5, 5), dpi=80)
plot_confusion_matrix(cnf_matrix, classes=class_names)
# CF_NonNormalized_filename = os.path.join(eps_dir, Dataset + "_Face_NonNormalized_" + str(trainepoch) + "_Epoch.eps")
# plt.savefig(CF_NonNormalized_filename, format='eps', dpi=1000, bbox_inches="tight")

# Plot normalized confusion matrix
plt.figure(num=9, figsize=(5, 5), dpi=80)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)
# CF_Normalized_filename = os.path.join(eps_dir, Dataset + "_Face_Normalized_" + str(trainepoch) + "_Epoch.eps")
# plt.savefig(CF_Normalized_filename, format='eps', dpi=1000, bbox_inches="tight")

cnf_matrix2 = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
print(np.average(cnf_matrix2.diagonal()))

HyperParameter_Setting = save_settings()
with open(os.path.join(cur_res_dir, Dataset + "_Face_HyperParameters.txt"), "w") as text_file:
    text_file.write(HyperParameter_Setting)

print(HyperParameter_Setting)
shutil.copyfile(os.path.join(file_path, "pretrained_convnet_vgg16_with_fine_tunning.py"),
                os.path.join(cur_res_dir, "executed_script.txt"))
plt.show(block=True)
