import os
import math
import time
import random
import string
import shutil
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interp
import matplotlib.pyplot as plt
from clr_callback import CyclicLR

from keras import models
from keras import layers
from keras.models import Model
from keras.optimizers import *
from keras import backend as bk
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# In Sake of Reproducible Results
seednum = 44
np.random.seed(seednum)
random_str = ''.join([random.choice(string.ascii_uppercase + string.digits) for n in xrange(16)])
print(random_str)

# Variable Definition
k = 10
epoch = 110
fitepoch = 75
patience = 20
batchsize = 32
max_lr = 6e-04
base_lr = 1e-04
min_lr = 1e-06
lr_factor = 0.15
dropout_rate = 0.25
classes_num = 6
image_width = 64
image_height = 64
smooth_factor = 0.8
start_point = 10
layer_name = "fc6"

data_type = 'ordered'
shuffle = True
Dataset = 'RML'
class_names = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seednum)
lrr = []
model_history = []
history_train_acc = []  # np.zeros(shape= [k, epoch])
history_vald_acc = []  # np.zeros(shape= [k, epoch])
history_train_loss = []  # np.zeros(shape= [k, epoch])
history_vald_loss = []  # np.zeros(shape= [k, epoch])
history_lrr = []  # np.zeros(shape= [k, epoch])

cnf_matrix = np.zeros(shape=[classes_num, classes_num])

# data lodaing and preparation
file_path = "/media/Data/IET IP/Code/"
base_dir = "/media/Data/IET IP/Data/" + Dataset + "/Eigenfaces/10/"
res_dir = "/media/Data/IET IP/Results/" + Dataset + "/EigenFaces/10/"

os.makedirs(os.path.join(res_dir, random_str))
cur_res_dir = os.path.join(res_dir, random_str)

os.makedirs(os.path.join(cur_res_dir, "csv"))
os.makedirs(os.path.join(cur_res_dir, "npy"))
os.makedirs(os.path.join(cur_res_dir, "eps"))
os.makedirs(os.path.join(cur_res_dir, "idx"))
os.makedirs(os.path.join(cur_res_dir, "feature"))

csv_dir = os.path.join(cur_res_dir, "csv")
npy_dir = os.path.join(cur_res_dir, "npy")
eps_dir = os.path.join(cur_res_dir, "eps")
idx_dir = os.path.join(cur_res_dir, "idx")
feature_dir = os.path.join(cur_res_dir, "feature")

# load data
train_data = np.load(os.path.join(base_dir, "Face_train_data_" + data_type + ".npy"))
test_data = np.load(os.path.join(base_dir, "Face_test_data_" + data_type + ".npy"))

print(train_data.shape)

train_labels = np.load(os.path.join(base_dir, "Face_train_label_" + data_type + ".npy"))
test_labels = np.load(os.path.join(base_dir, "Face_test_label_" + data_type + ".npy"))

# convert to one hot coded
all_train_labels_one_hot_coded = to_categorical(y=train_labels, num_classes=classes_num)
test_labels_one_hot_coded = to_categorical(y=test_labels, num_classes=classes_num)
print(all_train_labels_one_hot_coded.shape)

# changing data type to avoid type errors
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# Standardization
train_data = train_data/255
test_data = test_data/255


def normalizing_data(tr_data, ts_data):
    mean = tr_data.mean(axis=0)
    train_data2 = tr_data - mean
    std = train_data2.std(axis=0)

    if std.any != 0:
        tr_data -= mean
        tr_data /= std
        ts_data -= mean
        ts_data /= std

    return tr_data, ts_data


# saving labels or predicted probablities as a npy file
def save_npy(data, flag="Train", name="Probabilities", path=npy_dir):
    np.save(os.path.join(path, flag + "_Face_" + name + "_" + str(epoch) + "_Epochs.npy"), data)


# saving labels or predicted probablities as a csv file
def save_csv(data, flag="Train", name="Probabilities", path=csv_dir):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path, flag + "_Face_" + name + "_" + str(epoch) + "_Epochs.csv"))


def save_layer_output(xx, path=feature_dir, name="Train"):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(xx)
    filename = name + "_Face_fc6_Layer_Features"
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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
    plt.title(title)
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
        CF_NonNormalized_filename = os.path.join(eps_dir, Dataset + "_Video_NonNormalized_" + str(epoch) + "_Epoch_" + data_type + ".eps")
        plt.savefig(CF_NonNormalized_filename, format='eps', dpi=500, bbox_inches="tight")
    else:
        CF_Normalized_filename = os.path.join(eps_dir, Dataset + "_Video_Normalized_" + str(epoch) + "_Epoch_" + data_type + ".eps")
        plt.savefig(CF_Normalized_filename, format='eps', dpi=500, bbox_inches="tight")


def plot_train_vs_vald(train_points, vald_points, isloss=False):
    plot_label = 'Accuracy'
    if isloss:
        plot_label = 'Loss'

    epochs = range(1, len(train_points) + 1)

    lines1 = plt.plot(epochs, smooth_curve(train_points), label='Training ' + plot_label)
    plt.setp(lines1, color='red', linewidth=1.0)
    lines2 = plt.plot(epochs, smooth_curve(vald_points), 'b-', label='Validation ' + plot_label)
    plt.setp(lines2, color='black', linewidth=1.0)
    plt.title('Training and Validation ' + plot_label)
    plt.xlabel('Epochs')
    plt.ylabel(plot_label)
    plt.legend()
    filename = os.path.join(eps_dir, Dataset + "_Face_Training_Validation_" + plot_label + "_" + str(
        epoch) + "_Epoch.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


def plot_folds_accuracy(mdl_history):
    color_map = ['red', 'black', 'green', 'blue', 'magenta', 'cyan', 'yellow', 'orange', 'violet', 'pink']

    plt.title('Train Accuracy (T) vs Validation Accuracy (V)')

    pointslen = mdl_history[0].history['acc']
    pointslen = pointslen[start_point:]
    epochs = range(1, len(pointslen) + 1)

    for u in range(0, k):
        points1 = mdl_history[u].history['acc']
        points1 = points1[start_point:]
        lines1_1 = plt.plot(epochs, smooth_curve(points1), label='T Fold ' + str(u+1))
        plt.setp(lines1_1, color=color_map[u], linewidth=1.0)

        points2 = mdl_history[u].history['val_acc']
        points2 = points2[start_point:]
        lines1_2 = plt.plot(epochs, smooth_curve(points2), label='V Fold ' + str(u+1))
        plt.setp(lines1_2, color=color_map[u], linewidth=1.0, linestyle="dashdot")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=5)
    filename = os.path.join(eps_dir, Dataset + "_Face_Folds_Training_Validation_Accuracy_" + str(
        epoch) + "_Epoch.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


def plot_folds_loss(mdl_history):

    color_map = ['red', 'black', 'green', 'blue', 'magenta', 'cyan', 'yellow', 'orange', 'violet', 'pink']

    plt.title('Train Loss (T) vs Validation Loss (V)')

    pointslen = mdl_history[0].history['loss']
    pointslen = pointslen[start_point:]
    epochs = range(1, len(pointslen) + 1)

    for i in range(0, k):
        points1 = mdl_history[i].history['loss']
        points1 = points1[start_point:]
        lines1_1 = plt.plot(epochs, smooth_curve(points1), label='T Fold ' + str(i+1))
        plt.setp(lines1_1, color=color_map[i], linewidth=1.0)

        points2 = mdl_history[i].history['val_loss']
        points2 = points2[start_point:]
        lines1_2 = plt.plot(epochs, smooth_curve(points2), label='V Fold ' + str(i+1))
        plt.setp(lines1_2, color=color_map[i], linewidth=1.0, linestyle="dashdot")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=5)
    filename = os.path.join(eps_dir, Dataset + "_Face_Folds_Training_Validation_Loss_" + str(
        epoch) + "_Epoch.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


def plot_folds_barchart(t_data, v_data, fold_number, fold_dir):
    y_pos = np.arange(6)
    width = 0.32

    barfig = plt.figure(num=fold_number, figsize=(6, 4), dpi=80)
    ax = plt.subplot(111)

    t_values, t_counts = np.unique(t_data, return_counts=True)
    v_values, v_counts = np.unique(v_data, return_counts=True)
    rects1 = ax.bar(y_pos, t_counts, width, color='SkyBlue', alpha=0.5)
    rects2 = ax.bar(y_pos+width, v_counts, width, color='IndianRed', alpha=0.5)

    ax.set_ylabel('# Labels')
    ax.set_xlabel('Categories')
    ax.set_xticks(y_pos + width)
    ax.set_xticklabels(('Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise'))
    ax.legend((rects1[0], rects2[0]), ('train', 'validation'))
    # ax.set_title('Distribution of labels in each category')

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    filename = os.path.join(fold_dir, Dataset + "_Face_Training_Validation_Fold_" +
                            str(fold_number) + "_Distribution_" + str(epoch) + "_Epoch.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")
    plt.close(barfig)


def autolabel(ax, rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d' % int(h),
                ha='center', va='bottom')


# Compute and plot ROC curve and ROC area for each class
def plot_roc(y_test, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculating auc, false positive rate and true positive rate for each class
    for r in range(classes_num):
        fpr[r], tpr[r], _ = roc_curve(y_test[:, r], y_score[:, r])
        roc_auc[r] = auc(fpr[r], tpr[r])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[bb] for bb in range(classes_num)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for ii in range(classes_num):
        mean_tpr += interp(all_fpr, fpr[ii], tpr[ii])

    # Finally average it and compute AUC
    mean_tpr /= classes_num

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # smoothing the micro roc curve
    # micro_poly = np.polyfit(fpr["micro"], tpr["micro"], 5)
    # micro_poly_y = np.poly1d(micro_poly)(fpr["micro"])
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='magenta', linestyle=':', linewidth=1)

    # smoothing the macro roc curve
    # macro_poly = np.polyfit(fpr["macro"], tpr["macro"], 5)
    # macro_poly_y = np.poly1d(macro_poly)(fpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='blue', linestyle='-.', linewidth=1)

    plt.plot([0, 1], [0, 1], color='silver', linestyle='--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for each class')
    plt.legend(loc="lower right")
    filename = os.path.join(eps_dir, Dataset + "_Face_Micro_Macro_Avg_ROC_Curve_" + str(epoch) + "_Epoch.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


# plotting learning rate
def plot_lr(lr_points):

    epochs = range(1, len(lr_points) + 1)

    lines1 = plt.plot(epochs, lr_points, label='learning rate')
    plt.setp(lines1, color='black', linewidth=1.0)

    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    filename = os.path.join(eps_dir, Dataset + "_Face_Learning_Rate_" + str(epoch) + "_Epoch.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


# Saving Hyperparamters of the model
def save_settings():
    setting_info = "Dataset = " + str(Dataset)
    setting_info = setting_info + "\nSaving folder Path =" + random_str
    setting_info = setting_info + "\nSeed for Random Numbers = " + str(seednum)
    setting_info = setting_info + "\nNumber of Folds = " + str(k)
    setting_info = setting_info + "\nNumber of Epochs In Training = " + str(epoch)
    setting_info = setting_info + "\nNumber of Epochs After Training = " + str(fitepoch)
    setting_info = setting_info + "\nBatchsize = " + str(batchsize)
    setting_info = setting_info + "\nMinimum Learning Rate = " + str(min_lr)
    setting_info = setting_info + "\nLearning Rate = " + str(base_lr)
    setting_info = setting_info + "\nMaximum Learning Rate = " + str(max_lr)
    setting_info = setting_info + "\nLearning Rate decay factor = " + str(lr_factor)
    setting_info = setting_info + "\nLearning Rate Patience = " + str(patience)
    setting_info = setting_info + "\nDropout Rate = " + str(dropout_rate)
    setting_info = setting_info + "\nSmoothing Factor = " + str(smooth_factor)
    setting_info = setting_info + "\nFeatures Saved For Layer = " + str(layer_name)
    setting_info = setting_info + "\nStarting Point = " + str(start_point)
    setting_info = setting_info + "\nData Path = " + str(base_dir)
    setting_info = setting_info + "\nData Type = " + str(data_type)
    setting_info = setting_info + "\nShuffle = " + str(shuffle)
    setting_info = setting_info + "\nCallbacks = " + callbacks_list_str
    setting_info = setting_info + "\nTrain accuracy = " + str(train_acc)
    setting_info = setting_info + "\nTrain loss = " + str(train_loss)
    setting_info = setting_info + "\nValidation accuracy = " + str(np.mean(vald_acc))
    setting_info = setting_info + "\nValidation loss = " + str(np.mean(vald_loss))
    setting_info = setting_info + "\nTest accuracy = " + str(test_acc)
    setting_info = setting_info + "\nTest loss = " + str(test_loss)
    setting_info = setting_info + "\nProcess Time in seconds = " + str(process_time)
    return setting_info


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
    lrate = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=patience, min_lr=min_lr, mode='min', verbose=1)
    if typ == "step_decay":
        lrate = LearningRateScheduler(step_decay)
    elif typ == "lambda":
        lrate = LearningRateScheduler(lambda epoch: base_lr * 0.99 ** epoch)
    elif typ == "cyclic":
        lrate = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=500., mode='exp_range', gamma=0.99994)

    return lrate


def set_optimizer(typ="Adam"):
    # choosing between different optimizers
    optimizer = Adam(lr=base_lr, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=0.0)

    if typ == "SGD":
        optimizer = SGD(lr=base_lr, decay=0.0, momentum=0.9, nesterov=True)
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


# 3D model
def create_model():
    """ Return the Keras model of the network
    """
    model = models.Sequential()
    model.add(layers.Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                            input_shape=(25, image_width, image_height, 3),
                            kernel_initializer='random_uniform', activation="relu", name="conv3d_1",
                            kernel_regularizer=l2(0.0), bias_initializer='random_uniform', padding='SAME'))
    model.add(layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='VALID', name="maxpool3d_1"))

    model.add(layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                            kernel_initializer='random_uniform', activation="relu", name="conv3d_2",
                            kernel_regularizer=l2(0.0), bias_initializer='random_uniform', padding='SAME'))
    model.add(layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='VALID', name="maxpool3d_2"))

    model.add(layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                            kernel_initializer='random_uniform', activation="relu", name="conv3d_31",
                            kernel_regularizer=l2(0.0), bias_initializer='random_uniform', padding='SAME'))
    model.add(layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                            kernel_initializer='random_uniform', activation="relu", name="conv3d_32",
                            kernel_regularizer=l2(0.0), bias_initializer='random_uniform', padding='SAME'))
    model.add(layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='VALID', name="maxpool3d_3"))

    model.add(layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                            kernel_initializer='random_uniform', activation="relu", name="conv3d_4",
                            kernel_regularizer=l2(0.0), bias_initializer='random_uniform', padding='SAME'))
    model.add(layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='VALID', name="maxpool3d_4"))

    model.add(layers.Conv3D(filters=256, kernel_size=(1, 5, 5), strides=(1, 1, 1),
                            kernel_initializer='random_uniform', activation="relu", name="conv3d_5",
                            kernel_regularizer=l2(0.0), bias_initializer='random_uniform', padding='SAME'))

    model.add(layers.Dropout(dropout_rate, name="dropout"))
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(64, activation='relu', name="fc6"))
    model.add(layers.Dense(classes_num, activation='softmax'))

    # set the optimizer
    optimizer = set_optimizer(typ="Adam")

    # get lr value after each epoch
    lr_metric = get_lr_metric(optimizer)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', lr_metric])

    return model


idx = 0
start = time.clock()

for train, vald in kfold.split(train_data, train_labels):
    fold_number = idx + 1
    fold_folder_name = "Fold_0" + str(fold_number)
    if fold_number == 10:
        fold_folder_name = "Fold_" + str(fold_number)
    os.makedirs(os.path.join(cur_res_dir, "Folds", fold_folder_name))
    fold_dir = os.path.join(cur_res_dir, "Folds", fold_folder_name)
    checkpoint_dir = os.path.join(fold_dir,
                                  'weights-improvement-{epoch:03d}-{acc:.2f}-{loss:.2f}-{val_acc:.2f}-{val_loss:.2f}.hdf5')

    # plot the distribution of sample points of each category among folds
    plot_folds_barchart(train_labels[train], train_labels[vald], fold_number, eps_dir)

    # save indices of sample points within each fold
    save_csv(train, flag="Train", name="Fold_"+str(fold_number)+"_Indices", path=idx_dir)
    save_csv(vald, flag="Validation", name="Fold_" + str(fold_number) + "_Indices", path=idx_dir)

    print("< ------------ Fold Number ---------->", fold_number)
    # convert to one hot coded
    train_labels_one_hot_coded = to_categorical(y=train_labels[train], num_classes=6)
    vald_labels_one_hot_coded = to_categorical(y=train_labels[vald], num_classes=6)

    # checkpoint results
    checkpoint = set_checkpoint()

    # set learning schedule
    lrate = set_lr(typ="reduce_lr")

    # set callbacks
    callbacks_list = [checkpoint, lrate]
    callbacks_list_str = "[checkpoint, reduce_lr]"

    # create model
    model = create_model()
    if idx == 0:
        print(model.summary())

    # Fit the model
    history = model.fit(train_data[train], train_labels_one_hot_coded,
                        epochs=epoch, batch_size=batchsize, shuffle=shuffle,
                        validation_data=(train_data[vald], vald_labels_one_hot_coded),
                        callbacks=callbacks_list)
    model_history.append(history)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    tr_lrr = history.history['lr']

    history_train_acc.append(acc)
    history_vald_acc.append(val_acc)
    history_train_loss.append(loss)
    history_vald_loss.append(val_loss)
    history_lrr.append(tr_lrr)

    if fold_number == 10:
        save_csv(history_train_acc, "Train", "AllFolds_Accuracy_Details")
        save_csv(history_vald_acc, "Validation", "AllFolds_Accuracy_Details")
        save_csv(history_train_loss, "Train", "AllFolds_Loss_Details")
        save_csv(history_vald_loss, "Validation", "AllFolds_Loss_Details")
        save_csv(history_lrr, "Train", "AllFolds_LearningRate_Details")

    idx += 1

# averaging the accuracy and loss over all folds
train_acc = [np.mean([x[i] for x in history_train_acc]) for i in range(epoch)]
vald_acc = [np.mean([x[i] for x in history_vald_acc]) for i in range(epoch)]
train_loss = [np.mean([x[i] for x in history_train_loss]) for i in range(epoch)]
vald_loss = [np.mean([x[i] for x in history_vald_loss]) for i in range(epoch)]
train_lr = [np.mean([x[i] for x in history_lrr]) for i in range(epoch)]

# saving the average results from folds
save_csv(train_labels, flag="Train", name="True_Labels")
save_csv(train_acc, flag="Train", name="Averaged_Accuracy")
save_csv(vald_acc, flag="Validation", name="Averaged_Accuracy")
save_csv(train_loss, flag="Train", name="Averaged_Loss")
save_csv(vald_loss, flag="Validation", name="Averaged_Loss")
save_csv(train_lr, flag="Train", name="Averaged_LearningRate")

# saving the average smoothed results from folds
save_csv(smooth_curve(train_acc), flag="Train", name="Averaged_Accuracy_Smoothed_Points")
save_csv(smooth_curve(vald_acc), flag="Validation", name="Averaged_Accuracy_Smoothed_Points")
save_csv(smooth_curve(train_loss), flag="Train", name="Averaged_Loss_Smoothed_Points")
save_csv(smooth_curve(vald_loss), flag="Validation", name="Averaged_Loss_Smoothed_Points")

plt.figure(num=11, figsize=(8, 6), dpi=80)
plot_folds_accuracy(model_history)

plt.figure(num=12, figsize=(8, 6), dpi=80)
plot_folds_loss(model_history)

plt.figure(num=13, figsize=(8, 6), dpi=80)
plot_train_vs_vald(train_loss[start_point:], vald_loss[start_point:], isloss=True)

plt.figure(num=14, figsize=(8, 6), dpi=80)
plot_train_vs_vald(train_acc[start_point:], vald_acc[start_point:])

plt.figure(num=15, figsize=(8, 6), dpi=80)
plot_lr(train_lr[start_point:])

train_labels_one_hot_coded = to_categorical(y=train_labels, num_classes=6)
model = create_model()
model.fit(train_data, train_labels_one_hot_coded, epochs=fitepoch, batch_size=batchsize)

end = time.clock()
process_time = (end - start)

# for layer_num in range(1,7):
# save_layer_output(train_data, layer_num, feature_dir)


# serialize model to JSON
model_json = model.to_json()
with open(os.path.join(cur_res_dir, Dataset + "_Face_model.json"), "w") as json_file:
    json_file.write(model_json)

# evaluation on train
train_loss, train_acc, train_lr = model.evaluate(train_data, train_labels_one_hot_coded, batch_size=1)
print(train_loss, train_acc, train_lr)

train_predicted_probs = model.predict(train_data, batch_size=1)
train_predicted_labels = train_predicted_probs.argmax(axis=-1)

# Saving Train results
save_npy(train_predicted_probs, flag="Train", name="Probabilities")
save_npy(train_predicted_labels, flag="Train", name="ClassLabels")
save_csv(train_predicted_probs, flag="Train", name="Probabilities")
save_csv(train_predicted_labels, flag="Train", name="ClassLabels")


# evaluation on Test
test_loss, test_acc, test_lr = model.evaluate(test_data, test_labels_one_hot_coded, batch_size=1)
print(test_loss, test_acc, test_lr)


# Confution Matrix and Classification Report
test_predicted_probs = model.predict(test_data, batch_size=1)
test_predicted_labels = test_predicted_probs.argmax(axis=-1)


# saving features
save_layer_output(train_data, path=feature_dir, name="Train")
save_layer_output(test_data, path=feature_dir, name="Test")

# Plot all ROC curves
plt.figure(num=16, figsize=(8, 6), dpi=80)
plot_roc(test_labels_one_hot_coded, test_predicted_probs)

# Saving Test Results
save_npy(test_predicted_probs, flag="Test", name="Probabilities")
save_npy(test_predicted_labels, flag="Test", name="ClassLabels")
save_csv(test_predicted_probs, flag="Test", name="Probabilities")
save_csv(test_predicted_labels, flag="Test", name="ClassLabels")

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_labels, test_predicted_labels)
np.set_printoptions(precision=2)


# Plot and save non-normalized confusion matrix
plt.figure(num=17, figsize=(5, 5), dpi=80)
plot_confusion_matrix(cnf_matrix, classes=class_names)
CF_NonNormalized_filename = os.path.join(eps_dir,
                                         Dataset + "_Face_NonNormalized_" + str(epoch) + "_Epoch_" + data_type + ".eps")
plt.savefig(CF_NonNormalized_filename, format='eps', dpi=1000, bbox_inches="tight")

# Plot normalized confusion matrix
plt.figure(num=18, figsize=(5, 5), dpi=80)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)
CF_Normalized_filename = os.path.join(eps_dir,
                                      Dataset + "_Face_Normalized_" + str(epoch) + "_Epoch_" + data_type + ".eps")
plt.savefig(CF_Normalized_filename, format='eps', dpi=1000, bbox_inches="tight")

cnf_matrix2 = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
print(np.average(cnf_matrix2.diagonal()))

HyperParameter_Setting = save_settings()
with open(os.path.join(cur_res_dir, Dataset + "_Face_HyperParameters.txt"), "w") as text_file:
    text_file.write(HyperParameter_Setting)

print(HyperParameter_Setting)
shutil.copyfile(os.path.join(file_path, Dataset + "_AudioBase_Video_CNN_KFold.py"),
                os.path.join(cur_res_dir, Dataset + "_AudioBase_Video_CNN_KFold.txt"))
plt.show(block=True)
