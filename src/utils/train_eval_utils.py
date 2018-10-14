# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

import numpy as np
from tensorflow import keras as kr
from sklearn.metrics import precision_recall_fscore_support


def plot_train_history(train_history, acc_filename, loss_filename):
    # plot history for accuracy
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    with PdfPages(acc_filename) as page:
        plt.savefig(page, format='pdf')
        plt.close()
    # plot history for loss
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    with PdfPages(loss_filename) as page:
        plt.savefig(page, format='pdf')
        plt.close()


class Metrics(kr.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        val_prediction = (np.asarray(self.model.predict(self.validation_data[:-3]))).round()
        val_label = self.validation_data[-3]
        _, _, f1, _ = precision_recall_fscore_support(val_label, val_prediction, average='macro')
        print(' - val_macro_f1:{}'.format(round(f1, 4)))
        return