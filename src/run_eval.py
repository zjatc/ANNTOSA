# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from tensorflow import keras as kr
from tensorflow import set_random_seed
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import utils.preprocess as dp
from model import TBSAModel

np.random.seed(20180510)
#set_random_seed(20180510)
pad_dataset = True


parser = argparse.ArgumentParser(description='TBSA model')
parser.add_argument('-data', type=str, default=None, help='File path of data set used as training data, test data, or visualization data')
parser.add_argument('-dev', type=str, default=None, help='File path of validation data')
parser.add_argument('-test', type=str, default=None, help='File path of test data')
parser.add_argument('-emb', type=str, default=None, help='File path of pre-trained embedding')
parser.add_argument('-v', '--vocab', type=str, default=None, help='File path of vocabulary file')
parser.add_argument('-tf_limit', type=int, default=0)

parser.add_argument('-m', '--model_file', type=str, default=None, help='Path for saving model')
parser.add_argument('-acc', '--log_acc', type=str, default=None, help='File path of acc log file')
parser.add_argument('-loss', '--log_loss', type=str, default=None, help='File path of loss log file')

parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-max_epoch', type=int, default=50)
parser.add_argument('-use_early_stop', action='store_true', default=False)

parser.add_argument('-al', type=str, default='seg', help='one of seg, t2t, seg_tar')
parser.add_argument('-emb_dp', type=float, default=0.5)
parser.add_argument('-out_dp', type=float, default=0.5)

parser.add_argument('-lstm_dim', type=int, default=150)

parser.add_argument('-struc_dim', type=int, default=150)
parser.add_argument('-struc_r', type=int, default=8)
parser.add_argument('-struc_penal', type=float, default=1e-4)

parser.add_argument('-t2t_dim', type=int, default=64)
parser.add_argument('-tensor_k', type=int, default=4)
parser.add_argument('-align_model', type=str, default='tensor')

parser.add_argument('-is_train', type=bool, default=True)
args = parser.parse_args()

cat2id = {'-1': 0, '0': 1, '1': 2}
args.cat2id = cat2id
args.class_num = len(cat2id)

NOT_SAVED_ARGS = ['data', 'dev', 'log_acc', 'log_loss', 'embeddings', 'emb', 'is_train', 'tf_limit']


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


def process_dataset():
    train_dataset = dp.load_dataset(args.data, is_english=True, has_label=True, use_target='word', use_first_target=False)
    val_dataset = None
    if args.dev:
        val_dataset = dp.load_dataset(args.dev, is_english=True, has_label=True, use_target='word', use_first_target=False)
        full_dataset = train_dataset + val_dataset
    else:
        full_dataset = train_dataset
    args.text_seq_len = max([len(ins['text_words']) for ins in full_dataset])
    args.tar_seq_len = max([len(ins['tar_words']) for ins in full_dataset])
    print('text seq len: ', args.text_seq_len)
    print('tar seq len: ', args.tar_seq_len)
    args.vocab_size, token2id, args.embeddings, args.emb_dim = dp.build_vocab(args.vocab, data=[ins['text_words'] for ins in full_dataset],
                                                                              embedding=args.emb, tf_limit=args.tf_limit)
    # list of tuple (encoded_text, encoded_target, encoded_idx, encoded_label)
    train_encoded = dp.build_dataset(train_dataset, token2id, cat2id, args.text_seq_len, args.tar_seq_len)
    np.random.shuffle(train_encoded)
    if args.dev:
        val_encoded = dp.build_dataset(val_dataset, token2id, cat2id, args.text_seq_len, args.tar_seq_len)
    else:
        train_encoded, val_encoded = train_test_split(train_encoded, test_size=0.2, random_state=1234)
    if pad_dataset:
        train_encoded = dp.pad_dataset(train_encoded, args.batch_size)
    train_text, train_target, train_tar_idx, train_label = map(lambda filed: np.array(filed), zip(*train_encoded))
    val_text, val_target, val_tar_idx, val_label = map(lambda filed: np.array(filed), zip(*val_encoded))

    # load test dataset
    test_dataset = dp.load_dataset(args.test, is_english=True, has_label=True, use_target='word', use_first_target=False)
    test_encoded = dp.build_dataset(test_dataset, token2id, args.cat2id, args.text_seq_len, args.tar_seq_len)
    test_text, test_target, test_tar_idx, test_label = map(lambda filed: np.array(filed), zip(*test_encoded))

    return args, (train_text, train_target, train_tar_idx, train_label), (val_text, val_target, val_tar_idx, val_label), \
           (test_text, test_target, test_tar_idx, test_label)


def train_and_test(config, train_dataset, val_dataset, test_dataset):
    train_text, train_target, train_tar_idx, train_label = train_dataset
    val_text, val_target, val_tar_idx, val_label = val_dataset
    test_text, test_target, test_tar_idx, test_label = test_dataset

    #class_distri = class_weight.compute_class_weight('balanced', np.unique(train_label), train_label)
    #print('Set class weights: {}'.format(class_distri))

    model = TBSAModel(config).model
    # optimizer = kr.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
    optimizer = kr.optimizers.RMSprop(lr=config.lr, rho=0.9, epsilon=1e-8, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    callbacks = []
    #callbacks.append(Metrics())
    if config.use_early_stop:
        earlystop = kr.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=0)
        callbacks.append(earlystop)
    if config.model_file is None:
        save_model_path = './{}_tmp_{}.h5'.format(config.al, os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        save_model_path = config.model_file
    checkpoint = kr.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_acc',
                                              save_best_only=True, save_weights_only=True, verbose=1)
    callbacks.append(checkpoint)
    if not callbacks:
        callbacks = None

    if config.al == "xx":
        history = model.fit([train_text, train_target, train_tar_idx], train_label, batch_size=config.batch_size,
                            epochs=config.max_epoch, verbose=1, shuffle=True,
                            validation_data=([val_text, val_target, val_tar_idx], val_label), callbacks=callbacks)
        model.load_weights(save_model_path)
        pred_prob = model.predict([test_text, test_target, test_tar_idx], batch_size=1000, verbose=1)
    else:
        history = model.fit([train_text, train_tar_idx], train_label, batch_size=config.batch_size,
                            epochs=config.max_epoch, verbose=2, shuffle=True,
                            validation_data=([val_text, val_tar_idx], val_label), callbacks=callbacks)
        model.load_weights(save_model_path)
        pred_prob = model.predict([test_text, test_tar_idx], batch_size=1000, verbose=1)
    print('highest validation accuracy is {}'.format(round(max(history.history['val_acc']), 4)))
    # evaluation
    if pred_prob.shape[-1] > 1:
        pred = pred_prob.argmax(axis=-1)
    else:
        pred = (pred_prob > 0.5).astype('int32')
    ground_truth = test_label.argmax(axis=-1)
    target_names = ['negative', 'neutral', 'positive']
    print(classification_report(ground_truth, pred, target_names=target_names, digits=4))
    print('Test Accuracy: %.4f' % (accuracy_score(ground_truth, pred)))
    p, r, f, _ = precision_recall_fscore_support(ground_truth, pred, average='macro')
    print('Test Macro-based: %.4f\t%.4f\t%.4f' % (p, r, f))
    kr.backend.clear_session()
    os.remove(save_model_path)
    # plot_train_history(history, params.log_acc, params.log_loss)


if __name__ == '__main__':
    params, train_ds, val_ds, test_ds = process_dataset()
    train_and_test(params, train_ds, val_ds, test_ds)
    # save model configuration
    if params.model_file:
        for arg in NOT_SAVED_ARGS:
            delattr(params, arg)
        with open(params.model_file[:params.model_file.rfind('.')]+'.conf', 'w', encoding='utf-8') as conf_out:
            yaml.dump(params, conf_out, default_flow_style=False)
