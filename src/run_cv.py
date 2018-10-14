# -*- coding: utf-8 -*-

import os
import numpy as np
from tensorflow import keras as kr
from tensorflow import set_random_seed
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from conf import get_cv_args
import utils.preprocess as dp
from model import TBSAModel

np.random.seed(20180510)
set_random_seed(20180510)
pad_dataset = False
args = get_cv_args()

def process_dataset_cv():
    train_dataset = dp.load_dataset(args.data, is_english=True, has_label=True, use_target='word', use_first_target=False)
    if args.test:
        test_dataset = dp.load_dataset(args.test, is_english=True, has_label=True, use_target='word', use_first_target=False) 
        full_dataset = train_dataset + test_dataset
    else:
        full_dataset = train_dataset
    args.text_seq_len = max([len(ins['text_words']) for ins in full_dataset])
    args.tar_seq_len = max([len(ins['tar_words']) for ins in full_dataset])
    print('text seq len: ', args.text_seq_len)
    print('tar seq len: ', args.tar_seq_len)
    args.vocab_size, token2id, args.embeddings, args.emb_dim = dp.build_vocab(args.vocab, data=[ins['text_words'] for ins in full_dataset],
                                                                              embedding=args.emb, tf_limit=args.tf_limit)
    # list of tuple (encoded_text, encoded_target, encoded_idx, encoded_label)
    data_encoded = dp.build_dataset(full_dataset, token2id, args.cat2id, args.text_seq_len, args.tar_seq_len)
    text, target, tar_idx, label = map(lambda filed: np.array(filed), zip(*data_encoded))
    Y = [ins['label'] for ins in full_dataset]
    fold = list(StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=1234).split(text, Y))
    return args, fold, (text, target, tar_idx), label


def kfold_cv(config, fold, data, label):
    X_text, _, X_tar_idx = data
    cv_scores = []
    for k, (train, test) in enumerate(fold):
        print('\nFold {}:\n'.format(k))
        model = TBSAModel(config).model
        callbacks = []
        #callbacks.append(Metrics())
        if config.use_early_stop:
            earlystop = kr.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)
            callbacks.append(earlystop)
        if config.model_file is None:
            save_model_path = './{}_tmp.h5'.format(config.al)
        else:
            save_model_path = config.model_file
        checkpoint = kr.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_acc',
                                                save_best_only=True, save_weights_only=True, verbose=0)
        callbacks.append(checkpoint)
        if not callbacks:
            callbacks = None
  
        history = model.fit([X_text[train], X_tar_idx[train]], label[train], batch_size=config.batch_size,
                            epochs=config.max_epoch, verbose=2, shuffle=True,
                            validation_data=([X_text[test], X_tar_idx[test]], label[test]), callbacks=callbacks)
        # evaluate the model
        model.load_weights(save_model_path)
        pred_prob = model.predict([X_text[test], X_tar_idx[test]], batch_size=1000, verbose=0)
        # evaluation
        if pred_prob.shape[-1] > 1:
            pred = pred_prob.argmax(axis=-1)
        else:
            pred = (pred_prob > 0.5).astype('int32')
        ground_truth = label[test].argmax(axis=-1)
        acc = accuracy_score(ground_truth, pred)
        _, _, f, _ = precision_recall_fscore_support(ground_truth, pred, average='macro')
        print('Test Accuracy: %.4f, Test F1: %.4f' % (acc, f))
        best_epoch = np.argmax(history.history['val_acc'])
        cv_scores.append((round(acc,4), round(f,4), best_epoch+1))
        os.remove(save_model_path)
    acc_k, f_k, _ = map(lambda filed: np.array(filed), zip(*cv_scores))
    print('\n{} folder metrics: {}'.format(config.kfold, cv_scores))
    print("acc mean: %.4f, acc std: %.4f, f1 mean: %.4f, f1 std: %.4f\n" % (np.mean(acc_k), np.std(acc_k),np.mean(f_k), np.std(f_k)))
    kr.backend.clear_session()
    # plot_train_history(history, params.log_acc, params.log_loss)


if __name__ == '__main__':
    params, cv_folder, X, Y = process_dataset_cv()
    kfold_cv(params, cv_folder, X, Y)