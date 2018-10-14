# -*- coding: utf-8 -*-

import os
import numpy as np
from tensorflow import keras as kr
from tensorflow import set_random_seed
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import yaml
from conf import get_eval_args
import utils.preprocess as dp
from model import TBSAModel

np.random.seed(20180510)
set_random_seed(20180510)
pad_dataset = False
args = get_eval_args()


def process_dataset():
    train_dataset = dp.load_dataset(args.data, is_english=True, has_label=True, use_target='word', use_first_target=False)
    # Y = [ins['label'] for ins in train_dataset]
    # class_distri = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
    # print('Set class weights: {}'.format(class_distri))
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
    train_encoded = dp.build_dataset(train_dataset, token2id, args.cat2id, args.text_seq_len, args.tar_seq_len)
    np.random.shuffle(train_encoded)
    if args.dev:
        val_encoded = dp.build_dataset(val_dataset, token2id, args.cat2id, args.text_seq_len, args.tar_seq_len)
    else:
        train_encoded, val_encoded = train_test_split(train_encoded, test_size=0.2, random_state=1314)
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
    train_text, _, train_tar_idx, train_label = train_dataset
    val_text, _, val_tar_idx, val_label = val_dataset
    test_text, _, test_tar_idx, test_label = test_dataset

    metrics = []
    for i in range(config.run_n):
        print('\nround {}\n'.format(i+1))
        model = TBSAModel(config).model
        callbacks = []
        #callbacks.append(Metrics())
        if config.use_early_stop:
            callbacks.append(kr.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=0))
        if config.use_lr_decay:
            # lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: config.lr * (0.9 ** epoch)) 
            callbacks.append(kr.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
            patience=3, verbose=1, mode='auto', min_delta=0.0001, cooldown=1, min_lr=0))
        
        save_model_path = './{}_tmp_{}.h5'.format(config.al, i+1) if config.model_file is None else config.model_file
        callbacks.append(kr.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_acc',
                                                save_best_only=True, save_weights_only=True, verbose=0))
        if not callbacks:
            callbacks = None
        history = model.fit([train_text, train_tar_idx], train_label, batch_size=config.batch_size,
                                epochs=config.max_epoch, verbose=2, shuffle=True,
                                validation_data=([val_text, val_tar_idx], val_label), callbacks=callbacks)
        best_epoch = np.argmax(history.history['val_acc']) + 1
        print('\nhighest validation accuracy is {} in epoch {}'.format(round(max(history.history['val_acc']), 4), best_epoch))
        # evaluation
        model.load_weights(save_model_path)
        pred_prob = model.predict([test_text, test_tar_idx], batch_size=1000, verbose=0)
        pred =  pred_prob.argmax(axis=-1) if pred_prob.shape[-1]>1 else (pred_prob > 0.5).astype('int32')
        ground_truth = test_label.argmax(axis=-1)
        # target_names = ['negative', 'neutral', 'positive']
        # print(classification_report(ground_truth, pred, target_names=target_names, digits=4))
        acc = accuracy_score(ground_truth, pred)
        _, _, f, _ = precision_recall_fscore_support(ground_truth, pred, average='macro')
        metrics.append([round(acc,4), round(f,4), best_epoch])
        os.remove(save_model_path)
    acc_n, f_n, _ = map(lambda filed: np.array(filed), zip(*metrics))
    print('\n{} round metrics: \n{}\n'.format(config.run_n, np.asarray(metrics)))
    print("acc max: %.4f, f1 max: %.4f"%(np.max(acc_n), f_n[np.argmax(acc_n)]))
    print("acc mean: %.4f, acc std: %.4f, f1 mean: %.4f, f1 std: %.4f\n" % (np.mean(acc_n), np.std(acc_n),np.mean(f_n), np.std(f_n)))
    
    kr.backend.clear_session()
    # plot_train_history(history, params.log_acc, params.log_loss)


if __name__ == '__main__':
    params, train_ds, val_ds, test_ds = process_dataset()
    train_and_test(params, train_ds, val_ds, test_ds)
    # save model configuration
    # if params.model_file:
    #     for arg in NOT_SAVED_ARGS:
    #         delattr(params, arg)
    #     with open(params.model_file[:params.model_file.rfind('.')]+'.conf', 'w', encoding='utf-8') as conf_out:
    #         yaml.dump(params, conf_out, default_flow_style=False)
