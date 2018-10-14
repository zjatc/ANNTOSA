# -*- coding: utf-8 -*-


import numpy as np
from tensorflow import keras as kr
from tensorflow import set_random_seed
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import yaml
from conf import get_train_args, NOT_SAVED_ARGS
import utils.preprocess as dp
from model import TBSAModel

np.random.seed(20180510)
set_random_seed(20180510)
pad_dataset = True
args = get_train_args()


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
    train_encoded = dp.build_dataset(train_dataset, token2id, args.cat2id, args.text_seq_len, args.tar_seq_len)
    np.random.shuffle(train_encoded)
    if args.dev:
        val_encoded = dp.build_dataset(val_dataset, token2id, args.cat2id, args.text_seq_len, args.tar_seq_len)
    else:
        train_encoded, val_encoded = train_test_split(train_encoded, test_size=0.2, random_state=1234)
    if pad_dataset:
        train_encoded = dp.pad_dataset(train_encoded, args.batch_size)
    train_text, train_target, train_tar_idx, train_label = map(lambda filed: np.array(filed), zip(*train_encoded))
    val_text, val_target, val_tar_idx, val_label = map(lambda filed: np.array(filed), zip(*val_encoded))

    return args, (train_text, train_target, train_tar_idx, train_label), (val_text, val_target, val_tar_idx, val_label)


def train(config, train_dataset, val_dataset):
    train_text, train_target, train_tar_idx, train_label = train_dataset
    val_text, val_target, val_tar_idx, val_label = val_dataset

    #class_distri = class_weight.compute_class_weight('balanced', np.unique(train_label), train_label)
    #print('Set class weights: {}'.format(class_distri))

    model = TBSAModel(config).model
    callbacks = []
    #callbacks.append(Metrics())
    if config.use_early_stop:
        earlystop = kr.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=0)
        callbacks.append(earlystop)
    if config.model_file is not None:
        checkpoint = kr.callbacks.ModelCheckpoint(filepath=config.model_file, monitor='val_acc',
                                                  save_best_only=True, save_weights_only=True, verbose=1)
        callbacks.append(checkpoint)
    if not callbacks:
        callbacks = None

    if config.al == "seg_tar":
        history = model.fit([train_text, train_target, train_tar_idx], train_label, batch_size=config.batch_size,
                            epochs=config.max_epoch, verbose=1, shuffle=True,
                            validation_data=([val_text, val_target, val_tar_idx], val_label), callbacks=callbacks)
    else:
        history = model.fit([train_text, train_tar_idx], train_label, batch_size=config.batch_size,
                            epochs=config.max_epoch, verbose=2, shuffle=True,
                            validation_data=([val_text, val_tar_idx], val_label), callbacks=callbacks)
    print('highest accuracy is {}'.format(max(history.history['val_acc'])))
    kr.backend.clear_session()
    # plot_train_history(history, params.log_acc, params.log_loss)


if __name__ == '__main__':
    params, train_ds, val_ds = process_dataset()
    train(params, train_ds, val_ds)
    # save model configuration
    if params.model_file:
        for arg in NOT_SAVED_ARGS:
            delattr(params, arg)
        with open(params.model_file[:params.model_file.rfind('.')]+'.conf', 'w', encoding='utf-8') as conf_out:
            yaml.dump(params, conf_out, default_flow_style=False)
