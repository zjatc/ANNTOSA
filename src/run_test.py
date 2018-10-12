# -*- coding: utf-8 -*-

import argparse
import numpy as np
import yaml
import utils.preprocess as dp
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support

from model import TBSAModel

parser = argparse.ArgumentParser(description='Test TBSA model')
parser.add_argument('-data', type=str, default=None, help='Path of test data')
parser.add_argument('--conf-file', dest='config_file', type=argparse.FileType(mode='r'))
args = parser.parse_args()


def load_conf():
    if args.config_file:
        train_conf = yaml.load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in train_conf.__dict__.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        arg_dict['is_train'] = False
        print('configs:\n{}\n'.format(args.__dict__))
    else:
        raise Exception('no configuration file specified')


def process_dataset():
    test_dataset = dp.load_dataset(args.data, is_english=True, has_label=True, use_target='word', use_first_target=False)
    print('text seq len: ', args.text_seq_len)
    print('tar seq len: ', args.tar_seq_len)
    _, token2id = dp.build_vocab(args.vocab, data=[ins['text_words'] for ins in test_dataset], is_train=False)

    # list of tuple (encoded_text, encoded_target, encoded_idx, encoded_label)
    test_encoded = dp.build_dataset(test_dataset, token2id, args.cat2id, args.text_seq_len, args.tar_seq_len)
    test_text, test_target, test_tar_idx, test_label = map(lambda filed: np.array(filed), zip(*test_encoded))

    return args, test_text, test_target, test_tar_idx, test_label


def test(model, algorithm, test_text, test_target, test_tar_idx, bs=256, verbose=1, save_prob=False):
    if algorithm == "seg_tar":
        pred_prob = model.predict([test_text, test_target, test_tar_idx], batch_size=bs, verbose=verbose)
    else:
        pred_prob = model.predict([test_text, test_tar_idx], batch_size=bs, verbose=verbose)

    if save_prob:
        np.savetxt('predict-prob.txt', pred_prob)

    if pred_prob.shape[-1] > 1:
        pred = pred_prob.argmax(axis=-1)
    else:
        pred = (pred_prob > 0.5).astype('int32')

    return pred


if __name__ == '__main__':
    load_conf()
    params, text, target, tar_idx, label = process_dataset()

    model = TBSAModel(params).model
    model.load_weights(params.model_file)
    #model = kr.models.load_model(params.model_file)
    #model.summary()

    preds = test(model, params.al, text, target, tar_idx)
    ground_truth = label.argmax(axis=-1)

    # evaluation
    target_names = ['negative', 'neutral', 'positive']
    print(classification_report(ground_truth, preds, target_names=target_names, digits=4))
    print('Accuracy: %.4f' % (accuracy_score(ground_truth, preds)))
    #print(confusion_matrix(ground_truth, preds))
    p, r, f, _ = precision_recall_fscore_support(ground_truth, preds, average='macro')
    print('Macro-based: %.4f\t%.4f\t%.4f' % (p, r, f))
    p, r, f, _ = precision_recall_fscore_support(ground_truth, preds, average='micro')
    print('Micro-based: %.4f\t%.4f\t%.4f' % (p, r, f))
    p, r, f, _ = precision_recall_fscore_support(ground_truth, preds, average='weighted')
    print('Weighted-based: %.4f\t%.4f\t%.4f' % (p, r, f))
