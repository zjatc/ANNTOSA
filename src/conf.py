# -*- coding: utf-8 -*-

import argparse

Label2ID = {'-1': 0, '0': 1, '1': 2}
NOT_SAVED_ARGS = ['data', 'dev', 'log_acc', 'log_loss', 'embeddings', 'emb', 'is_train', 'tf_limit']

def add_common_args(stage):
    parser = argparse.ArgumentParser(description='TBSA model {} args'.format(stage))
    parser.add_argument('-data', type=str, default=None, help='File path of data set used as training data, test data, or visualization data')
    parser.add_argument('-dev', type=str, default=None, help='File path of validation data')
    parser.add_argument('-emb', type=str, default=None, help='File path of pre-trained embedding')
    parser.add_argument('-v', '--vocab', type=str, default=None, help='File path of vocabulary file')
    parser.add_argument('-tf_limit', type=int, default=0)

    parser.add_argument('-m', '--model_file', type=str, default=None, help='Path for saving model')
    parser.add_argument('-acc', '--log_acc', type=str, default=None, help='File path of acc log file')
    parser.add_argument('-loss', '--log_loss', type=str, default=None, help='File path of loss log file')

    parser.add_argument('-opt', type=str, default='adam')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-max_epoch', type=int, default=50)
    parser.add_argument('-use_early_stop', action='store_true', default=False)
    parser.add_argument('-use_lr_decay', action='store_true', default=False)

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

    # don't change
    parser.add_argument('-is_train', type=bool, default=True)
    parser.add_argument('-cat2id', type=dict, default=Label2ID)
    parser.add_argument('-class_num', type=int, default=len(Label2ID))
    return parser

def get_train_args():
    parser = add_common_args('train')
    return parser.parse_args()

def get_eval_args():
    parser = add_common_args('evaluation')
    parser.add_argument('-test', type=str, default=None, help='File path of test data')
    parser.add_argument('-run_n', type=int, default=1)
    return parser.parse_args()

def get_cv_args():
    parser = add_common_args('cross validation')
    parser.add_argument('-test', type=str, default=None, help='File path of test data')
    parser.add_argument('-kfold', type=int, default=3)
    return parser.parse_args()
