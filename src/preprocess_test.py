# -*- coding: utf-8 -*-

import utils.preprocess as dp
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    # train_file = 'C:\\Users\\jasonzhang\\Desktop\\sentiment\\targetBased-sentiment\\dataset\\z\\format-process\\train.csv'
    # test_file = 'C:\\Users\\jasonzhang\\Desktop\\sentiment\\targetBased-sentiment\\dataset\\z\\format-process\\test.csv'
    # #vocab_file = 'C:\\Users\\jasonzhang\\Desktop\\sentiment\\targetBased-sentiment\\dataset\\t\\tnet\\vocab.txt'
    # #embedding_file = 'C:\\Users\\jasonzhang\\Desktop\\sentiment\\targetBased-sentiment\\dataset\\glove-emb\\glove.840B.300d.txt'
    # embedding_file = None
    # cat2id = {'-1': 0, '0': 1, '1': 2}
    #
    # train_dataset = dp.load_dataset(train_file, has_label=False)
    # test_dataset = dp.load_dataset(test_file, has_label=False)
    # full_dataset = train_dataset + test_dataset
    # print('train size: {}'.format(len(train_dataset)))
    # print('test size: {}'.format(len(test_dataset)))
    # print('total size: {}'.format(len(full_dataset)))
    #
    # max_len = max([len(ins['text_words']) for ins in full_dataset])
    # max_tar_len = max([len(ins['tar_words']) for ins in full_dataset])
    # print('max seq length: {}'.format(max_len))
    # print('max target length: {}'.format(max_tar_len))

    #vocab_size, token2id, emb_matrix, emb_dim = dp.build_vocab(vocab_file, data=[ins['text_words'] for ins in full_dataset], embedding=embedding_file)
    #print('vocabSize: {}'.format(vocab_size))
    #print('token2id len: {}'.format(len(token2id)))
    #print('emb_matrix len: {}'.format(len(emb_matrix)))
    #print('embDim: {}'.format(emb_dim))

    # list of tuple (encoded_text, encoded_target, encoded_idx, encoded_label)
    #train_encoded = dp.build_dataset(train_dataset, token2id, cat2id, max_len, max_tar_len)
    #print('encoded train size: {}'.format(len(train_encoded)))
    #print('encoded_text\n{}'.format(train_encoded[1][0]))
    #print('encoded_tar\n{}'.format(train_encoded[1][1]))
    #print('encoded_idx\n{}'.format(train_encoded[1][2]))
    #print('encoded_label\n{}'.format(train_encoded[1][3]))

    #batch_size = 128
    #pad_dataset = False
    #drop_last_batch = True

    # pad dataset
    # if pad_dataset:
    #     train_encoded = dp.pad_dataset(train_encoded, batch_size)
    #
    # np.random.seed(20180905)
    # np.random.shuffle(train_encoded)
    #
    # train_batches = dp.batch_generater(train_encoded, batch_size=batch_size)
    # bn = 0
    # for text_batch, target_batch, tar_idx_batch, label_batch in train_batches:
    #     if len(text_batch) != batch_size:
    #         if drop_last_batch:
    #             continue
    #         else:
    #             print('not fulfil a batch: {}'.format(len(text_batch)))
    #
    #     if len(text_batch) != len(target_batch):
    #         print('ERROR')
    #         break
    #     if len(text_batch) != len(tar_idx_batch):
    #         print('ERROR')
    #         break
    #     if len(text_batch) != len(label_batch):
    #         print('ERROR')
    #         break
    #     bn += 1
    # print('batch num: {}'.format(bn))
    # print('total ins num: {}'.format(((bn-1)*batch_size+len(text_batch))))

    '''
    test_features, test_label = dp.build_dataset(test_dataset, token2id, cat2id, max_len, max_tar_len)
    print('encoded test size: {}'.format(len(test_features)))
    print('encoded_text type\n{}'.format(test_features[0][0]))
    print('encoded_tar type\n{}'.format(test_features[0][1]))
    print('encoded_idx type\n{}'.format(test_features[0][2]))
    print('encoded_label type\n{}'.format(test_label[0]))
    '''

