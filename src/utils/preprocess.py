# -*- coding: utf-8 -*-

"""
python 3.5
"""
import csv
import os
import math
import tensorflow.keras as kr
import numpy as np
from collections import Counter
from hanziconv import HanziConv


TARGET = '$TARGET$'
PAD = '$PAD$'
UNKNOWN = '$UNKNOWN$'


def load_dataset(path, is_english=True, has_label=True, use_target='word', use_first_target=False):
    """
    loading csv file
    data: list of dict in format: 'target, text, indices, label'
    dataset: list of dict in format: 'ori_text, text_words, tar_words, tar_idx, label'
    """
    data = read_csv_file(path, has_label)

    if is_english:
        # ori_text, text_words, tar_words, tar_idx, label = dataset
        dataset = list(map(lambda line: process_english_data(line, use_target, use_first_target), data))
    else:
        dataset = None
        # dataset = list(map(lambda line: process_chinese_data(line, use_target, use_first_target), data))

    # ori_text, tokenized_text, target_words, target_idx, label = list(zip(*processed_data))
    return dataset


def build_vocab(vocab_file, data=None, embedding=None, emb_dim=200, is_train=True, tf_limit=0):
    """
    read_prebuilt vocabulary from vocab_file,
    if vocab_file doesn't exist, then build vacabulary from tokenized_train_data first.
    if pre_trainEMB is not None, then use pre_trainEMB to initialize the embedding matrix
    """
    # build vocabulary if vocab_file doesn't exist
    if not os.path.exists(vocab_file):
        all_tokens = [token for instance in data for token in instance]
        if tf_limit > 0:
            token_counter = Counter(all_tokens)
            valid_tokens = filter(lambda c: c[1] >= tf_limit, token_counter.items())
            vocab = list(list(zip(*list(valid_tokens)))[0])
        else:
            vocab = list(set(all_tokens))
        vocab = list(filter(lambda t: t.strip() != '', vocab))
        vocab = [PAD, UNKNOWN] + vocab
        #os.makedirs(os.path.dirname(vocab_file))
        with open(vocab_file, mode='w', encoding='utf-8') as vf:
            vf.write('\n'.join(vocab) + '\n')
        vocab_size = len(vocab)
        print('Built vocabulary of size %d. and stored in path %s' % (vocab_size, vocab_file))
    else:
        # read vocabulary from vocab_file
        with open(vocab_file, 'r', encoding='utf-8') as vf:
            vocab = vf.read().strip().split('\n')
        vocab_size = len(vocab)
        print('Loaded %d tokens from vocabulary file %s' % (vocab_size, vocab_file))
    token2id = dict(zip(vocab, range(vocab_size)))
    if is_train:
        emb_matrix, emb_dim = build_embedding(vocab_size, token2id, embedding, emb_dim)
        return vocab_size, token2id, emb_matrix, emb_dim
    else:
        return vocab_size, token2id


def build_dataset(dataset, token2id, cat2id, max_text_len, max_tar_len):
    """
    encode dataset: text_words, tar_idx, label, tar_words
    """
    if dataset[0]['label'] is None:
        encoded_label = [None]*len(dataset)
    else:
        encoded_label = encode_label([ins['label'] for ins in dataset], cat2id)
    encoded_text = encode_text([ins['text_words'] for ins in dataset], token2id, max_text_len)
    encoded_idx = encode_target_idx([ins['tar_idx'] for ins in dataset], max_text_len)
    encoded_target = encode_target([ins['tar_words'] for ins in dataset], token2id, max_tar_len)

    encoded = list(zip(encoded_text, encoded_target, encoded_idx, encoded_label))
    return encoded


def batch_generater(dataset, batch_size=1000):
    """
    to generate batched data using given batch_size
    default shuffle before generate batch data
    """
    dataset_size = len(dataset)
    num_batch = math.ceil(dataset_size / batch_size)
    text, target, tar_idx, label = map(lambda filed: np.array(filed), zip(*dataset))
    for batch in range(num_batch):
        s_idx = batch * batch_size
        e_idx = min((batch + 1) * batch_size, dataset_size)
        yield text[s_idx:e_idx], target[s_idx:e_idx], tar_idx[s_idx:e_idx], label[s_idx:e_idx]


def pad_dataset(dataset, batch_size):
    """
    pad dataset to guarantee each batch fulfil batch size
    """
    _size = len(dataset)
    padded_num = batch_size - (_size % batch_size)
    new_dataset = [data for data in dataset]
    new_dataset.extend(dataset[:padded_num])
    return new_dataset


def encode_label(labels, category2id):
    """
    encode categories with one-hot encoding
    """
    if labels is None:
        return None
    label_ids = [category2id[y] for y in labels]
    encoded_label = kr.utils.to_categorical(label_ids, num_classes=len(category2id))
    return encoded_label


def encode_text(tokenized_text, token2id, max_seq_length):
    """
    encode text with corresponding id and padding or truncating to max_seq_length
    if input length less than predefined sequence length, than padding with 0 in sequence tail
    if input length greater than predefined sequence length, than truncating to max_seq_length from sequence tail
    """
    token_ids = list(map(lambda ins: [token2id[token] if token in token2id else token2id[UNKNOWN] for token in ins], tokenized_text))
    encoded_text = kr.preprocessing.sequence.pad_sequences(token_ids, max_seq_length, padding='post', truncating='post')
    return encoded_text


def encode_target(tar_text, token2id, max_tar_length):
    """
    encode target text with corresponding id and padding or truncating to max_tar_length
    if target length less than predefined target length, than padding with 0 in sequence tail
    if target length greater than predefined target length, than truncating to max_tar_length from sequence tail
    """
    token_ids = list(map(lambda ins: [token2id[token] if token in token2id else token2id[UNKNOWN] for token in ins], tar_text))
    encoded_tar = kr.preprocessing.sequence.pad_sequences(token_ids, max_tar_length, padding='post', truncating='post')
    return encoded_tar


def encode_target_idx(target_idx, max_seq_length):
    """
    paded target_idx to max_seq_length
    1 if is a target word, else 0
    """
    encoded_target_idx = kr.preprocessing.sequence.pad_sequences(target_idx, max_seq_length, padding='post', truncating='post')
    return encoded_target_idx


def build_embedding(vocab_size, token2id, embedding, emb_dim):
    """
    build embedding matrix,
    """
    if embedding is None:
        emb_matrix = np.zeros([vocab_size, emb_dim], dtype='float32')
    else:
        with open(embedding, 'r', encoding='utf-8') as embf:
            lines = embf.readlines()
            emb_dim = len((lines[0].strip().split())[1:])
            emb_matrix = np.zeros([vocab_size, emb_dim], dtype='float32')
            found_counter = 0
            for line in lines:
                line = line.strip().split()
                token = line[0]
                if token in token2id:
                    try:
                        emb = [float(x) for x in line[1:]]
                    except ValueError:
                        continue
                    emb_matrix[token2id[token]] = np.asarray(emb)
                    found_counter += 1
            print('%d out of %d tokens has pre-trained embeddings' % (found_counter, vocab_size))
    for i in range(len(emb_matrix)):
        if i and np.count_nonzero(emb_matrix[i]) == 0:
            emb_matrix[i] = np.random.uniform(-0.25, 0.25, emb_dim)
    return emb_matrix, emb_dim


def read_csv_file(path, has_label=True):
    """
    read csv file,
    format "target, text, indices, label"
    """
    assert path.endswith(".csv")
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as csvf:
        reader = csv.DictReader(csvf, delimiter=',', quotechar='"')
        for line in reader:
            tar_idx = [idx.split('_') for idx in line['indices'].split('#')]
            if not tar_idx:
                continue
            text = line['text'].strip()
            if not text:
                continue
            tar_word = line['target'].strip()
            if not tar_word:
                continue
            if has_label:
                label = line['label']
            else:
                label = None
            data.append({'text': text, 'target': tar_word, 'indices': tar_idx, 'label': label})
    print('total loaded {} instances'.format(len(data)))
    return data


def process_english_data(line, use_target, use_first_target):
    """
    process english data, encode text, target_idx and label
    if use_target=='token', then replace all target words with special token '$TARGET$'
    if use_target=='word' or None, then use target original words
    """
    ori_text = line['text'].strip().lower()
    text_words = ori_text.split()
    label = line['label']
    tar_idx = sorted(int(idx[0]) for idx in line['indices'])

    target_words = line['target'].strip().lower().split()
    tar_len = len(target_words)
    new_target_words = target_words if use_target == 'word' else [TARGET]
    new_tar_len = tar_len if use_target == 'word' else 1

    tar_indexer = []
    tokenized_text = []
    found_target = False
    for i, token in enumerate(text_words):
        token = token.strip()
        if i in tar_idx:
            if not use_first_target:
                tokenized_text.extend(new_target_words)
                tar_indexer.extend([1]*new_tar_len)
            elif not found_target:
                tokenized_text.extend(new_target_words)
                tar_indexer.extend([1]*new_tar_len)
                found_target = True
            else:
                tokenized_text.extend(target_words)
                tar_indexer.extend([0]*tar_len)
        elif token:
            tokenized_text.append(token)
            tar_indexer.append(0)
    record = {'ori_text': ori_text, 'text_words': tokenized_text, 'tar_words': target_words, 'tar_idx': tar_indexer, 'label': label}
    return record


def process_chinese_data(line, use_target, use_first_target):
    text = line['text']
    target = line['target']
    tar_idx = line['indices']
    label = line['label']
    words_text = text.split()

    tar_idx_list = []
    tokenized_text = []
    found_target = False
    words_text = [HanziConv.toSimplified(word).lower() for word in words_text]
    if use_target == 'token':
        for idx in tar_idx:
            words_text = [TARGET if str(i) in idx else word for i, word in enumerate(words_text)]
        tokenized_text.extend(words_text)
        tar_idx_list = [1 if word == TARGET else 0 for word in tokenized_text]
    else:
        norm_target = [HanziConv.toSimplified(target).lower()]
        last_tar_end_idx = 0
        for idx in tar_idx:
            tar_start_idx = int(idx[0])
            if tar_start_idx != 0:
                norm_non_target_words = words_text[last_tar_end_idx:tar_start_idx]
                tokenized_text.extend(norm_non_target_words)
                tar_idx_list.extend([0]*len(norm_non_target_words))
            tokenized_text.extend(norm_target)
            if use_first_target and found_target:
                tar_idx_list.extend([0]*len(norm_target))
            else:
                tar_idx_list.extend([1]*len(norm_target))
                found_target = True
            last_tar_end_idx = tar_start_idx + 1

        if last_tar_end_idx < len(words_text)-1:
            norm_non_target_words = words_text[last_tar_end_idx:]
            tokenized_text.extend(norm_non_target_words)
            tar_idx_list.extend([0]*len(norm_non_target_words))

    return tokenized_text, target, tar_idx_list, label
