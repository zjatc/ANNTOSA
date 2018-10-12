# -*- coding: utf-8 -*-

import tensorflow as tf


def get_sequence_actual_length(inputs):
    """
    Calculate actual length of each sequence in a batch
    index of padded token is 0, 'post' padding method should be used
    idx_sequence shape = [batch_size, max_seq_length]
    actual_length shape = [batch_size]
    """
    used_pos = tf.sign(tf.abs(inputs))
    actual_length = tf.reduce_sum(used_pos, axis=1)
    actual_length = tf.cast(actual_length, tf.int32)
    return actual_length


def extract_vectors(inputs):
    """
    extract vectors from inputs in a batch based on to_be_extracted index
    to_be_extracted shape = [batch_size, max_sequence_length], which 1 refers to be extracted at each position
    remained_matrix shape = [batch_size, max_sequence_length, inputs_dim], which remove extracted vectors in inputs and padded 0-vectors at end
    extracted_count = scalar.int32, refers to how many vectors have be extracted
    if method is 'mean', then
    extracted_vector shape = [batch_size, 1, inputs_dim], which average all vectors extracted
    if method is 'padding', then
    extracted_vector shape = [batch_size, max_sequence_length, inputs_dim], which padded 0-vectors after extracted vectors
    """
    to_be_extracted, inputs, method = inputs
    (extracted_vec, remained_vec) = tf.map_fn(lambda ins: splice_matrix(ins[0], ins[1], method),
                                              (to_be_extracted, inputs), parallel_iterations=200,
                                              dtype=(tf.float32, tf.float32))
    extracted_count = tf.to_int32(tf.reduce_sum(to_be_extracted, -1))
    return extracted_vec, remained_vec, extracted_count


def splice_matrix(to_be_extracted, inputs, extracted_method):
    # input_dim = inputs.get_shape()[-1]
    # seq_length = inputs.get_shape()[-2]

    indicate_i = tf.expand_dims(to_be_extracted, 1)
    selected = tf.multiply(inputs, indicate_i)
    extracted_idx = tf.where(tf.not_equal(tf.reduce_sum(tf.abs(selected), -1), 0))
    extracted_vector_only = tf.gather_nd(selected, extracted_idx)

    if extracted_method == 'mean':
        extracted_vector_i = tf.reduce_mean(extracted_vector_only, -2, keepdims=True)
    elif extracted_method == 'max':
        extracted_vector_i = tf.reduce_max(extracted_vector_only, -2, keepdims=True)
    elif extracted_method == 'padding':
        extracted_zeros_idx = tf.where(tf.equal(tf.reduce_sum(tf.abs(selected), -1), 0))
        extracted_zeros_i = tf.gather_nd(selected, extracted_zeros_idx)
        extracted_vector_i = tf.concat([extracted_vector_only, extracted_zeros_i], 0)
    else:
        raise NotImplementedError('No implementation for extraction method : ' + extracted_method)

    remained_indicate_i = 1 - indicate_i
    remained_selected = tf.multiply(inputs, remained_indicate_i)
    remained_idx = tf.where(tf.not_equal(tf.reduce_sum(tf.abs(remained_selected), -1), 0))
    remained_vector_only = tf.gather_nd(remained_selected, remained_idx)
    remained_zeros_idx = tf.where(tf.equal(tf.reduce_sum(tf.abs(remained_selected), -1), 0))
    remained_zeros_i = tf.gather_nd(remained_selected, remained_zeros_idx)
    remained_vectors_i = tf.concat([remained_vector_only, remained_zeros_i], 0)

    return extracted_vector_i, remained_vectors_i
