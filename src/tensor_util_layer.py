# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras as kr
import numpy as np


class LayerNormalization(kr.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='ln_gamma', shape=input_shape[-1:],
                                     initializer=kr.initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='ln_beta', shape=input_shape[-1:],
                                    initializer=kr.initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        mean, var = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
        std = tf.sqrt(var)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'eps': self.eps}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PositionwiseFeedForward(kr.layers.Layer):
    """input_shape[-1] must be equal with output_dim"""
    def __init__(self, output_dim, hidden_dim, dropout=0.1, **kwargs):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.w1 = kr.layers.Conv1D(hidden_dim, 1, activation='relu')
        self.w2 = kr.layers.Conv1D(output_dim, 1)
        self.layer_norm = LayerNormalization()
        self.dropout_layer = kr.layers.Dropout(self.dropout)
        super(PositionwiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PositionwiseFeedForward, self).build(input_shape)

    def __call__(self, inputs, **kwargs):
        output = self.w1(inputs)
        output = self.w2(output)
        output = self.dropout_layer(output)
        output = kr.layers.add([output, inputs])
        return self.layer_norm(output)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout
        }
        base_config = super(PositionwiseFeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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


class ExtractVector(kr.layers.Layer):
    def __init__(self, method, **kwargs):
        self.supports_masking = True
        self.method = method
        super(ExtractVector, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ExtractVector, self).build(input_shape)

    def call(self, inputs, mask=None):
        extract_idx, to_be_extract = inputs

        def splice_matrix(idx, ori_tensor, extracted_method):
            indicate_i = tf.expand_dims(idx, 1)
            selected = tf.multiply(ori_tensor, indicate_i)
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
            remained_selected = tf.multiply(ori_tensor, remained_indicate_i)
            remained_idx = tf.where(tf.not_equal(tf.reduce_sum(tf.abs(remained_selected), -1), 0))
            remained_vector_only = tf.gather_nd(remained_selected, remained_idx)
            remained_zeros_idx = tf.where(tf.equal(tf.reduce_sum(tf.abs(remained_selected), -1), 0))
            remained_zeros_i = tf.gather_nd(remained_selected, remained_zeros_idx)
            remained_vectors_i = tf.concat([remained_vector_only, remained_zeros_i], 0)

            return extracted_vector_i, remained_vectors_i

        (extracted_vec, remained_vec) = tf.map_fn(lambda ins: splice_matrix(ins[0], ins[1], self.method),
                                                  (extract_idx, to_be_extract), parallel_iterations=200,
                                                  dtype=(tf.float32, tf.float32))
        extracted_count = tf.cast(tf.reduce_sum(extract_idx, -1), tf.int32)
        return extracted_vec, remained_vec, extracted_count

    def compute_output_shape(self, input_shape):
        ori_shape = input_shape[1]
        if self.method == 'padding':
            return [ori_shape, ori_shape, (ori_shape[0])]
        else:
            return [(ori_shape[0], 1, ori_shape[2]), ori_shape, (ori_shape[0])]

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'method': self.method,
            'supports_masking': self.supports_masking
        }
        base_config = super(ExtractVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TileWrapper(kr.layers.Layer):
    def __init__(self, multiples=None, scalar=None, axis=None, **kwargs):
        self.supports_masking = True
        self.multiples = multiples
        self.scalar = scalar
        self.axis = axis
        super(TileWrapper, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TileWrapper, self).build(input_shape)

    def call(self, inputs, mask=None):
        if self.multiples is not None:
            tiled = tf.tile(inputs, self.multiples)
        else:
            tiled = tf.tile([self.scalar], [tf.shape(inputs)[self.axis]])
        return tiled

    def compute_output_shape(self, input_shape):
        if self.multiples is not None:
            output_shape = tf.TensorShape(input_shape) * tf.constant(np.array(self.multiples))
        else:
            batch_size = tf.TensorShape(input_shape).as_list()[0]
            output_shape = [batch_size]
        return tf.TensorShape(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'multiples': self.multiples,
            'scalar': self.scalar,
            'axis': self.axis,
            'supports_masking': self.supports_masking
        }
        base_config = super(TileWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
