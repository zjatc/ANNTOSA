# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras as kr


"""
 ref. "A structured self-attentive sentence embedding"
 ref. align(H) = W_2*tanh(W_1*H^T)
 attention_matrix A = softmax(align(H))
 context_vector C = A*H
"""
class StructuredSelfAttention(kr.layers.Layer):
    def __init__(self, output_dim, input_length=None, hidden_dim=None, r=None, penal_coefficient=0., output_method='matrix', **kwargs):
        self.output_dim = output_dim
        self.L = input_length
        self.hidden_dim = hidden_dim
        self.r = r
        self.penal_factor = penal_coefficient
        self.output_method = output_method
        self.initializer = kr.initializers.RandomUniform(-0.01, 0.01)
        super(StructuredSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = H.shape = [B, L, H_dim]
        H_dim = self.output_dim

        # Create a trainable weight variable for this layer.
        self.W1 = self.add_weight(name='W1_{}'.format(self.name), shape=(self.hidden_dim, H_dim),
                                  initializer=self.initializer, trainable=True)
        self.W2 = self.add_weight(name='W2_{}'.format(self.name), shape=(self.r, self.hidden_dim),
                                  initializer=self.initializer, trainable=True)
        super(StructuredSelfAttention, self).build(input_shape)

    '''
    H.shape = [B, L, H_dim]
    _mask.shape = [B, 1, L]
    _align.shape = [B, r, L]
    attention.shape = [B, r, L]
    attended_matrix.shape = [B, r, H_dim]
    penal_term.shape = [B]
    output.shape = [B, 1, H_dim] / [B, r, H_dim] if output_method = matrix
    '''
    def call(self, inputs, *args, **kwargs):
        H, actual_len = inputs
        _mask = tf.expand_dims(tf.sequence_mask(actual_len, maxlen=self.L, dtype=tf.float32, name='struc_att_mask'), 1)

        # align(H) = W_2*tanh(W_1*H^T)
        _w1H = tf.transpose(kr.backend.dot(self.W1, tf.transpose(H, perm=[0, 2, 1])), perm=[1, 0, 2])  # [B, hidden_dim, L]
        _w1H_tanh = tf.tanh(tf.multiply(_w1H, _mask))  # [B, hidden_dim, L]
        _align = tf.transpose(kr.backend.dot(self.W2, _w1H_tanh), perm=[1, 0, 2])  # [B, r, L]
        attention = tf.nn.softmax(_align)
        # mask and renormalize
        attention = tf.multiply(attention, _mask)
        attention = attention / tf.expand_dims(tf.reduce_sum(attention, axis=-1), -1)
        attended_matrix = tf.matmul(attention, H)

        # penalization term to be added in loss function
        frobenius_norm = tf.norm(tf.matmul(attention, tf.transpose(attention, perm=[0, 2, 1])) - tf.eye(self.r), ord='fro', axis=[-2, -1])
        penal_term = tf.square(frobenius_norm)
        penal_term = self.penal_factor * penal_term

        # different pooling function can be used
        if self.output_method == 'matrix':
            output = attended_matrix
        elif self.output_method == 'max':
            output = tf.reduce_max(attended_matrix, [1], keepdims=True)
        elif self.output_method == 'mean':
            output = tf.reduce_mean(attended_matrix, [1], keepdims=True)
        else:
            raise RuntimeError('invalid structured self attention output method')

        return output, attention, penal_term

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape[0]).as_list()
        if self.output_method == 'matrix':
            shape[-2] = self.r
        else:
            shape[-1] = 1
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(StructuredSelfAttention, self).get_config()
        base_config['output_dim'] = self.output_dim
        base_config['output_method'] = self.output_method
        base_config['hidden_dim'] = self.hidden_dim
        base_config['r'] = self.r
        base_config['seq_len'] = self.L
        base_config['penal_factor'] = self.penal_factor

    @classmethod
    def from_config(cls, config):
        return cls(**config)
