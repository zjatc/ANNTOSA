# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras as kr


class GlobalAttention(kr.layers.Layer):
    """
    support alignment model: ['tensor', 'tb', 'concat', 'dot', 'bilinear']
    'tensor':
    ref. "Reasoning With Neural Tensor Networks for Knowledge Base Completion"
    ref. align(q, K) = v^T*tanh(q^T*W{1:k}*K + U*[q:K] + b)
    NOTE: W is a tensor with shape = [k, q_dims, K_dim]
    'tb':
    align(q, K) = q^T * tanh(W*K + b)

    'dot'/'bilinear'/'concat':
    ref. "Effective Approaches to Attention-based Neural Machine Translation"  # all
         "Attention-Based LSTM for Target-Dependent Sentiment Classification"  # dot, bilinear
         "Attention Modeling for Targeted Sentiment" # concat

    dot:      align(q, K) = q^T*K
              NOTE: dot alignment model, the target vector and source vector K_i must have same dimension
    bilinear: align(q, K) = q^T*W*K
    concat:   align(q, K) = v^T * tanh(W*[q:K] + b)

    attention_vector a = softmax(align(q, K))
    context_vector c = a*K
    """
    def __init__(self, output_dim, align_model, input_length=None, hidden_dim=None, tensor_k=None, **kwargs):
        self.supports_masking = True
        self.output_dim = output_dim
        self.L = input_length
        self.align_model = align_model
        self.hidden_dim = hidden_dim
        self.tensor_k = tensor_k
        self.initializer = kr.initializers.RandomUniform(-0.01, 0.01)
        super(GlobalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (qT.shape, KT.shape)
        # qT.shape = q^T = [B,1,q_dim]
        # KT.shape = K^T = [B,L,K_dim]
        q_dim = input_shape[0][-1].value
        K_dim = self.output_dim

        # Create a trainable weight variable for this layer.
        if self.align_model == 'tb':
            self.W = self.add_weight(name='tb_W_{}'.format(self.name), shape=(q_dim, K_dim),
                                     initializer=self.initializer, trainable=True)
            self.b = self.add_weight(name='tb_b_{}'.format(self.name), shape=self.L,
                                     initializer=kr.initializers.zeros, trainable=True)
        elif self.align_model == 'tensor':
            self.v_T = self.add_weight(name='tensor_v_T_{}'.format(self.name), shape=(1, self.tensor_k),
                                     initializer=self.initializer, trainable=True)
            self.W = self.add_weight(name='tensor_W_{}'.format(self.name), shape=(self.tensor_k, q_dim, K_dim),
                                     initializer=self.initializer, trainable=True)
            self.U = self.add_weight(name='tensor_U_{}'.format(self.name), shape=(self.tensor_k, q_dim+K_dim),
                                     initializer=self.initializer, trainable=True)
            self.b = self.add_weight(name='tensor_b_{}'.format(self.name), shape=self.L,
                                     initializer=kr.initializers.zeros, trainable=True)
        elif self.align_model == 'bilinear':
            self.W = self.add_weight(name='bilinear_W_{}'.format(self.name), shape=(q_dim, K_dim),
                                     initializer=self.initializer, trainable=True)
        elif self.align_model == 'concat':
            self.v_T = self.add_weight(name='concat_v_T_{}'.format(self.name), shape=(1, self.hidden_dim),
                                     initializer=self.initializer, trainable=True)
            self.W = self.add_weight(name='concat_W_{}'.format(self.name), shape=(self.hidden_dim, q_dim + K_dim),
                                     initializer=self.initializer, trainable=True)
            self.b = self.add_weight(name='concat_b_{}'.format(self.name), shape=self.L,
                                     initializer=kr.initializers.zeros, trainable=True)
        super(GlobalAttention, self).build(input_shape)

    '''
    q.shape = [B, 1, q_dim]
    K.shape = [B, L, K_dim]
    _mask.shape = [B, 1, L]
    _align.shape = [B, 1, L]
    attention.shape = [B, 1, L]
    attended_vector.shape = [B, 1, K_dim]
    '''
    def call(self, inputs, mask=None, *args, **kwargs):
        q, K, actual_len = inputs
        _mask = tf.expand_dims(tf.sequence_mask(actual_len, maxlen=self.L, dtype=tf.float32, name='t2t_mask'), 1)

        if self.align_model == 'tb':
            # align(q, K) = q^T * tanh(W*K + b)
            _wK_b = tf.transpose(kr.backend.dot(self.W, tf.transpose(K, perm=[0, 2, 1])), perm=[1, 0, 2]) + self.b  # [B, q_dim, L]
            _wK_b_tanh = tf.tanh(tf.multiply(_wK_b, _mask))  # [B, q_dim, L]
            _align = kr.backend.batch_dot(q, _wK_b_tanh)
            #_align = kr.backend.batch_dot(q, _wK_b_tanh, axes=[2, 1])
        elif self.align_model == 'tensor':
            # align(q, K) = v^T*tanh(q^T*W{1:k}*K + U*[q:K] + b)
            # tensor_bilinear = q^T*W{1:k}*K
            _qW = kr.backend.squeeze(kr.backend.dot(q, self.W), -3)
            _qWK = tf.matmul(_qW, K, transpose_b=True)  # [B, tensor_k , L]
            # _qWK = kr.backend.batch_dot(_qW, tf.transpose(K, perm=[0, 2, 1]))  # [B, tensor_k , L]
            # concat = U*[q:K]
            _concat_q_K = tf.concat([tf.tile(q, [1, self.L, 1]), K], 2)  # [B, L, q_dim + K_dim]
            _masked_concat_q_K = tf.multiply(tf.transpose(_concat_q_K, perm=[0, 2, 1]), _mask)  # [B, q_dim + K_dim, L]
            _UqK = tf.transpose(kr.backend.dot(self.U, _masked_concat_q_K), perm=[1, 0, 2])  # [B, tensor_k, L]
            # tensor_bilinear + concat + bias
            _combine = _qWK + _UqK + self.b
            _combine_tanh = tf.tanh(tf.multiply(_combine, _mask))  # [B, tensor_k, L]
            _align = tf.transpose(kr.backend.dot(self.v_T, _combine_tanh), perm=[1, 0, 2])
        elif self.align_model == 'dot':
            # align(q, K) = q^T*K
            _align = tf.matmul(q, tf.transpose(K, perm=[0, 2, 1]))
        elif self.align_model == 'bilinear':
            # align(q, K) = q^T*W*K
            _align = tf.matmul(tf.tensordot(q, self.W, axes=[[-1], [0]]), tf.transpose(K, perm=[0, 2, 1]))
        elif self.align_model == 'concat':
            # align(q, K) = v^T * tanh(W*[q:K] + b)
            _concat_q_K = tf.concat([tf.tile(q, [1, self.L, 1]), K], 2)  # [B, L, q_dim + K_dim]
            _masked_concat_q_K = tf.multiply(tf.transpose(_concat_q_K, perm=[0, 2, 1]), _mask)  # [B, q_dim + K_dim, L]
            _wqK_b = tf.transpose(kr.backend.dot(self.W, _masked_concat_q_K), perm=[1, 0, 2]) + self.b  # [B, hidden_dim, L]
            _wqK_b_tanh = tf.tanh(tf.multiply(_wqK_b, _mask))  # [B, hidden_dim, L]
            _align = tf.transpose(kr.backend.dot(self.v_T, _wqK_b_tanh), perm=[1, 0, 2])
        else:
            raise RuntimeError('invalid align model')
        attention = tf.nn.softmax(_align)
        # mask and renormalize
        attention = tf.multiply(attention, _mask)
        attention = attention / tf.expand_dims(tf.reduce_sum(attention, axis=-1), 1)
        attended_vector = tf.matmul(attention, K)

        return attended_vector, attention

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape[1]).as_list()
        shape[-2] = 1
        return tf.TensorShape(shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        base_config = super(GlobalAttention, self).get_config()
        base_config['output_dim'] = self.output_dim
        base_config['align_model'] = self.align_model
        base_config['seq_len'] = self.L
        if self.hidden_dim:
            base_config['hidden_dim'] = self.hidden_dim
        if self.tensor_k:
            base_config['tensor_k'] = self.tensor_k

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class StructuredSelfAttention(kr.layers.Layer):
    """
    ref. "A structured self-attentive sentence embedding"
    ref. align(H) = W_2*tanh(W_1*H^T)
    attention_matrix A = softmax(align(H))
    context_vector C = A*H
    """
    def __init__(self, output_dim, input_length=None, hidden_dim=None, r=None, penal_coefficient=0., output_method='matrix', bias=False, **kwargs):
        self.supports_masking = True
        self.output_dim = output_dim
        self.L = input_length
        self.hidden_dim = hidden_dim
        self.r = r
        self.penal_factor = penal_coefficient
        self.output_method = output_method
        self.bias = bias
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
        if self.bias:
            self.b = self.add_weight(name='b_{}'.format(self.name), shape=(self.L,),
                                     initializer=kr.initializers.zeros, trainable=True)
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
    def call(self, inputs, mask=None, *args, **kwargs):
        H, actual_len = inputs
        _mask = tf.expand_dims(tf.sequence_mask(actual_len, maxlen=self.L, dtype=tf.float32, name='struc_att_mask'), 1)

        # align(H) = W_2*tanh(W_1*H^T)
        _w1H = tf.transpose(kr.backend.dot(self.W1, tf.transpose(H, perm=[0, 2, 1])), perm=[1, 0, 2])  # [B, hidden_dim, L]
        if self.bias:
            _w1H += self.b
        _w1H_tanh = tf.tanh(tf.multiply(_w1H, _mask))  # [B, hidden_dim, L]
        _align = tf.transpose(kr.backend.dot(self.W2, _w1H_tanh), perm=[1, 0, 2])  # [B, r, L]
        attention = tf.nn.softmax(_align)
        # mask and renormalize
        attention = tf.multiply(attention, _mask)
        attention = attention / tf.expand_dims(tf.reduce_sum(attention, axis=-1), -1)
        attended_matrix = tf.matmul(attention, H)

        if self.penal_factor > 0.0:
            # penalization term to be added in loss function
            # frobenius_norm = tf.norm(tf.matmul(attention, tf.transpose(attention, perm=[0, 2, 1])) - tf.eye(self.r), ord='fro', axis=[-2, -1])
            # sqr_frobenius_norm = tf.square(frobenius_norm)
            batch_size = tf.cast(tf.shape(attention)[0], tf.float32)
            #sqr_frobenius_norm = kr.backend.sum(kr.backend.square(kr.backend.batch_dot(attention, kr.backend.permute_dimensions(attention, (0, 2, 1))) - tf.eye(self.r)))
            sqr_frobenius_norm = tf.reduce_sum(tf.square(tf.matmul(attention, tf.transpose(attention, perm=[0, 2, 1])) - tf.eye(self.r)))
            penal_term = self.penal_factor * sqr_frobenius_norm / batch_size
            self.add_loss(penal_term)

        # different pooling function can be used
        if self.output_method == 'matrix':
            output = attended_matrix
        elif self.output_method == 'max':
            output = tf.reduce_max(attended_matrix, [1], keepdims=True)
        elif self.output_method == 'mean':
            output = tf.reduce_mean(attended_matrix, [1], keepdims=True)
        else:
            raise RuntimeError('invalid structured self attention output method')

        return output, attention

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape[0]).as_list()
        if self.output_method == 'matrix':
            shape[-2] = self.r
        else:
            shape[-1] = 1
        return tf.TensorShape(shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        base_config = super(StructuredSelfAttention, self).get_config()
        base_config['output_dim'] = self.output_dim
        base_config['output_method'] = self.output_method
        base_config['hidden_dim'] = self.hidden_dim
        base_config['r'] = self.r
        base_config['seq_len'] = self.L
        base_config['penal_factor'] = self.penal_factor
        base_config['bias'] = self.bias
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class StructuredGlobalAttention(kr.layers.Layer):
    def __init__(self, output_dim, input_length=None, **kwargs):
        self.supports_masking = True
        self.output_dim = output_dim
        self.L = input_length
        self.initializer = kr.initializers.RandomUniform(-0.01, 0.01)
        super(StructuredGlobalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape[0] = Q.shape = [B, r, H_dim]
        # input_shape[1] = K.shape = [B, L, H_dim]
        k_dim = self.output_dim

        # Create a trainable weight variable for this layer.
        self.w = self.add_weight(name='W_{}'.format(self.name), shape=(k_dim, k_dim),
                                  regularizer=kr.regularizers.l2(0.01), initializer=self.initializer, trainable=True)
        super(StructuredGlobalAttention, self).build(input_shape)

    '''
    H.shape = [B, L, H_dim]
    _mask.shape = [B, 1, L]
    _align.shape = [B, r, L]
    attention.shape = [B, r, L]
    attended_matrix.shape = [B, r, H_dim]
    penal_term.shape = [B]
    output.shape = [B, 1, H_dim] / [B, r, H_dim] if output_method = matrix
    '''
    def call(self, inputs, mask=None, *args, **kwargs):
        q, k, actual_len = inputs
        _mask = tf.expand_dims(tf.sequence_mask(actual_len, maxlen=self.L, dtype=tf.float32, name='struc_global_mask'), 1)

        # align(Q, K) = Q*tanh(W*K^T)
        _wk = tf.transpose(kr.backend.dot(self.w, tf.transpose(k, perm=[0, 2, 1])), perm=[1, 0, 2])  # [B, K_dim, L]
        _wk_tanh = tf.tanh(tf.multiply(_wk, _mask))  # [B, K_dim, L]
        # _wk_tanh = tf.multiply(_wk, _mask)  # [B, K_dim, L]
        _align = tf.matmul(q, _wk_tanh)  # [B, r, L]
        attention = tf.nn.softmax(_align)
        # mask and renormalize
        attention = tf.multiply(attention, _mask)
        attention = attention / tf.expand_dims(tf.reduce_sum(attention, axis=-1), -1)
        attended_matrix = tf.matmul(attention, k)

        return attended_matrix, attention

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape[1]).as_list()
        return tf.TensorShape(shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        base_config = super(StructuredGlobalAttention, self).get_config()
        base_config['output_dim'] = self.output_dim
        base_config['seq_len'] = self.L
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
