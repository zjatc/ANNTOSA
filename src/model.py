# -*- coding: utf-8 -*-

#from tensorflow import keras as kr
from utils.tensor_utils import get_sequence_actual_length
from tensor_util_layer import *
from attention import *


class TBSAModel(object):
    def __init__(self, args):
        self.is_train = args.is_train
        self.lr = args.lr
        self.opt = args.opt
        self.current_model = args.al
        self.class_num = args.class_num
        self.text_len = args.text_seq_len
        self.tar_len = args.tar_seq_len
        self.vocab_size = args.vocab_size
        self.emb_dim = args.emb_dim
        if self.is_train:
            self.embeddings = args.embeddings
        self.emb_dp = args.emb_dp
        self.out_dp = args.out_dp
        # lstm params
        self.text_rnn_dim = args.lstm_dim
        # structured self attention params
        self.struc_att_dim = args.struc_dim
        self.r = args.struc_r
        self.penal = args.struc_penal
        # global attention params
        self.align_model = args.align_model
        self.tensor_k = args.tensor_k
        self.t2t_dim = args.t2t_dim

        # self.uniform = kr.initializers.RandomUniform(minval=-0.01, maxval=0.01)
        self.build_common_layers()

        if self.current_model == 'seg':
            self.build_seg_model()
        elif self.current_model == 't2t':
            self.build_t2t_model()
        # elif self.current_model == 'seg_con_tar':
        #     self.build_seg_model_with_tar_concat()
        elif self.current_model == 'seg_tar':
            self.build_structured_target()
        elif self.current_model == 'test':
            self.test_1()
        print('built %s model using alignment model %s' % (self.current_model, self.align_model))
        if self.opt == 'adam':
            optimizer = kr.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
        elif self.opt == 'rmsp':    
            optimizer = kr.optimizers.RMSprop(lr=self.lr, rho=0.9, epsilon=1e-8, decay=0.0)
        else:
            raise NotImplementedError('No implementation for optimizer: ' + self.opt)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def build_common_layers(self):
        # build layers
        # encoder
        if self.is_train:
            self._get_embedding = kr.layers.Embedding(self.vocab_size, self.emb_dim, input_length=self.text_len, trainable=True, mask_zero=False, name='emb_layer', weights=[self.embeddings])
        else:
            self._get_embedding = kr.layers.Embedding(self.vocab_size, self.emb_dim, input_length=self.text_len, trainable=True, mask_zero=False, name='emb_layer')
        self._dropout_embedding = kr.layers.Dropout(self.emb_dp)
        self._rnn_encode = kr.layers.Bidirectional(kr.layers.LSTM(self.text_rnn_dim, return_sequences=True))
        self.encode_dim = 2 * self.text_rnn_dim
        # functional layers
        self._get_rlen = kr.layers.Lambda(get_sequence_actual_length)
        self._split_context_tar_avg = ExtractVector('mean')
        self._split_context_tar_pad = ExtractVector('padding')
        # output
        self._flatten = kr.layers.Flatten()
        self._dropout_output = kr.layers.Dropout(self.out_dp)
        self._classify = kr.layers.Dense(self.class_num, activation='softmax')

    def build_t2t_model(self):
        print("Build t2t model, which means bilstm + global attention")

        text_seq = kr.Input(shape=(self.text_len,), dtype='int32', name='text_seq_input')
        tar_idx = kr.Input(shape=(self.text_len,), dtype='float32', name='tar_idx_input')
        # model-based layers
        self.t2t = GlobalAttention(self.encode_dim, self.align_model, hidden_dim=self.t2t_dim,
                                   input_length=self.text_len, tensor_k=self.tensor_k)
        # forward process
        x_text_rlen = self._get_rlen(text_seq)
        x = self._get_embedding(text_seq)
        x = self._dropout_embedding(x)
        x = self._rnn_encode(x)
        x_tar_avg, x_context, x_tar_rlen = self._split_context_tar_avg([tar_idx, x])
        x_context_rlen = kr.layers.subtract([x_text_rlen, x_tar_rlen])
        attended_rep, attention = self.t2t([x_tar_avg, x_context, x_context_rlen])
        attended_rep = self._flatten(attended_rep)
        attended_rep = self._dropout_output(attended_rep)
        pred = self._classify(attended_rep)
        # defined model
        self.model = kr.Model(inputs=[text_seq, tar_idx], outputs=pred)

    def build_seg_model(self):
        print("Build segment attention model, which means bilstm + structured self attention + global attention")

        text_seq = kr.Input(shape=(self.text_len,), dtype='int32', name='text_seq_input')
        tar_idx = kr.Input(shape=(self.text_len,), dtype='float32', name='tar_idx_input')
        '''model-based layers'''
        self.context_att = StructuredSelfAttention(self.encode_dim, input_length=self.text_len, hidden_dim=self.struc_att_dim,
                                                   r=self.r, penal_coefficient=self.penal, output_method='matrix')
        # self.ffd_context_att = PositionwiseFeedForward(self.encode_dim, 16)
        self.tile_tar = TileWrapper([1, self.r, 1])
        self.fuse = kr.layers.Dense(self.encode_dim, activation='relu')
        self.tile_r = TileWrapper(scalar=self.r, axis=0)
        self.t2t = GlobalAttention(self.encode_dim, self.align_model, input_length=self.r, hidden_dim=self.t2t_dim,
                                   tensor_k=self.tensor_k)
        # self.ln = LayerNormalization()
        # self.ffd_t2t = PositionwiseFeedForward(self.encode_dim, 2*self.encode_dim)
        '''forward process'''
        x_text_rlen = self._get_rlen(text_seq)
        x = self._get_embedding(text_seq)
        x = self._dropout_embedding(x)
        x = self._rnn_encode(x)
        x_tar_avg, x_context, x_tar_rlen = self._split_context_tar_avg([tar_idx, x])
        x_context_rlen = kr.layers.subtract([x_text_rlen, x_tar_rlen])

        x_context, context_mat_attention = self.context_att([x_context, x_context_rlen])
        # x_context = self.ffd_context_att(x_context)
        '''fusion target vector with each sentence representation'''
        x_tar_tiled = self.tile_tar(x_tar_avg)
        x_context = kr.layers.concatenate([x_context, x_tar_tiled], -1)
        x_context = self.fuse(x_context)
        '''build t2t self attention layer to get context vector'''
        x_context_rlen = self.tile_r(x_context)
        attended_rep, attention = self.t2t([x_tar_avg, x_context, x_context_rlen])
        # attended_rep = self.ffd_t2t(attended_rep)
        # attended_rep = self.ln(attended_rep)
        attended_rep = self._flatten(attended_rep)
        attended_rep = self._dropout_output(attended_rep)
        pred = self._classify(attended_rep)
        '''defined model input and output'''
        self.model = kr.Model(inputs=[text_seq, tar_idx], outputs=pred)

    """
    def build_seg_model_with_tar_concat(self):
        print("Build segment attention concat target model, which means bilstm + structured self attention + global attention + target concat")

        text_seq = kr.Input(shape=(self.text_len,), dtype='int32', name='text_seq_input')
        tar_idx = kr.Input(shape=(self.text_len,), dtype='float32', name='tar_idx_input')
        # model-based layers
        self.context_att = StructuredSelfAttention(self.encode_dim, input_length=self.text_len, hidden_dim=self.struc_att_dim,
                                                   r=self.r, penal_coefficient=self.penal, output_method='matrix')
        self.tile_tar = TileWrapper([1, self.r, 1])
        self.fuse = kr.layers.Dense(self.encode_dim, activation='relu')
        self.tile_r = TileWrapper(scalar=self.r, axis=0)
        self.t2t = GlobalAttention(self.encode_dim, self.align_model, input_length=self.r, hidden_dim=self.t2t_dim,
                                   tensor_k=self.tensor_k)
        self.fuse_2 = kr.layers.Dense(64, activation='relu')
        # forward process
        x_text_rlen = self._get_rlen(text_seq)
        x = self._get_embedding(text_seq)
        x = self._dropout_embedding(x)
        x = self._rnn_encode(x)
        x_tar_avg, x_context, x_tar_rlen = self._split_context_tar_avg([tar_idx, x])
        x_context_rlen = kr.layers.subtract([x_text_rlen, x_tar_rlen])
        x_context, context_mat_attention = self.context_att([x_context, x_context_rlen])
        # fusion target vector with each sentence representation
        x_tar_tiled = self.tile_tar(x_tar_avg)
        x_context = kr.layers.concatenate([x_context, x_tar_tiled], -1)
        x_context = self.fuse(x_context)
        # build t2t self attention layer to get context vector
        x_context_rlen = self.tile_r(x_context)
        attended_rep, attention = self.t2t([x_tar_avg, x_context, x_context_rlen])
        # fusion target vector with final context representation
        attended_rep = kr.layers.concatenate([attended_rep, x_tar_avg])
        attended_rep = self._flatten(attended_rep)
        attended_rep = self.fuse_2(attended_rep)
        attended_rep = self._dropout_output(attended_rep)
        pred = self._classify(attended_rep)
        # defined model input and output
        self.model = kr.Model(inputs=[text_seq, tar_idx], outputs=pred)

    def build_seg_model_with_tar_rnn(self):
        pass
        print("Build segment attention model with a separated bilstm for target sequence, "
              "which means context bilstm + target bilstm + structured self attention + global attention")
        # define model-based layers
        self.tar_bilstm_layer = kr.layers.Bidirectional(kr.layers.LSTM(self.text_rnn_dim, return_sequences=False))
        self.tar_vec_reshape_layer = kr.layers.Reshape((1, 2 * self.text_rnn_dim))
        self.struc_att_layer = StructuredSelfAttention(2 * self.text_rnn_dim, input_length=self.text_len, hidden_dim=self.struc_att_dim,
                                                       r=self.r, penal_coefficient=self.penal, output_method='matrix')
        self.extractor_layer = ExtractVector('mean')
        self.seq_len_layer = kr.layers.Lambda(get_sequence_actual_length)
        self.fusion_layer = kr.layers.Dense(2 * self.text_rnn_dim, activation='tanh')
        self.tar_vec_tile_layer = TileWrapper([1, self.r, 1])
        self.struc_rep_len_tile_layer = TileWrapper(scalar=self.r, axis=0)
        self.t2t_layer = GlobalAttention(2 * self.text_rnn_dim, self.align_model, input_length=self.r, hidden_dim=self.t2t_dim,
                                         tensor_k=self.tensor_k)
        # define placeholder
        text_seq = kr.Input(shape=(self.text_len,), dtype='int32', name='text_seq_input')
        tar_seq = kr.Input(shape=(self.tar_len,), dtype='int32', name='tar_seq_input')
        tar_idx = kr.Input(shape=(self.text_len,), dtype='float32', name='tar_idx_input')
        # feedforward process
        text_emb = self.emb_layer(text_seq)
        text_emb = self.emb_dropout_layer(text_emb)
        text_bilstm = self.text_bilstm_layer(text_emb)
        tar_emb = self.emb_layer(tar_seq)
        tar_bilstm = self.tar_bilstm_layer(tar_emb)
        tar_vec = self.tar_vec_reshape_layer(tar_bilstm)
        text_rlen = self.seq_len_layer(text_seq)
        _, nontar_matrix, tar_rlen = self.extractor_layer([tar_idx, text_bilstm])
        nontar_rlen = kr.layers.subtract([text_rlen, tar_rlen])
        matrix_context, matrix_attention = self.struc_att_layer([nontar_matrix, nontar_rlen])
        # fusion target vector with each sentence representation
        tile_target_vector = self.tar_vec_tile_layer(tar_vec)
        context_concat_tar = kr.layers.concatenate([matrix_context, tile_target_vector], -1)
        fused_context = self.fusion_layer(context_concat_tar)

        # build t2t self attention layer to get context vector
        fused_context_len = self.struc_rep_len_tile_layer(fused_context)
        attended_text, attention = self.t2t_layer([tar_vec, fused_context, fused_context_len])
        attended_text = self.output_dropout_layer(attended_text)
        flatten_output = self.flat_layer(attended_text)
        pred = self.output_layer(flatten_output)
        # defined model input and output
        self.model = kr.Model(inputs=[text_seq, tar_seq, tar_idx], outputs=pred)
    """

    def build_structured_target(self):
        print("Build model with bilstm(seq) + struc_self_att(non-tar_seq) + struc_self_att(tar_seq) + global_attention")

        text_seq = kr.Input(shape=(self.text_len,), dtype='int32', name='text_seq_input')
        tar_idx = kr.Input(shape=(self.text_len,), dtype='float32', name='tar_idx_input')
        '''model-based layers'''
        tar_structured_rep_r = 3
        ffd_dim = 8
        self.context_att = StructuredSelfAttention(self.encode_dim, input_length=self.text_len, hidden_dim=self.struc_att_dim,
                                                   r=self.r, penal_coefficient=self.penal, output_method='matrix')
        self.ffd_context_att = PositionwiseFeedForward(self.encode_dim, ffd_dim)
        self.tar_att_1 = StructuredSelfAttention(self.encode_dim, input_length=self.text_len, hidden_dim=self.struc_att_dim,
                                                 r=tar_structured_rep_r, penal_coefficient=0.0, output_method='matrix')
        self.ffd_tar_att_1 = PositionwiseFeedForward(self.encode_dim, ffd_dim)
        self.tile_tar_r = TileWrapper(scalar=3.0, axis=0)
        self.tar_att_2 = StructuredSelfAttention(self.encode_dim, input_length=tar_structured_rep_r, hidden_dim=64,
                                                 r=1, penal_coefficient=0.0, output_method='matrix', bias=True)
        self.ffd_tar_att_2 = PositionwiseFeedForward(self.encode_dim, ffd_dim)
        self.tile_tar = TileWrapper([1, self.r, 1])
        self.fuse = kr.layers.Dense(self.encode_dim, activation='relu')
        self.tile_r = TileWrapper(scalar=self.r, axis=0)
        self.t2t = GlobalAttention(self.encode_dim, self.align_model, input_length=self.r, hidden_dim=self.t2t_dim, tensor_k=self.tensor_k)
        '''forward process'''
        x_text_rlen = self._get_rlen(text_seq)
        x = self._get_embedding(text_seq)
        x = self._dropout_embedding(x)
        x = self._rnn_encode(x)
        x_tar_seq, x_context, x_tar_rlen = self._split_context_tar_pad([tar_idx, x])
        x_context_rlen = kr.layers.subtract([x_text_rlen, x_tar_rlen])
        x_context, context_mat_attention = self.context_att([x_context, x_context_rlen])
        x_context = self.ffd_context_att(x_context)
        '''get proper target representation via 2-layers-self-attention'''
        x_tar_seq, tar_mat_attention_1 = self.tar_att_1([x_tar_seq, x_tar_rlen])
        x_tar_seq = self.ffd_tar_att_1(x_tar_seq)
        x_tar_rlen = self.tile_tar_r(x_tar_seq)
        x_tar, tar_mat_attention_2 = self.tar_att_2([x_tar_seq, x_tar_rlen])
        x_tar = self.ffd_tar_att_2(x_tar)
        '''fusion target vector with each sentence representation'''
        x_tar_tiled = self.tile_tar(x_tar)
        x_context = kr.layers.concatenate([x_context, x_tar_tiled], -1)
        x_context = self.fuse(x_context)
        '''build t2t self attention layer to get context vector'''
        x_context_rlen = self.tile_r(x_context)
        attended_rep, _ = self.t2t([x_tar, x_context, x_context_rlen])
        attended_rep = self._flatten(attended_rep)
        attended_rep = self._dropout_output(attended_rep)
        pred = self._classify(attended_rep)
        '''defined model input and output'''
        self.model = kr.Model(inputs=[text_seq, tar_idx], outputs=pred)

    def test_1(self):
        print("Build model with bilstm(seq) + struc_self_att(tar_seq) + struc_t2t_att(non-tar_seq) + select_one_tar_rep_t2t")

        ffd_dim = 8

        text_seq = kr.Input(shape=(self.text_len,), dtype='int32', name='text_seq_input')
        tar_idx = kr.Input(shape=(self.text_len,), dtype='float32', name='tar_idx_input')
        '''model-based layers'''
        self.tar_att_1 = StructuredSelfAttention(self.encode_dim, input_length=self.text_len, hidden_dim=self.struc_att_dim,
                                                 r=self.r, penal_coefficient=self.penal, output_method='matrix')
        self.ffd_tar_att_1 = PositionwiseFeedForward(self.encode_dim, ffd_dim)
        self.tile_tar_r = TileWrapper(scalar=self.r, axis=0)
        self.tar_att_2 = StructuredSelfAttention(self.encode_dim, input_length=self.r, hidden_dim=self.struc_att_dim,
                                                 r=1, penal_coefficient=0.0, output_method='matrix', bias=True)
        self.ffd_tar_att_2 = PositionwiseFeedForward(self.encode_dim, ffd_dim)
        self.context_att = StructuredGlobalAttention(self.encode_dim, input_length=self.text_len)
        self.ffd_context_att = PositionwiseFeedForward(self.encode_dim, ffd_dim)
        self.tile_tar = TileWrapper([1, self.r, 1])
        self.fuse = kr.layers.Dense(self.encode_dim, activation='relu')
        self.t2t = GlobalAttention(self.encode_dim, self.align_model, input_length=self.r,
                                   hidden_dim=self.t2t_dim, tensor_k=self.tensor_k)
        '''forward process'''
        x_text_rlen = self._get_rlen(text_seq)
        x = self._get_embedding(text_seq)
        x = self._dropout_embedding(x)
        x = self._rnn_encode(x)
        x_tar_seq, x_context, x_tar_rlen = self._split_context_tar_pad([tar_idx, x])
        x_context_rlen = kr.layers.subtract([x_text_rlen, x_tar_rlen])
        '''structured attention on target's hidden status and select best target rep'''
        x_tar_seq, tar_mat_attention_1 = self.tar_att_1([x_tar_seq, x_tar_rlen])
        x_tar_seq = self.ffd_tar_att_1(x_tar_seq)
        x_tar_r_rlen = self.tile_tar_r(x_tar_seq)
        x_tar, tar_mat_attention_2 = self.tar_att_2([x_tar_seq, x_tar_r_rlen])
        x_tar = self.ffd_tar_att_2(x_tar)

        '''structured global attention using structured rep of target and context's hidden status'''
        x_context, context_mat_attention = self.context_att([x_tar_seq, x_context, x_context_rlen])
        x_context = self.ffd_context_att(x_context)
        '''fusion target vector with each sentence representation'''
        x_tar_tiled = self.tile_tar(x_tar)
        x_context = kr.layers.concatenate([x_context, x_tar_tiled], -1)
        x_context = self.fuse(x_context)

        attended_rep, _ = self.t2t([x_tar, x_context, x_tar_r_rlen])
        attended_rep = self._flatten(attended_rep)
        attended_rep = self._dropout_output(attended_rep)
        pred = self._classify(attended_rep)
        '''defined model input and output'''
        self.model = kr.Model(inputs=[text_seq, tar_idx], outputs=pred)
