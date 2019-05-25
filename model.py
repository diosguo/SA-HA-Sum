import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow.contrib as contrib
import time
import os


FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):

    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        hps=self._hps

        self._enc_batch = tf.placeholder(tf.int32,[hps.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size,None], name='enc_padding_mask')

        if FLAGS.pointer_gen:
            self._enc_batch_extend_vocadb = tf.placeholder(tf.int32, [hps.batch_size,None], name='enc_batch_extend_vocab')
            self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

        if hps.mode=='decode' and hps.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')

    def _add_emb_vis(self, embedding_var):
        train_dir = os.path.join(FLAGS.log_root,'train')
        vocab_metadata_path = os.path.join(train_dir,'vocab_metdata.tsv')
        self._vocab.write_metadata(vocab_metadata_path)
        summary_writer = tf.summary.FileWriter(train_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = vocab_metadata_path
        projector.visualize_embeddings(summary_writer, config)

    def _add_encoder(self, encoder_inputs, seq_len):
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs,
                                                                                dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
        return encoder_outputs, fw_st, bw_st

    def _reduce_states(self, fw_st, bw_st):
        hidden_dim = self._hps.hidden_dim
        with tf.variable_scope('reduce_final_st'):

            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim*2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim*2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c',[hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_c',[hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c)+bias_reduce_c)
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h)+bias_reduce_h)

            return contrib.rnn.LSTMStateTuple(new_c, new_h)

    def _add_decoder(self, decoder_inputs):
        hps = self._hps
        cell = contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

        prev_coverage = self.prev_coverage if hps.mode=='decode' and hps.coverage else None

        outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(decoder_inputs,
                                                                             self._dec_in_state,
                                                                             self._enc_states,
                                                                             self._enc_padding_mask,
                                                                             cell,
                                                                             initial_state_attention=(hps.mode=='decode'),
                                                                             pointer_gen=hps.pointer_gen,
                                                                             use_coverage=hps.coverage,
                                                                             prev_coverage=prev_coverage)

    def _add_seq2seq(self):
        hps = self._hps
        vsize = self._vocab.size()

        with tf.variable_scope('seq2seq'):
            self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding',[vsize,hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            if hps.mode == 'train':
                self._add_emb_vis(embedding)
            emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)
            emb_dec_inputs = [tf.nn.embedding_lookup(embedding,x) for x in tf.unstack(self._dec_batch, axis=1)]

        enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
        self._enc_states = enc_outputs

        self._dec_in_state = self._reduce_states(fw_st, bw_st)

        with tf.variable_scope('decoder'):
            decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(emb_dec_inputs)


    def build_graph(self):
        tf.logging.info('Building graph...')
        t0 = time.time()
        self._add_placeholders()
        with tf.device('/gpu:0'):
            self._add_seq2seq()
        self.global_step = tf.Variable(0,name='global_step',trainable=False)