import tensorflow as tf
import time

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

    def build_graph(self):
        tf.logging.info('Building graph...')
        t0 = time.time()
        self._add_placeholders()