# import something
import tensorflow as tf
import os
from collections import namedtuple
from Vocab import Vocab
from Batcher import Batcher
from model import SummarizationModel
from decode import BeamSearchDecoder
from tensorflow.python import debug as tf_debug
import time
import numpy as np

# define some flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_path','','Pa')

tf.app.flags.DEFINE_string('log_root','','root dir to log')
tf.app.flags.DEFINE_string('exp_name','','name of experiment')

tf.app.flags.DEFINE_string('mode','train','mode of running')
tf.app.flags.DEFINE_boolean('single_pass',False,'process data one by one or not')

tf.app.flags.DEFINE_string('vocab_path','','path of vocab file')
tf.app.flags.DEFINE_integer('vocab_size',5000,'Size of vocab')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):

    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay +(1-decay)*loss
    running_avg_loss = min(running_avg_loss,12)
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f'%decay
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    tf.logging.info('running_avg_loss:%f', running_avg_loss)
    return running_avg_loss


def restore_best_model():
    tf.logging.info('Restoring best model for training')

    sess = tf.Session(config=util.get_config())
    tf.logging.info('Initializing all variables...')
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver([v for v in tf.all_variables() if 'Adagrad' not in v.name])
    curr_ckpt = util.load_ckpt(saver, sess, 'eval')
    tf.logging.info('Restored %s.'%curr_ckpt)

    new_model_name = curr_ckpt.split('/')[-1].replace('bestmodel','model')
    new_fname = os.path.join(FLAGS.log_root, 'train',new_model_name)
    tf.logging.info('Saving model to %s' % new_fname)
    new_saver = tf.train.Saver()
    new_saver.save(sess, new_fname)
    print('Saved')
    exit()


def convert_to_coverage_model():
    tf.logging.info('converting non-coverage model to coverage model...')

    sess = tf.Session(config=util.get_config())
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver([v for v in tf.global_variables() if 'coverage' not in v.name and 'Adagrad' not in v.name])
    curr_ckpt = util.load_ckpt(saver, sess)

    new_fname = curr_ckpt + '_cov_init'

    new_saver = tf.train.Saver()
    new_saver.save(sess, new_fname)
    exit()


def setup_training(model, batcher):
    train_dir = os.path.join(FLAGS.log_root, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    model.build_graph()
    if FLAGS.convert_to_coverage_model:
        assert FLAGS.coverage, 'To convert you none-coverage model to a coverage model, ' \
                               'run with convert_to_coverage_model=True and coverage=True"'
        convert_to_coverage_model()
    if FLAGS.restore_best_model():
        restore_best_model()
    saver = tf.train.Saver(max_to_keep=3)

    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=60,
                             global_step=model.global_step)
    summary_writer = sv.summary_writer

    tf.logging.info('Preparing or waiting for session...')
    sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
    tf.logging.info('Created Session.')
    try:
        run_training(model, batcher, sess_context_manager,sv, summary_writer)
    except KeyboardInterrupt:
        tf.logging.info('Caught keyboard interrupt on worker. Stopping...')
        sv.stop()


def run_training(model, batcher, sess_context_manager, sv, summary_writer):
    tf.logging.info('starting run_training')
    with sess_context_manager as sess:
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        while True:
            batch = batcher.next_batch()

            tf.logging.info('running training step...')
            t0 = time.time()
            results = model.run_train_setp(sess, batch)
            t1 = time.time()
            tf.logging.info('Seconds for training setp:%.3f'%t1-t0)

            loss = results['loss']
            tf.logging.info('loss: %f'%loss)

            if not np.isfinite(loss):
                raise Exception('Loss is not finite. Stoping...')

            if FLAGS.coverage:
                coverage_loss = results['coverage_loss']
                tf.logging.info('coverage_loss: %f'%coverage_loss)

            summaries = results['summaries']
            train_step = results['global_step']

            summary_writer.add_summary(summaries, train_step)

            if train_step % 100 == 0:
                summary_writer.flush()


def run_eval(model, batcher, vocab):
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=3)
    sess = tf.Session(config=util.get_config())
    eval_dir = os.path.join(FLAGS.log_root,'eval')
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel')
    summary_writer = tf.summary.FileWriter(eval_dir)

    running_avg_loss = 0
    best_loss = None

    while True:
        _ = util.load_ckpt(saver, sess)
        batch = batcher.next_batch()

        t0 = time.time()
        results = model.run_eval_step(sess, batch)
        t1 = time.time()
        tf.logging.info('seconds for batch: %.2f'%(t1-t0))

        loss = results['loss']
        tf.logging.info('loss: %f'%loss)
        if FLAGS.coverage:
            coverage_loss = results['coverage_loss']
            tf.logging.info("coverage_loss: %f", coverage_loss)

        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)

        running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

        if best_loss is None or running_avg_loss < best_loss:
            tf.logging.info('Fount new best model with %.3f running_avg_loss. Saving to %s'%(running_avg_loss, bestmodel_save_path))
            saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_loss = running_avg_loss

        if train_step % 100 == 0:
            summary_writer.flush()


def main(argv_unused):
    if len(argv_unused) > 1:
        raise Exception("Problem with flags: %s"%argv_unused)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting...')

    # log root dir
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)

    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == 'train':
            os.mkdir(FLAGS.log_root)
        else:
            raise Exception('Logdir %s is not exists.' % FLAGS.log_root)

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)

    hparam_list = ['mode','lr','adagrad_init_acc','rand_unif_init_mag','trunc_norm_init_std','max_grad_norm',
                   'hidden_dim','emb_dim','batch_size','max_dec_steps','max_enc_steps','coverage','cov_loss_wt',
                   'pointer_gen']

    hps_dict = {}
    for key,val in FLAGS.__flags.items():
        if key in hparam_list:
            hps_dict[key] = val

    hps = namedtuple('HParams',hps_dict.keys())(**hps_dict)

    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    tf.set_random_seed(111)

    if hps.mode == 'train':
        tf.logging.info('Creating model..')
        model = SummarizationModel(hps, vocab)
        setup_training(model, batcher)
    elif hps.mode == 'eval':
        model = SummarizationModel(hps, vocab)
        run_eval(model, batcher, vocab)
    elif hps.mode == 'decode':
        decode_model_hps = hps
        decode_model_hps = hps._replace(max_dec_steps=1)
        model = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab)
        decoder.decode()
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")

# main
if __name__ == '__main__':
    tf.app.run()