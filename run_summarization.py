# import something
import tensorflow as tf
import os
from collections import namedtuple
from Vocab import Vocab

# define some flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_path','','Pa')
tf.app.flags.DEFINE_string('log_root','','root dir to log')
tf.app.flags.DEFINE_string('exp_name','','name of experiment')
tf.app.flags.DEFINE_string('mode','train','mode of running')
tf.app.flags.DEFINE_boolean('single_pass',False,'process data one by one or not')

tf.app.flags.DEFINE_string('vocab_path','','path of vocab file')
tf.app.flags.DEFINE_integer('vocab_size',5000,'Size of vocab')




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








# main
if __name__ == '__main__':
    tf.app.run()