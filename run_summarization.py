# import something
import tensorflow as tf
import os

from Vocab import Vocab

# define some flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_path','','Pa')
tf.app.flags.DEFINE_string('log_root','','root dir to log')
tf.app.flags.DEFINE_string('exp_name','','name of experiment')
tf.app.flags.DEFINE_string('mode','train','mode of running')

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








# main
if __name__ == '__main__':
    tf.app.run()