from queue import Queue
from threading import Thread
import os
import glob
import random
import struct
from tensorflow.core.example import example_pb2
import tensorflow as tf
import Vocab
import data
import time

class Example(object):

    def __init__(self, article, abstract_sentences, vocab, hps):
        self.hps = hps

        start_decoding = vocab.word2id(Vocab.SPECIAL_TOKEN[4])
        stop_decoding = vocab.word2id(Vocab.SPECIAL_TOKEN[5])

        article_words = article.split()
        if len(article_words) > hps.max_enc_steps:
            article_words = article_words[:hps.max_enc_steps]
        self.enc_len = len(article_words)
        self.enc_input = [vocab.word2id(w) for w in article_words]

        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()
        abstract_ids = [vocab.word2id(w) for w in abstract_words]

        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abstract_ids, hps.max_dec_steps,
                                                                 start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        if hps.pointer_gen:
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, hps.max_dec_steps,
                                                        start_decoding, stop_decoding)

        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences



    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id]+sequence[:]
        target = sequence[:]
        if len(inp) > max_len:
            inp = inp[:max_len]
            target = target[:max_len]
        else:
            target.append(stop_id)
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input)<max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if self.hps.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)



class Batch:

    def __init__(self, example_list, hps, vocab):
        self.pad_id = vocab.word2id(Vocab.SPECIAL_TOKEN[2])
        self.init_encoder_seq(example_list, hps)
        self.init_decoder_seq(example_list, hps)
        self.store_orig_strings(example_list)

    def init_encoder_seq(self, example, hps):
        pass

    def init_decoder_seq(self, example, hps):
        pass

    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list]
        self.original_abstracts = [ex.original_abstract for ex in example_list]
        self.original_abstract_sents = [ex.originalabstract_sents for ex in example_list]


class Batcher:

    BATCH_QUEUE_MAX=100

    def __init__(self, data_path, vocab, hps, single_pass):

        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass

        self._batch_queue = Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue(self.BATCH_QUEUE_MAX*self._hps.batch_size)


        if not single_pass:
            self._num_example_q_threads = 16
            self._num_batch_q_threads = 4
            self._bucketing_cache_size = 100

        self._example_q_threads = list()
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self._fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()

        self._batch_q_threads = list()
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self._fill_batch_queue))
            self._batch_q_threads[-1].daemon=True
            self._batch_q_threads[-1].start()

        if not single_pass:
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon=True
            self._watch_thread.start()

    def _fill_example_queue(self):
        input_gen = self.text_generator(self.example_generator())


        for article, abstract in input_gen:
            abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)]
            example = Example(article, abstract_sentences, self._vocab, self._hps)
            self._example_queue.put(example)

        if self._single_pass:
            tf.logging.info('The example generator for this '
                            'example queue filling thread has exhausted data.')
            self._finished_reading = True
        else:
            raise Exception('single pass mode is off but the example generator is dead')

    def _fill_batch_queue(self):
        while True:
            if self._hps.mode != 'decode':
                inputs = []
                for _ in range(self._hps.batch_size*self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp:inp.enc_len)

                batches = []
                for i in range(0, len(inputs), self._hps.batch_size):
                    batches.append(inputs[i:i+self._hps.batch_size])
                if not self._single_pass:
                    random.shuffle(batches)
                for b in batches:
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))
            else:
                ex = self._example_queue.get()
                b = [ex for _ in range(self._hps.batch_size)]
                self._batch_queue.put(Batch(b, self._hps, self._vocab))

    def watch_threads(self):
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():
                    tf.logging.error('Found example queue thread dead. Restarting')
                    new_t = Thread(target=self._fill_example_queue)
                    new_t.daemon = True
                    new_t.start()
                    self._example_q_threads[idx] = new_t
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():
                    tf.logging.error('Found batch queue thread dead. Restarting')
                    new_t = Thread(target=self._fill_batch_queue)
                    new_t.daemon = True
                    new_t.start()
                    self._batch_q_threads[idx]=t

    def example_generator(self):
        while True:
            file_list = glob.glob(self._data_path)
            assert file_list, ('Error: Empty file_list at %s', self._data_path)
            if self._single_pass:
                file_list = sorted(file_list)
            else:
                random.shuffle(file_list)

            for f in file_list:
                reader = open(f,'rb')
                while True:
                    len_bytes = reader.read(8)
                    if not len_bytes: break
                    str_len = struct.unpack('q',len_bytes)[0]
                    example_str = struct.unpack('%ds'%str_len, reader.read(str_len))[0]
                    yield example_pb2.Example.FromString(example_str)

            if self._single_pass:
                print("example_generator completed reading all datafiles. No more data.")
                break

    def text_generator(self,example_generator):
        while True:
            e = next(example_generator)
            try:
                article_text = e.features.feature['article'].bytes_list.value[0]
                abstract_text = e.features.feature['abstract'].bytes_list.value[0]
            except ValueError:
                tf.logging.error('Failed to get article or abstract from example')
                continue
            if len(article_text)==0:
                tf.logging.warning('Found an expmple with empty article text. Skipping')
            else:
                yield (article_text,abstract_text)