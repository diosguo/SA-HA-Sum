from mxnet import nd
from mxnet.gluon import Block
from mxnet.gluon import nn, rnn
from mxnet import nd
from stanfordcorenlp import StanfordCoreNLP
from parse_parse import parse
from vocab import Vocab
from encoder import ParseEncoder


class BaseModel(Block):
    def __init__(self, vocab, vocab_tag):
        super(BaseModel, self).__init__()
        self.tag_embedding = nn.Embedding(vocab_tag.size, 100)
        self.word_embedding = nn.Embedding(vocab.size, 100)
        params = [self.tag_embedding, self.word_embedding, vocab, vocab_tag]
        self.encoder = ParseEncoder(*params)

    def forward(self, inputs, targets):
        encoder_h, encoder_state = self.encoder(inputs)  # N * T * 2C
        encoder_c = encoder_state[1]  # 2 * T * C
        return encoder_h, encoder_c


class Model(object):
    def __init__(self, mode, original_path, summary_path, vocab_path, vocab_tag_path):
        self.original_path = original_path
        self.summary_path = summary_path
        self.vocab_path = vocab_path
        self.vocab_tag_path = vocab_tag_path
        self.vocab_tag = Vocab(vocab_tag_path)
        self.vocab = Vocab(vocab_path)
        self.mode = mode

    def encoder_test(self, inputs):
        """

        :param inputs:  List of parse tree
        :return:
        """
        encoder = BaseModel(self.vocab, self.vocab_tag)
        encoder.initialize()
        print(encoder(inputs, None))
