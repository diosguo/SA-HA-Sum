from mxnet import nd
from mxnet.gluon import Block
from mxnet.gluon import nn, rnn
from mxnet import nd
from stanfordcorenlp import StanfordCoreNLP
from parse_parse import parse
from vocab import Vocab


class ParseEncoder(Block):
    def __init__(self, tag_emb, word_emb, vocab, vocab_tag):
        super(ParseEncoder, self).__init__()
        self.tag_embedding = tag_emb
        self.word_embedding = word_emb
        self.vocab = vocab
        self.vocab_tag = vocab_tag
        self.word_ass = nn.Dense(100)
        self.word_set = rnn.LSTM(50, layout='NTC', bidirectional=True)
        self.sentence_lstm = rnn.LSTM(100, layout='NTC', bidirectional=True)

    def emb_tree(self, root):
        next_emb = None
        if isinstance(root.next, str):
            # Leaf Node
            # next shape = 1 * word_size
            if root.next in self.vocab.word2id:
                next_emb = self.word_embedding(nd.array([self.vocab.word2id[root.next]]))
            else:
                next_emb = self.word_embedding(nd.array([self.vocab.word2id['<unk>']]))
        elif isinstance(root.next, list):
            # Mid Node
            begin_state = self.word_set.begin_state(batch_size=1)
            next_emb = []
            for i in root.next:
                next_emb.append(self.emb_tree(i))
            print('@emb_tree list')
            print(next_emb)
            next_emb = nd.stack(*next_emb,axis=1)  # T * C
            print('@stacked')
            print(next_emb)
            # next_emb = nd.expand_dims(next_emb, axis=0)  # 1 * T * C
            print(next_emb)
            _, next_emb = self.word_set(next_emb, states=begin_state)
            next_emb = nd.reshape(next_emb[-1], [1, next_emb[-1].shape[-1]])  # 1 * C

        else:
            # Wrong Node
            raise Exception('Error with Parse Tree Node.next type' + str(type(root.nexts)))

        tag_emb = self.tag_embedding(nd.array([self.vocab_tag.word2id[root.val]]))
        emb = nd.concat(tag_emb, next_emb)

        return self.word_ass(emb)

    def forward(self, inputs):
        inputs = [self.emb_tree(x) for x in inputs]
        inputs = nd.stack(inputs)  # T * C
        h, state = self.sentence_lstm(nd.expand_dims(inputs, axis=0), states=self.sentence_lstm.begin_state(batch_size=1))
        return h, state


class BaseModel(Block):
    def __init__(self, vocab, vocab_tag):
        super(BaseModel, self).__init__()
        self.tag_embedding = nn.Embedding(vocab_tag.size, 100)
        self.word_embedding = nn.Embedding(vocab.size, 100)
        params = [self.tag_embedding, self.word_embedding, vocab, vocab_tag]
        self.encoder = ParseEncoder(*params)

    def forward(self, inputs):
        return self.encoder(inputs)


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
        print(encoder(inputs))




