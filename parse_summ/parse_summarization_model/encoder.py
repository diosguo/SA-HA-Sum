from mxnet import nd
from mxnet.gluon import Block
from mxnet.gluon import nn, rnn


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

        self.element = []

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
            _, next_emb = self.word_set(next_emb, states=begin_state)
            next_emb = nd.reshape(next_emb[-1], [1, next_emb[-1].shape[-1]])  # 1 * C

        else:
            # Wrong Node
            raise Exception('Error with Parse Tree Node.next type' + str(type(root.nexts)))

        tag_emb = self.tag_embedding(nd.array([self.vocab_tag.word2id[root.val]]))
        emb = nd.concat(tag_emb, next_emb)
        emb = (self.word_ass(emb) + next_emb) / 2  # 残差
        self.element.append(emb)  # 记录元素，后面Attention
        return emb

    def forward(self, inputs):
        inputs = [self.emb_tree(x) for x in inputs]
        inputs = nd.stack(inputs, axis=1)  # T * C
        # 句子表示，进入BiLSTM
        h, state = self.sentence_lstm(nd.expand_dims(inputs, axis=0),
                                      states=self.sentence_lstm.begin_state(batch_size=1))
        return h, state
