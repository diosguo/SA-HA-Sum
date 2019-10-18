from mxnet import nd
from mxnet.gluon import Block
from mxnet.gluon import nn, rnn
from mxnet import nd, autograd

from mxnet.gluon import Trainer

from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from stanfordcorenlp import StanfordCoreNLP
from parse_parse import parse
from vocab import Vocab
from encoder import ParseEncoder
from decoder import Decoder


class BaseModel(Block):
    def __init__(self, vocab, vocab_tag):
        super(BaseModel, self).__init__()
        self.tag_embedding = nn.Embedding(vocab_tag.size, 100)
        self.word_embedding = nn.Embedding(vocab.size, 100)
        params = [self.tag_embedding, self.word_embedding, vocab, vocab_tag]
        self.encoder = ParseEncoder(*params)
        self.decoder = Decoder(self.word_embedding, vocab)

    def forward(self, inputs, targets):
        encoder_h, encoder_state = self.encoder(inputs)  # N * T * 2C
        encoder_c = encoder_state[1]  # 2 * T * C

        return self.decoder(targets, encoder_c, encoder_h)


class Model(object):
    def __init__(self, mode, original_path, summary_path, vocab_path, vocab_tag_path):
        self.original_path = original_path
        self.summary_path = summary_path
        self.vocab_path = vocab_path
        self.vocab_tag_path = vocab_tag_path
        self.vocab_tag = Vocab(vocab_tag_path)
        self.vocab = Vocab(vocab_path)
        self.mode = mode
        self.loss = SoftmaxCrossEntropyLoss()
        self.model = BaseModel(self.vocab, self.vocab_tag)

    def sequence_loss(self, logits, targets, weight=None):
        if weight is None:
            logits = nd.reshape(logits, [-1, self.vocab.size])
        else:
            logits = logits * weight
            targets = logits * weight
        loss = self.loss(logits, targets)

        return loss

    def run(self):
        trainer = Trainer(self.model.collect_params(), 'adam', {'learning_rate': 0.01})
        data = []
        epoch_num = 10
        for epoch in range(epoch_num):
            loss_sum = 0.0
            for x, y in data:
                with autograd.record():
                    logits = self.model(x, y)
                    loss = self.sequence_loss(logits, y)
                loss.backward()
                trainer.step(1)
                loss_sum += loss.asscalar()
            print('epoch %d, loss= %.3f'%(epoch+1, loss_sum/len(data)))
        self.model.collect_params().save('')


    def encoder_test(self, inputs):
        """

        :param inputs:  List of parse tree
        :return:
        """
        encoder = BaseModel(self.vocab, self.vocab_tag)
        encoder.initialize()
        print(encoder(inputs, None))
