from mxnet import nd
from mxnet.gluon import Block
from mxnet.gluon import nn
from mxnet import nd, autograd
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from .vocab import Vocab
from .encoder import ParseEncoder
from .decoder import BaseDecoder
from mxnet import cpu


class BaseModel(Block):
    """

    BaseLine 模型，使用论文Encoder，普通LSTMDecoder，使用Sentence级Attention，无Pointer及Coverage

    """
    def __init__(self, vocab, vocab_tag, model_params,ctx=cpu(0)):
        """

        初始化模型

        :param vocab: 词典类 Vocab
        :param vocab_tag: 标签的词典类 Vocab
        :param model_param: 用来传递模型参数的字典 dict
        """

        super(BaseModel, self).__init__()
        self.model_params = model_params
        self.tag_embedding = nn.Embedding(vocab_tag.size, model_params['tag_emb_dim'])
        self.word_embedding = nn.Embedding(vocab.size, model_params['word_emb_dim'])
        params = [self.tag_embedding, self.word_embedding, vocab, vocab_tag]
        self.encoder = ParseEncoder(*params, model_params,ctx=ctx)
        self.decoder = BaseDecoder(self.word_embedding, vocab, model_params,ctx=ctx)

    def forward(self, inputs, targets):
        # print('starting encode')
        encoder_h, encoder_state, attention_value= self.encoder(inputs)  # N * T * 2C
        encoder_c = encoder_state[1]  # 2 * T * C
        # print('end encode')
        return self.decoder(targets, encoder_c, encoder_h, attention_value)


class Model(object):
    """
    最顶层框架，控制模型参数、文件路径、梯度训练
    """

    def __init__(self, mode, vocab_path, vocab_tag_path, model_param, original_path, ctx=cpu(), summary_path=None):
        """

        根据参数与模式，构建模型

        :param mode: train|decode|test 控制当前模型的用途
        :param original_path: 被句法解析后的文件路径
        :param summary_path: 摘要文档的路径
        :param vocab_path: 词典路径
        :param vocab_tag_path: 句法解析标记词典路径
        :param model_param: 模型中的超参数
        """
        self.original_path = original_path
        self.summary_path = summary_path
        self.vocab_path = vocab_path
        self.vocab_tag_path = vocab_tag_path
        self.vocab_tag = Vocab(vocab_tag_path)
        self.vocab = Vocab(vocab_path)
        self.mode = mode
        self.loss = SoftmaxCrossEntropyLoss()
        self.model_param = model_param
        self.model = BaseModel(self.vocab, self.vocab_tag, model_param, ctx)
        self.model.initialize(ctx=ctx)
        self.ctx = ctx
        self.trainer = Trainer(self.model.collect_params(), 'adam', {'learning_rate': 0.01})

    def sequence_loss(self, logits, targets, weight=None):
        """

        计算序列的损失，也就是一个批次的损失（当然这个模型是一个批次）

        :param logits: 预测出的结果
        :param targets: 真实摘要
        :param weight: 根据句子长度来的padding weight，训练过程不需要
        :return: 损失值
        """
        if weight is None:
            logits = nd.reshape(logits, [-1, self.vocab.size])
        else:
            logits = logits * weight
            targets = logits * weight
        targets = nd.array(targets,ctx=self.ctx)
        loss = self.loss(logits, targets)
        loss = loss.sum() / len(targets)
        return loss


    def train_one_step(self, data, optimizer='adam', learning_rate=0.01):

        loss_sum = 0.0
        for x, y in data:

            with autograd.record():
                logits = self.model(x, y)
                loss = self.sequence_loss(logits, y)
            loss.backward()
            self.trainer.step(1)
            loss_sum += loss.asscalar()
        print('loss= %.3f' % (loss_sum / len(data)))
        # self.model.collect_params().save('')

    def run(self, epoch_num=15, optimizer='adam', learning_rate=0.01):
        """

        根据定义好的模型，训练模型

        :param epoch_num: 训练迭代数据轮次
        :param optimizer:  优化器，或者是优化器名字str
        :param learning_rate: 学习率
        :return:
        """

        trainer = Trainer(self.model.collect_params(), optimizer, {'learning_rate': learning_rate})
        data = []
        for epoch in range(epoch_num):
            loss_sum = 0.0
            for x, y in data:
                with autograd.record():
                    logits = self.model(x, y)
                    loss = self.sequence_loss(logits, y)
                loss.backward()
                trainer.step(1)
                loss_sum += loss.asscalar()
            print('epoch %d, loss= %.3f' % (epoch + 1, loss_sum / len(data)))
        self.model.collect_params().save('')


class Seq2SeqRNN(nn.Block):

    """implemention of alesee seq2seq and beyond with mxnet"""

    def __init__(self, rnn_type, input_size, emb_size, hidden_size, batch_size, output_size, max_tgt_len, attention_type, tied_weight_type, pre_trained_vector, pre_trained_vector_type, padding_id, num_layers=1, encoder_drop=(0.2,0.3), decoder_drop=(0.2,0.3),bidirectional=True, bias=False, teacher_forcing=True):
        """TODO: to be defined.

        :rnn_type: TODO
        :input_size: TODO
        :emb_size: TODO
        :hidden_size: TODO
        :batch_size: TODO
        :output_size: TODO
        :max_tgt_len: TODO
        :attention_type: TODO
        :tied_weight_type: TODO
        :pre_trained_vector: TODO
        :pre_trained_vector_type: TODO
        :padding_id: TODO
        :num_layers: TODO
        :encoder_drop: TODO
        :decoder_drop: TODO
        :bidirectional: TODO
        :bias: TODO
        :teacher_forcing: TODO

        """
        nn.Block.__init__(self)
        rnn_type, attention_type, tied_weight_type = rnn_type.upper(), attention_type.title(), tied_weight_type.lower()
        if rnn_type in ['LSTM','GRU']:
            self._rnn_type = rnn_type
        else:
            raise ValueError("""Invalid option for 'rnn_type' was supplied, options are ['LSTM','GRU']""")
        
        if attention_type in ['Luong','Bahdanau']:
            self.attention_type = attention_type
        else:
            raise ValueError("""Invalid option for 'attention_type', options are ['Luong','Bahdanau']""")

        if tied_weight_type in ['three_way','two_way']:
            self.tied_weight_type = tied_weight_type
        else:
            raise ValueError("""Invalid option for 'tied_weight_type' options are ['three_way','two_way']""")

        self.input_size = input_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size//2
        self.batch_size = batch_size
        self.output_size = output_size
        self.max_tgt_len = max_tgt_len
        self.pre_trained_vector = pre_trained_vector
        self.pre_trained_vector_type = pre_trained_vector_type
        self.padding_id = padding_id
        self.num_layers = num_layers
        self.encoder_drop = encoder_drop
        self.decoder_drop = decoder_drop
        self.bidirectional = bidirectional
        self.bias = bias
        self.teacher_forcing = teacher_forcing
        
        if self.teacher_forcing:
            self.force_prob = 0.5

        if self.bidirectional:
            self.num_directions=2
        else:
            self.num_directions = 1

        #TODO Encoder and Decoder
