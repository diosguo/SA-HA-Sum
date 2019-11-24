from mxnet import nd, cpu
from mxnet.gluon import Block
from mxnet.gluon import nn, rnn


class ParseEncoder(Block):
    def __init__(self, tag_emb, word_emb, vocab, vocab_tag, model_params, ctx):
        """

        初始化Encoder，结构见论文

        :param tag_emb: parse标记嵌入
        :param word_emb: 词嵌入
        :param vocab: 词典
        :param vocab_tag: 标记词典
        :param model_params: 模型参数
        """
        super(ParseEncoder, self).__init__()
        self.ctx = ctx
        self.model_params = model_params
        self.tag_embedding = tag_emb
        self.word_embedding = word_emb
        self.vocab = vocab
        self.vocab_tag = vocab_tag
        # 用来处理标记和词的
        self.word_ass = nn.Dense(model_params['word_emb_dim'])
        # 用来连接多个同级元素
        self.word_set = rnn.LSTM(model_params['word_emb_dim'], layout='NTC', bidirectional=False)
        # 对句子做传统Encoder的操作
        self.sentence_lstm = rnn.LSTM(model_params['encoder_lstm_dim'], layout='NTC', bidirectional=True)
        # 存储encoder的输出结果
        self.element = []

    def emb_tree(self, root):
        """

        将句法解析转换为向量形式

        :param root: 句子元素的TNode根节点
        :return: 输入元素的嵌入向量
        """
        next_emb = None
        if isinstance(root.next, int):
            # Leaf Node，如果是叶子节点，也就是单独词的情况
            # next shape = 1 * word_size
            next_emb = self.word_embedding(nd.array([root.next],ctx=self.ctx))
        elif isinstance(root.next, list):
            # Mid Node
            # 非叶子节点，则需要计算所有子节点的整合向量
            begin_state = self.word_set.begin_state(batch_size=1, ctx=self.ctx)
            next_emb = []
            for i in root.next:
                next_emb.append(self.emb_tree(i))
            next_emb = nd.stack(*next_emb,axis=1)
            _, next_emb = self.word_set(next_emb, states=begin_state)
            next_emb = nd.reshape(next_emb[-1], [1, next_emb[-1].shape[-1]])  # 1 * C

        else:
            # Wrong Node
            raise Exception('Error with Parse Tree Node.next type' + str(type(root.nexts)))

        # 将标签与嵌入向量整合
        tag_emb = self.tag_embedding(nd.array([self.vocab_tag.word2id(root.val)],ctx=self.ctx))
        emb = nd.concat(tag_emb, next_emb)
        emb = (self.word_ass(emb) + next_emb) / 2  # 残差
        self.element.append(emb)  # 记录元素，后面元素Attention
        return emb

    def forward(self, inputs):
        """

        对输入进行编码，得到全文的语义向量

        :param inputs: 输入，存储解析树TNode的列表，长度等于句子数量
        :return: h, state, 结点元素向量
        """

        self.element = []
        inputs = [self.emb_tree(x) for x in inputs]
        inputs = nd.stack(*inputs, axis=1)  # T * C
        # 句子表示，进入BiLSTM
        # print(inputs)
        h, state = self.sentence_lstm(inputs,
                                      states=self.sentence_lstm.begin_state(batch_size=1, ctx=self.ctx))
        atten = nd.concat(*self.element, dim=0)
        self.element = []
        return h, state, atten



class RNNEncoder(nn.Block):

    """Encoder with LSTM or GRU"""

    def __init__(self,rnn_type, hidden_size, output_size, num_layers, dropout, bidirectional=True, ctx=cpu()):
        """TODO: to be defined.

        :hidden_size: TODO
        :num_layers: TODO
        :dropout: TODO
        :bidirectional: TODO

        """
        nn.Block.__init__(self)
        
        self._rnn_type = rnn_type.upper()
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._bidirectional = bidirectional
        self.ctx = ctx
       
        if self._rnn_type == 'LSTM':
            self.rnn = rnn.LSTM(self._hidden_size, self._num_layers, 'NTC', self._dropout, self._bidirectional)
        elif self._rnn_type == 'GRU':
            self.rnn = rnn.GRU(self._hidden_size, self.num_layers, 'NTC', self._dropout, self._bidirectional)

        # self.linear = nn.Dense(self._output_size)

    def forward(self, seq):
        """TODO: Docstring for forward.

        :seq: TODO
        :returns: TODO

        """
        batch_size = seq.shape[0]
        begin_state = self.rnn.begin_state(batch_size=batch_size, ctx=self.ctx)

        output, hidden = self.rnn(seq, begin_state)
        # hidden[0] = nd.transpose(hidden[0],[1,0,2])
        # hidden[1] = nd.transpose(hidden[1], [1,0,2])
        
        return output, hidden


