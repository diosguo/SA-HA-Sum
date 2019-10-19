from mxnet import nd
from mxnet.gluon import Block
from mxnet.gluon import nn, rnn


class ParseEncoder(Block):
    def __init__(self, tag_emb, word_emb, vocab, vocab_tag, model_params):
        """

        初始化Encoder，结构见论文

        :param tag_emb: parse标记嵌入
        :param word_emb: 词嵌入
        :param vocab: 词典
        :param vocab_tag: 标记词典
        :param model_params: 模型参数
        """
        super(ParseEncoder, self).__init__()
        self.model_params = model_params
        self.tag_embedding = tag_emb
        self.word_embedding = word_emb
        self.vocab = vocab
        self.vocab_tag = vocab_tag
        # 用来处理标记和词的
        self.word_ass = nn.Dense(model_params['word_emb_dim'])
        # 用来连接多个同级元素
        self.word_set = rnn.LSTM(model_params['word_emb_dim']/2, layout='NTC', bidirectional=True)
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
        if isinstance(root.next, str):
            # Leaf Node，如果是叶子节点，也就是单独词的情况
            # next shape = 1 * word_size
            if root.next in self.vocab.word2id:
                next_emb = self.word_embedding(nd.array([self.vocab.word2id[root.next]]))
            else:
                next_emb = self.word_embedding(nd.array([self.vocab.word2id['<unk>']]))
        elif isinstance(root.next, list):
            # Mid Node
            # 非叶子节点，则需要计算所有子节点的整合向量
            begin_state = self.word_set.begin_state(batch_size=1)
            next_emb = []
            for i in root.next:
                next_emb.append(self.emb_tree(i))
            _, next_emb = self.word_set(next_emb, states=begin_state)
            next_emb = nd.reshape(next_emb[-1], [1, next_emb[-1].shape[-1]])  # 1 * C

        else:
            # Wrong Node
            raise Exception('Error with Parse Tree Node.next type' + str(type(root.nexts)))

        # 将标签与嵌入向量整合
        tag_emb = self.tag_embedding(nd.array([self.vocab_tag.word2id[root.val]]))
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

        inputs = [self.emb_tree(x) for x in inputs]
        inputs = nd.stack(*inputs, axis=1)  # T * C
        # 句子表示，进入BiLSTM
        # print(inputs)
        h, state = self.sentence_lstm(inputs,
                                      states=self.sentence_lstm.begin_state(batch_size=1))
        return h, state, nd.stack(self.element)
