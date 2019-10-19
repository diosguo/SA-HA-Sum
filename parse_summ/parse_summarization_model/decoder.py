from mxnet.gluon import nn, rnn
from mxnet import nd
from attention import BahdanauAttention


class BaseDecoder(nn.Block):
    """

    最基础的Decoder，LSTM，只有句子级Attention

    """
    def __init__(self, word_emb, vocab, model_params):
        """

        初始化Decoder与Attention

        :param word_emb: 词嵌入模型，用来获取词向量
        :param vocab: 词典
        :param model_params: 模型参数
        """
        super(BaseDecoder, self).__init__()
        self.model_params = model_params
        self.cell = rnn.LSTMCell(model_params['decoder_hidden_size'])
        self.attention = BahdanauAttention(model_params['attention_hidden_size'])
        self.word_emb = word_emb
        self.vocab_projection = nn.Dense(vocab.size)

    def forward(self, inputs, encoder_states, encoder_outputs, padding_mask=None):
        """

        对Encoder的输出进行解码，生成摘要

        :param inputs: 真实摘要句子
        :param encoder_states: Encoder输出的States
        :param encoder_outputs: Encoder每一步的输出h
        :param padding_mask: 训练时几乎没用
        :return: 返回nd.array，其中为输出的每个词的字典分布
        """
        decoder_outputs = []
        for i in range(len(inputs)):
            # 根据当前真实摘要，计算Attention
            inp_step = self.word_emb(inputs[i])
            context_vector, attention_weights = self.attention.forward(inp_step, encoder_states)
            inp_step = nd.concat(inp_step, context_vector)
            # 传入lstm计算输出，忽略states
            dec_out, _ = self.cell(inp_step, encoder_states)
            word_dist = self.vocab_projection(dec_out)
            word_dist = nd.softmax(word_dist)
            decoder_outputs.append(word_dist)
        return nd.stack(*decoder_outputs)
