import random
from mxnet.gluon import nn, rnn
from mxnet import nd
from attentions import BahdanauAttention, LuongAttention
from dlstm import DLSTMCell

class BaseDecoder(nn.Block):
    """

    最基础的Decoder，LSTM，只有句子级Attention

    """
    def __init__(self, word_emb, vocab, model_params,ctx):
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
        self.ctx=ctx

    def forward(self, inputs, encoder_states, encoder_outputs, encoder_elem, padding_mask=None):
        """

        对Encoder的输出进行解码，生成摘要

        :param inputs: 真实摘要句子
        :param encoder_states: Encoder输出的States
        :param encoder_outputs: Encoder每一步的输出h
        :param padding_mask: 训练时几乎没用
        :return: 返回nd.array，其中为输出的每个词的字典分布
        """
        # print('begin_state',self.cell.begin_state(batch_size=1))
        self.cell(nd.ones((1,200),ctx=self.ctx), self.cell.begin_state(batch_size=1, ctx=self.ctx))
        decoder_outputs = []
        for i in range(len(inputs)):
            # 根据当前真实摘要，计算Attention
            inp_step = self.word_emb(nd.array([inputs[i]],ctx=self.ctx))
            context_vector, attention_weights = self.attention(inp_step, encoder_elem)
            # print('atttention_values:', encoder_elem.shape)
            # print(context_vector.shape, attention_weights.shape, inp_step.shape)
            inp_step = nd.concat(inp_step, context_vector)
            # 传入lstm计算输出，忽略states
            # print('inp_step:',inp_step.shape)
            if i == 0:
                hidden = self.cell.begin_state(batch_size=1, ctx=self.ctx)[1]
                states = [nd.reshape(encoder_states, (1, -1)), hidden]
            # print('encoder_states:',encoder_states.shape)
            # print('hidden_size:', self.model_params['decoder_hidden_size'])
            dec_out, states = self.cell(inp_step, states)
            word_dist = self.vocab_projection(dec_out)
            word_dist = nd.softmax(word_dist)
            decoder_outputs.append(word_dist)
        return nd.stack(*decoder_outputs)


class RNNDecoder(nn.Block):

    """Docstring for RNNDecoder. """

    def __init__(self, rnn_type, hidden_size, emb_size, output_size, dropout, target_len, teaching_force, force_prob):
        """TODO: to be defined.

        :hidden_size: TODO
        :emb_size: TODO
        :dropout: TODO
        :target_len: TODO

        """
        nn.Block.__init__(self)

        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.dropout = dropout
        self.target_len = target_len
        self.teaching_force = teaching_force
        self.force_prob = force_prob

        rnn_type = rnn_type.upper()

        self.rnn_type = rnn_type.upper()

        if rnn_type == 'LSTM':
            self.rnn = rnn.LSTM(hidden_size, layout='NTC', dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = rnn.GRU(hidden_size, layout='NTC', dropout=dropout)
        elif rnn_type == 'DLSTM':
            self.rnn = DLSTMCell(hidden_size)
        else:
            raise ValueError('Unspport rnn type %s'%rnn_type)

        self.attention = LuongAttention(64)

        self.input_linear = nn.Dense(emb_size)
        self.output_layer = nn.Dense(output_size)


    def forward(self, batch_size, encoder_output, decoder_hidden, y=None):
        """TODO: Docstring for forward.

        :batch_size: TODO
        :encoder_output: TODO
        :encoder_hidden: TODO
        :y: TODO
        :returns: TODO

        """
        
        output_seq = []

        decoder_input =y[0]

        for i in self.target_len:
            atten_weight, atten_context = self.attention(decoder_input, encoder_output)
            context = self.input_linear(nd.concat(decoder_input, atten_context))
            decoder_output, decoder_hidden = self.rnn(context, decoder_hidden)
            
            output = self.output_layer(decoder_output)

            decoder_input = output

            if self.teaching_force:
                if y is not None and round(random.random(),1) < self.force_prob:
                    if i < len(y):
                        decoder_input = y[i]

        return nd.stack(*output_seq)


                    
            
            


                
            
