from mxnet.gluon import nn
from mxnet import nd


class BahdanauAttention(nn.Block):
    """

    Bahdanau Attention模型

    """
    def __init__(self, units):
        """

        构建Attention模型

        :param units: 注意力机制隐层单元大小
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Dense(units, flatten=False)
        self.W2 = nn.Dense(units, flatten=False)
        self.V = nn.Dense(1,flatten=False)

    def forward(self, query, values, head=False):
        """

        计算Attention权重与输出向量

        :param query: 查询，即当前步Decoder的输入
        :param values: 值，即Encoder中每一个时间步向量
        :return: (Attention输出向量， Attention权重)
        """
        #print('In Attention')
        hidden_with_time_axis = nd.expand_dims(query, 1)
        #print('hidden_with_time:', hidden_with_time_axis.shape)
        score = self.V(nd.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)
        ))
        #print('\t score:',score.shape)
        attention_weights = nd.softmax(score, axis=1)
        
        #print('\t attention_weight:', attention_weights.shape)
        #print('\t values:', values.shape)
        
        context_vector = attention_weights * values
        
        #print('\t mid_context_vector:',context_vector.shape)
        
        if head is True:
            context_vector = nd.sum(context_vector, axis=2)
        else:
            context_vector = nd.sum(context_vector, axis=1)
        # print('\t context',context_vector.shape)
        context_vector = nd.expand_dims(context_vector,axis=0)
        return context_vector, attention_weights


class LuongAttention(nn.Block):

    """Docstring for LuongAttention. """

    def __init__(self, units):
        """TODO: to be defined.

        :units: TODO

        """
        nn.Block.__init__(self)

        self._units = units

    def forward(self, decoder_output, encoder_output):
        """TODO: Docstring for forward.

        :decoder_output: TODO
        :encoder_output: TODO
        :returns: TODO

        """
        
        decoder_output = decoder_output.transpose([0,2,1])

        score = nd.batch_dot(encoder_output, decoder_output)

        weight = nd.softmax(score,axis=1)

        context = nd.batch_dot(nd.transpose(weight,[0,2,1]), encoder_output)

        return context, nd.squeeze(weight)

