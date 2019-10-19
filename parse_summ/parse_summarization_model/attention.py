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
        self.W1 = nn.Dense(units)
        self.W2 = nn.Dense(units)
        self.V = nn.Dense(1)

    def forward(self, query, values):
        """

        计算Attention权重与输出向量

        :param query: 查询，即当前步Decoder的输入
        :param values: 值，即Encoder中每一个时间步向量
        :return: (Attention输出向量， Attention权重)
        """

        hidden_with_time_axis = nd.expand_dims(query, 1)

        score = self.V(nd.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)
        ))

        attention_weights = nd.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = nd.sum(context_vector, axis=1)

        return context_vector, attention_weights
