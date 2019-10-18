from mxnet.gluon import nn
from mxnet import nd


class BahdanauAttention(nn.Block):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Dense(units)
        self.W2 = nn.Dense(units)
        self.V = nn.Dense(1)

    def forward(self, query, values):

        hidden_with_time_axis = nd.expand_dims(query, 1)

        score = self.V(nd.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)
        ))

        attention_weights = nd.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = nd.sum(context_vector, axis=1)

        return context_vector, attention_weights
