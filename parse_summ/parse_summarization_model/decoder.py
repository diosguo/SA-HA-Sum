from mxnet.gluon import nn, rnn


class Decoder(nn.Block):
    def __init__(self):
        super(Decoder, self).__init__()
        cell = rnn.LSTMCell(100)

    def forward(self, inputs, encoder_states):
        pass
