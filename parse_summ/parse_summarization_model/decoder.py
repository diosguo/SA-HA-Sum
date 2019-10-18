from mxnet.gluon import nn, rnn
from mxnet import nd
from attention import BahdanauAttention


class Decoder(nn.Block):
    def __init__(self, word_emb, vocab):
        super(Decoder, self).__init__()
        self.cell = rnn.LSTMCell(100)
        self.attention = BahdanauAttention(100)
        self.word_emb = word_emb
        self.vocab_projection = nn.Dense(vocab.size)

    def forward(self, inputs, encoder_states, encoder_outputs, padding_mask=None):
        decoder_outputs = []
        for i in range(len(inputs)):
            context_vector, attention_weights = self.attention.forward(inputs[i], encoder_states)
            inp_step = self.word_emb(inputs[i])
            inp_step = nd.concat(inp_step, context_vector)
            dec_out = self.cell(inp_step, encoder_states)
            word_dist = self.vocab_projection(dec_out)
            word_dist = nd.softmax(word_dist)
            decoder_outputs.append(word_dist)
        return nd.stack(*decoder_outputs)
