#!/usr/bin/env python3
"""
File: dlstm.py
Author: yourname
Email: 943024256@qq.com
Github: theDoctor2013
Description: double memory cell LSTM with mxnet
"""

from mxnet import nd
from mxnet.gluon import nn, rnn


class DLSTMCell(rnn.LSTMCell):

    """Docstring for DLSTM. """

    def __init__(self, hidden_size):
        """TODO: to be defined. """
        rnn.LSTMCell.__init__(self, hidden_size)
        

    def hybrid_forward(self, F, inputs, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias):
        """TODO: Docstring for forward.

        :x: TODO
        :state: TODO
        :returns: TODO

        """
        prefix = 't%d'%self._counter
        i2h = F.FullyConnected(data=inputs, weight=i2h_weight, bias=i2h_bias, num_hidden=self._hidden_size*4, name=prefix+'i2h')
        h2h = F.FullyConnected(data=states[0], weight=h2h_weight, bias=h2h_bias, num_hidden=self._hidden_size*4, name=prefix+'h2h')
        gates = F.elemwise_add(i2h, h2h, name=prefix+'plus0')
        slice_gates = F.SliceChannel(gates, num_outputs=4, name=prefix+'slice')

        in_gate = self._get_activation(
                F, slice_gates[0], self._recurrent_activation, name=prefix+'f'
                )
        forget_gate = self._get_activation(
                F, slice_gates[1], self._recurrent_activation, name=prefix+'f'
                )

        in_transform = self._get_activation(
                F, slice_gates[2], self._activation, name=prefix+'c'
                )

        out_gate = self._get_activation(
                F, slice_gates[3], self._recurrent_activation, name=prefix+'o'
                )

        to_input = F.elemwise_mul(in_gate, in_transform, name=prefix+'mul0')
        next_c = F.elemwise_sub(
                F.elemwise_mul(forget_gate, states[1], name=prefix+'mul1'),
                to_input,
                name=prefix+'minus0'
                )
        ones = F.ones_like(forget_gate, name=prefix+'ones_like0')
        next_s = F.elemwise_add(
                    F.elemwise_mul(
                        F.elemwise_sub(ones, forget_gate, name=prefix+'minus1'),
                        states[2], name=prefix+'mul2'
                    ),
                    to_input,name=prefix+'add'
                )
        next_h = F.elemwise_mul(out_gate, F.Activation(next_c, act_type=self._activation, name=prefix+'activation0'), name=prefix+'out')

        return next_h, [next_h, next_c, next_s]



    def state_info(self, batch_size=0):
        return [{'shape': (batch_size, self._hidden_size), '__layout__': 'NC'},
                {'shape': (batch_size, self._hidden_size), '__layout__': 'NC'},
                {'shape': (batch_szie, self._hidden_size), '__layout__': 'NC'}]



class DLSTM(nn.Block):

    """Docstring for DLSTM. """

    def __init__(self, hidden_size, dropout, state=None):
        """TODO: to be defined.
        
        layout: NTC

        :hidden_size: TODO
        :dropout: TODO
        :state: TODO

        """
        nn.Block.__init__(self)

        self.cell = DLSTMCell(hidden_size)

    def forward(self, x, state=None):
        """TODO: Docstring for forward.

        :x: TODO
        :state: TODO
        :returns: TODO

        """
        outputs = []
        hidden = x[:,0] 
        if state = None:
            state = self.cell.begin_state(batch_size=x.shape[0])

        for i in range(x.shape[1]):
            hidden, state = self.cell(hidden, state)
            outputs.append(hidden)


        
        return nd.stack(*output, dim=1)
        

        



if __name__ == "__main__":
    lstm = DLSTMCell()
