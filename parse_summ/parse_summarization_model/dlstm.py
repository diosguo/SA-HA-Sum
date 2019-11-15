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


class DLSTM(rnn.LSTMCell):

    """Docstring for DLSTM. """

    def __init__(self):
        """TODO: to be defined. """
        rnn.LSTMCell.__init__(self)

    def forward(self, x,state):
        """TODO: Docstring for forward.

        :x: TODO
        :state: TODO
        :returns: TODO

        """
        pass

        
