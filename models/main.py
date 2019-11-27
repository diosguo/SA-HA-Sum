from models.models import Model
from mxnet import gpu, nd
import pickle
import os
from tqdm import trange, tqdm


model_param = {
    'emb_size':200,
    'hidden_size':200
}
vocab_path = 'data/vocab'
source_path = 'data/cnn_articles'
target_path = 'data/cnn_abstracts'
lda_path = 'data/cnn_head_lda'

model = Model(model_param, vocab_path, head_attention=True, decoder_cell='dlstm')

model.train(source_path, target_path, lda_path, epoch_num=50, learning_rate=0.00001)
