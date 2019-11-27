from models.models import Model
from mxnet import gpu, nd, cpu
import pickle
import os
from tqdm import trange
from models.vocab import Vocab


model_param = {
    'emb_size':200,
    'hidden_size':200
}
vocab_path = 'data/vocab'
source_path = 'data/cnn_articles'
target_path = 'data/cnn_abstracts'
lda_path = 'data/cnn_head_lda'


model = Model(model_param, vocab_path, mode='decode', head_attention=True, decoder_cell='dlstm',ctx=cpu())


res = model.decode(source_path, lda_path, 'best.model')

res = [ int(i.asscalar()) for i in res.tokens]

vocab = Vocab(vocab_path)

res = [ vocab.id2word(i) for i in res]
print(' '.join(res))

abstract = pickle.load(open(os.path.join(target_path, os.listdir(target_path)[0]),'rb'))
res = [ vocab.id2word(i) for i in abstract]
print(' '.join(res))
