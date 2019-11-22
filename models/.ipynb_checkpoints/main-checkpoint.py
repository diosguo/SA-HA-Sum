import json
import time
from stanfordcorenlp import StanfordCoreNLP
from parse_summarization_model.model import Model
from parse_summarization_model.parse_parse import parse, TNode
import pickle
from mxnet import gpu

debug = True

params = json.load(open('config.json','r'))


model = Model(
    params['mode'],
    params['vocab_path'],
    params['vocab_tag_path'],
    params['model_param'],
    params['original_path']
)

if __name__ == '__main__':

    if debug is True:
        x = pickle.load(open('cnn_tree/000c835555db62e319854d9f8912061cdca1893e.story','rb'))
        y = pickle.load(open('cnn_abstract/000c835555db62e319854d9f8912061cdca1893e.story','rb'))
        print(str(x[0]))
        print(y)
        for i in range(10):
            model.train_one_step([[x, y]])


