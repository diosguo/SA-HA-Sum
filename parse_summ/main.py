import json
import time
from stanfordcorenlp import StanfordCoreNLP
from model import Model
from parse_parse import parse, TNode


params = json.load(open('config.json','r'))


model = Model(**params['model_param'])
try:
    t = TNode()
    t.next = 'hello'
    t.val = 'ROOT'
    model.encoder_test([t, t])
finally:
    pass