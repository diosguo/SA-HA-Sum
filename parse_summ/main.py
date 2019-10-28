import json
import time
from stanfordcorenlp import StanfordCoreNLP
from model import Model
from parse_parse import parse, TNode

params = json.load(open('config.json','r'))


model = Model(
    params['mode'],
    params['vocab_path'],
    params['vocab_tag_path'],
    params['model_param'],
    params['original_path'],
    params['summary_path']
)


