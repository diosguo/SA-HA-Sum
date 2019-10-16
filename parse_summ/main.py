import json
import time
from stanfordcorenlp import StanfordCoreNLP
from model import Model
from parse_parse import parse


params = json.load(open('config.json','r'))
nlp = StanfordCoreNLP(params['stanford_path'])

model = Model(**params['model_param'])
print(nlp.parse('I love you'))
try:
    model.encoder_test([parse(nlp.parse('I love you'))])
finally:
    nlp.close()
