import json
import time
from stanfordcorenlp import StanfordCoreNLP

print(json.load(open('config.json','r')))
params = json.load(open('config.json','r'))
nlp = StanfordCoreNLP(params['stanford_path'])
print('start')
start_time = time.time()
for i in range(100):
    nlp.parse('I love learning.')
print(time.time()-start_time)
