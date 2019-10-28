from make_parse import parse_stories
import os
import json
from stanfordcorenlp import StanfordCoreNLP
import parsed2tree 
from ..parse_summarization_model.vocab import Vocab


config = json.load(open("../config.json",'r'))
nlp = StanfordCoreNLP(config['stanford_path'])
vocab = Vocab(config['vocab_path'])
vocab_tag = Vocab(config['vocab_tag_path'])
path_to_save_parsed = 'parsed'
path_to_save_tree = 'tree'
path_to_cnn_stories = 'cnn_stories'
path_to_dm_stories = 'dm_stories'
if not os.path.exists(path_to_save_parsed):
    os.mkdir(path_to_save_parsed)
if not os.path.exists(path_to_save_tree):
    os.mkdir(path_to_save_tree)

print('parse cnn')
parse_stories(path_to_cnn_stories, path_to_save_parsed)
print('parse dm')
parse_stories(path_to_dm_stories, path_to_save_parsed)

parsed2tree.main(path_to_save_parsed, path_to_save_tree)

