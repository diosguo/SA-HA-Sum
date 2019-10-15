import json
from stanfordcorenlp import StanfordCoreNLP
import os


if __name__ == '__main__':
    config = json.load('config.json')
    cnn_stories_dir = config['']
    cnn_parsed_stories_dir = 'cnn_parsed'

    if not os.path.exists(cnn_parsed_stories_dir):
        os.makedirs(cnn_parsed_stories_dir)

    parse_stories()
