import json
from stanfordcorenlp import StanfordCoreNLP
import os
import pickle
from tqdm import tqdm


def parse_stories(stories_path, parsed_stories_path):
    filenames = os.listdir(stories_path)
    ind = 0
    for filename in tqdm(filenames):
        output_path = os.path.join(parsed_stories_path, filename)
        parsed_doc = []
        for line in open(os.path.join(stories_path, filename), 'r', encoding='utf-8'):
            if line.strip() == '':
                continue
            if line.strip().startswith('@highlight'):
                break
            try:
                parsed_doc.append(nlp.parse(line.strip()))
            except:
                continue
        pickle.dump(parsed_doc, open(output_path, 'wb'))


if __name__ == '__main__':

    config = json.load(open('../config.json', 'r'))
    nlp = StanfordCoreNLP(config['stanford_path'])
    cnn_stories_dir = r'cnn_stories/cnn/stories'
    cnn_parsed_stories_dir = 'cnn_parsed'

    if not os.path.exists(cnn_parsed_stories_dir):
        os.makedirs(cnn_parsed_stories_dir)

    parse_stories(cnn_stories_dir, cnn_parsed_stories_dir)
