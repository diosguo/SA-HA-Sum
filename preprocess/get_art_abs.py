import os
import sys
import json
sys.path.append('../')
from models.vocab import Vocab
import pickle
from tqdm import tqdm



# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence


def read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    # print line[-1]
    return line + " ."


def word2id(line):
    words = line.split(' ')
    ids = ' '.join(
        list(
            map(
                str,
                [vocab.word2id(x) for x in words]
            )
        )
    )
    return ids


def get_art_abs(story_file):
    lines = read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them
    # (this is a problem in the dataset because many image captions don't
    # end in periods; consequently they end up in the body of the article
    # as run-on sentences)

    # lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(word2id(line))
        else:
            article_lines.append(word2id(line))

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = ' '.join(["%s %s %s" % (vocab.word2id('<s>'), sent, vocab.word2id('</s>')) for sent in highlights])
    abstract = '%s %s %s' % (vocab.word2id(vocab.DECODING_START), abstract, vocab.word2id(vocab.DECODING_STOP))
    
    article = ' '.join(["%s %s %s" % (vocab.word2id('<s>'), sent, vocab.word2id('</s>')) for sent in article_lines])
    article = '%s %s %s' % (vocab.word2id(vocab.DECODING_START), article, vocab.word2id(vocab.DECODING_STOP))
    
    return article, abstract


if __name__ == '__main__':
    vocab = Vocab('../data/vocab')
    tokenized_path = '/home/xuyang/data/tokenized/cnn_stories_tokenized'
    article_path = '/home/xuyang/data/articles'
    abstract_path = '/home/xuyang/data/abstracts'
    file_list = os.listdir(tokenized_path)

    for file in tqdm(file_list, ncols=10):
        articles_line, abstracts_line = get_art_abs(os.path.join(tokenized_path, file))
        
        abstracts = list(map(int, abstracts_line.split()))
        try:
            articles = list(map(int, articles_line.split()))
        except:
            print('#'+articles_line+'#')
            break
        pickle.dump(articles, open(os.path.join(article_path, file),'wb'))
        pickle.dump(abstracts, open(os.path.join(abstract_path, file), 'wb'))
