import os
import json
from parse_summ.parse_summarization_model.vocab import Vocab
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


def get_abs(story_file):
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
    return abstract


if __name__ == '__main__':
    config = json.load(open("../config.json", 'r'))
    vocab = Vocab(config['vocab_path'])
    vocab_tag = Vocab(config['vocab_tag_path'])
    file_list = os.listdir('D:\Projects\cnn_stories_tokenized')
    for file in tqdm(file_list):
        abstract = get_abs(os.path.join('D:\Projects\cnn_stories_tokenized', file))
        abstract = list(map(int, abstract.split(' ')))
        pickle.dump(abstract, open(os.path.join('D:\Projects\cnn_summaries', file), 'wb'))
