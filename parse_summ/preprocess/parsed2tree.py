from parse_summ.parse_summarization_model.vocab import Vocab
import json
import pickle
import os


class TNode(object):
    def __init__(self):
        self.next = []
        self.val = None

    def __str__(self):
        return '{' + str(self.val) + ' ' + str(self.next) + '}'

    def __repr__(self):
        return self.__str__()


# sub part count
def count_sub(sub: str):
    p2i = {'(': 1, ')': -1}
    count = 0
    split_ind = []
    num_of_left = 0
    for k, i in enumerate(sub):
        if i in p2i:
            num_of_left += p2i[i]
            if num_of_left == 0:
                count += 1
                split_ind.append(k + 1)
    return count, split_ind


def parse(dep: str):
    root = TNode()
    t = [x.strip() for x in dep[1:-1].split(' ', 1)]
    root.val = vocab_tag.word2id(t[0])
    sub_num, sub_split_ind = count_sub(t[1])
    pre = 0
    if sub_num == 0:
        root.next = vocab.word2id(t[1].strip())
    else:
        for i in range(sub_num):
            root.next.append(parse(t[1][pre:sub_split_ind[i]].strip()))
            pre = sub_split_ind[i]
    return root


def main(parsed_path, save_path):
    files = os.listdir(parsed_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file in files:
        doc = []
        sentences = pickle.load(open(os.path.join(parsed_path, file), 'rb'))
        for sent in sentences:
            doc.append(parse(sent))
        print(doc)
        pickle.dump(doc, open(os.path.join(save_path, file), 'wb'))


if __name__ == '__main__':
    config = json.load(open('../config.json', 'r'))
    vocab = Vocab(config['vocab_path'])
    vocab_tag = Vocab(config['vocab_tag_path'])
    main('../cnn_parsed', '../cnn_tree')
