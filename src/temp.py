from stanfordcorenlp import StanfordCoreNLP
from mxnet import nd

nlp = StanfordCoreNLP(r'D:\ProgramData\stanford-corenlp-full-2018-10-05')

out = nlp.parse("I love china.")

w2v = {'I': [0.1, 0.2, 0.3], 'love': [0.2, 0.4, 0.1], 'china': [0.7, 0.1, 0.4]}

platform = 'win'
next_line = '\r\n' if platform == 'win' else '\n'

class TNode(object):

    def __init__(self):
        self.value = None
        self.next = []


def parse_dependency(dep: str):
    root = TNode()
    print(dep)
    print(dep.replace(next_line, ''))
    print(dep[1:-1].split(' ', 1))

parse_dependency(out)
