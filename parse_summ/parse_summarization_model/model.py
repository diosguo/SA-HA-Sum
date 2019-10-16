from mxnet import nd
from mxnet.gluon import Block
from mxnet.gluon import nn
from mxnet import nd
from stanfordcorenlp import StanfordCoreNLP
from parse_parse import parse


class ParseEncoder(Block):
    def __init__(self, tag_emb, word_emb):
        super(ParseEncoder, self).__init__()
        self.tag_embedding = tag_emb
        self.word_embedding = word_emb
        self.word_ass = nn.Dense(200)

    def emb_tree(self, root):
        next_emb = None
        if isinstance(root.next, str):
            # Leaf Node
            next_emb = self.word_embedding(root.next)
        elif isinstance(root.next, list):
            # Mid Node
            next_emb = self.emb_tree(root.next[0])
            for i in root.next[1:]:
                next_emb += self.emb_tree(i)
            next_emb /= len(root.next)
        else:
            # Wrong Node
            raise Exception('Error with Parse Tree Node.next type' + str(type(root.nexts)))

        return self.word_ass(nd.concat(self.tag_embedding(root.value), next_emb))

    def forward(self, root):
        pass


class BaseModel(object):
    def __init__(self):
        self.tag_embedding = nn.Embedding(10, 100)
        self.word_embedding = nn.Embedding(20000, 100)
        params = [self.tag_embedding, self.word_embedding]
        self.encoder = ParseEncoder(*params)


class Model(object):

    def __init__(self, stanford_path, mode, original_path, summary_path, model_params):
        nlp = StanfordCoreNLP(stanford_path)
        self.original_path = original_path
        self.summary_path = original_path
        self.model_params = model_params
        self.mode = mode


output = parse('(ROOT HELLO)')
print(output.val, output.next)
