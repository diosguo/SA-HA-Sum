from mxnet import nd
from mxnet.gluon import Block
from stanfordcorenlp import StanfordCoreNLP


class Model(object):

    def __init__(self, stanford_path, mode, original_path, summary_path, model_params):

        nlp = StanfordCoreNLP(stanford_path)
        self.original_path = original_path
        self.summary_path = original_path
        self.model_params = model_params
        self.mode = mode
