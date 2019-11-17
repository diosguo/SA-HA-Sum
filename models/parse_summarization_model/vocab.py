class Vocab(object):
    PAD_TOKEN = '<pad>'
    UNKNOWN_TOKEN = '<unk>'
    SENTENCE_START = '<s>'
    SENTENCE_END = '</s>'
    DECODING_START = '<start>'
    DECODING_STOP = '<stop>'

    def __init__(self, vocab_path):
        self._word2id = {}
        for w in [Vocab.PAD_TOKEN, Vocab.UNKNOWN_TOKEN, Vocab.SENTENCE_START,
                  Vocab.SENTENCE_END, Vocab.DECODING_START, Vocab.DECODING_STOP]:
            self._word2id[w] = len(self._word2id) + 1

        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            for line in vocab_file:
                word, count = line.strip().split(' ', 1)
                self._word2id[word] = len(self._word2id) + 1
        self._id2word = {value: word for word, value in self._word2id.items()}
        self.size = len(self._word2id)

    def word2id(self, word):
        if word not in self._word2id:
            return 2
        else:
            return self._word2id[word]

    def id2word(self, ind):
        if ind in self._id2word:
            return self._id2word[ind]
        else:
            raise ValueError('Convert id to word error: id not in vocab')
