class Vocab(object):
    def __init__(self, vocab_path):
        self.word2id = {'<pad>': 1, '<unk>': 2}

        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            for line in vocab_file:
                word, count = line.strip().split(' ', 1)
                self.word2id[word] = len(self.word2id) + 1
        self.id2word = {value: word for word, value in self.word2id.items()}
        self.size = len(self.word2id)
