
SPECIAL_TOKEN = ['<s>','</s>','[PAD]','[UNK]','[START]','[STOP]']

class Vocab(object):
    def __init__(self, vocab_path, vocab_size):
        """

        :param vocab_path:
            vocab file format: one word per line, 'token times'
        :param vocab_size:
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        for t in SPECIAL_TOKEN[2:]:
            self._word_to_id[t] = self._count
            self._id_to_word[self._count] = t
            self._count += 1

        with open(vocab_path,'r',encoding='utf-8') as vocab_f:
            for line in vocab_f:
                pieces = line.split()

                if len(pieces) != 2:
                    print('Warnning: incorrectly formatted in %s',line)
                w = pieces[0]
                if w in SPECIAL_TOKEN:
                    raise Exception('vocab can\'t contains special tokens')
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabluary')

                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1

                if vocab_size > 0 and self._count >= vocab_size:
                    print('vocab size is :', vocab_size)
                    break

        print('Finished construct vocab of %d words'%self._count)




