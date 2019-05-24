import Vocab

def article2ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word2id(Vocab.SPECIAL_TOKEN[3])

    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)

    return ids, oovs


def abstract2ids(abstract_words, vocab, oovs):
    ids = []
    unk_id = vocab.word2id(Vocab.SPECIAL_TOKEN[3])

    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if w in oovs:
                vocab_idx = vocab.size() + oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)

    return ids


def abstract2sents(abstract):
    cur = 0
    sents = []

    while True:
        try:
            start_p = abstract.index(Vocab.SPECIAL_TOKEN[0], cur)
            end_p = abstract.index(Vocab.SPECIAL_TOKEN[1], start_p+1)

            cur = end_p + len(Vocab.SPECIAL_TOKEN[1])
            sents.append(abstract[start_p+len(Vocab.SPECIAL_TOKEN[0]):end_p])
        except ValueError:
            return sents
