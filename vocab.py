import numpy as np
import os

EMBED_DIM = 300

PAD_TOKEN = 'PAD'
UNK_TOKEN = 'UNK'


class Vocab:
    def __init__(self):
        self.t2i = {}
        self.i2t = []

        self.add_token_get_idx(PAD_TOKEN)
        self.add_token_get_idx(UNK_TOKEN)
        self.random_idx = 2

    def add_token_get_idx(self, token):
        if token not in self.t2i:
            self.t2i[token] = len(self.t2i)
            self.i2t.append(token)
        return self.t2i[token]

    def get_idx(self, token):
        if token not in self.t2i:
            token = UNK_TOKEN
        return self.t2i[token]

    def get_token(self, idx):
        assert idx >= 0 and idx < len(self.i2t)
        return self.i2t[idx]

    def has_token(self, token):
        return not self.get_idx(token) == self.get_idx(UNK_TOKEN)

    def size(self):
        return len(self.i2t)

    def set_random_idx(self):
        """
        Every vocabulary instance with idx > self.random_idx is not in GloVE vocabulary. Assigned to random embedding.
        """
        self.random_idx = self.size()

    def tok_seq_to_ids(self, toks):
        return list(map(lambda x: self.get_idx(x.lower()), toks))


def _add_tokens(data, full_tok_set, col_tok_set):
    for ex in data:
        for tok in ex['question_toks']:
            full_tok_set.add(tok.lower())
    for col in ex['schema']['column_names']:
        for col_split in col[1].split():
            full_tok_set.add(col_split.lower())
            col_tok_set.add(col_split.lower())


def generate_embedding_matrix(train_data, dev_data, embed_fn='~/Desktop/glove.6B/glove.6B.300d.txt'):
    vocab = Vocab()
    data_tokens = set()
    col_toks = set()
    _add_tokens(train_data, data_tokens, col_toks)
    _add_tokens(dev_data, data_tokens, col_toks)

    embeddings = open(os.path.expanduser(embed_fn), 'r').readlines()
    embed_matrix = list([[0.] * EMBED_DIM])
    embed_matrix.append([0.] * EMBED_DIM)
    for e in embeddings:
        esplit = e.split()
        token = esplit[0]
        if token in data_tokens:
            vidx = vocab.add_token_get_idx(token)
            assert vidx == len(embed_matrix)
            embed_matrix.append(list(map(float, esplit[1:])))
    embed_matrix[vocab.get_idx(UNK_TOKEN)] = list(np.random.normal(size=(EMBED_DIM, )))
    vocab.set_random_idx()

    # Add non-GloVE column names as random embeddings
    for col_tok in col_toks - set(vocab.i2t):
        vidx = vocab.add_token_get_idx(col_tok)
        assert vidx == len(embed_matrix)
        embed_matrix.append(list(np.random.normal(size=(EMBED_DIM, ))))
    return vocab, np.array(embed_matrix)
