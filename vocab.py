import numpy as np
import os

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


def generate_embedding_matrix(train_data, embed_fn='~/Desktop/glove.6B/glove.6B.300d.txt'):
    embed_dim = 300
    vocab = Vocab()
    data_tokens = set()
    col_toks = set()
    for ex in train_data:
        for tok in ex['question_toks']:
            data_tokens.add(tok.lower())
        for col in ex['schema']['column_names']:
            for col_split in col[1].split():
                data_tokens.add(col_split.lower())
                col_toks.add(col_split.lower())
    embeddings = open(os.path.expanduser(embed_fn), 'r').readlines()
    embed_matrix = list([[0.] * embed_dim])
    embed_matrix.append([0.] * embed_dim)
    for e in embeddings:
        esplit = e.split()
        token = esplit[0]
        if token in data_tokens:
            vidx = vocab.add_token_get_idx(token)
            assert vidx == len(embed_matrix)
            embed_matrix.append(list(map(float, esplit[1:])))
    embed_matrix[vocab.get_idx(UNK_TOKEN)] = list(np.random.normal(size=(embed_dim, )))
    vocab.set_random_idx()

    # Add non-GloVE column names as random embeddings
    for col_tok in col_toks - set(vocab.i2t):
        vidx = vocab.add_token_get_idx(col_tok)
        assert vidx == len(embed_matrix)
        embed_matrix.append(list(np.random.normal(size=(embed_dim, ))))
    return vocab, np.array(embed_matrix)
