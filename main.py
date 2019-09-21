import argparse
import numpy as np
import torch
import torch.nn as nn

from load import link_schemas
from vocab import generate_embedding_matrix
from models.entity_linker import EntityLinker

MAX_COL_TOK_LEN = 5  # Maximum # of words in each column for truncation (i.e. 'patient name id' has col_tok_len of 3)
BATCH_SIZE = 32


def _get_schema_col_idx(col_tok, schema):
    colum_names_flat = list(map(lambda x: x[1], schema['column_names']))
    table, column = col_tok.split('::')
    # TODO in case there is a db with the same col name across tables, disambiguate by using table name
    return colum_names_flat.index(column)


def tensorize_batch(X, y, idxs):
    question_ids, col_ids = X
    batch_qids, batch_cids, batch_y = [], [], []
    max_q_toks = 0
    max_c_toks = 0
    for idx in idxs:
        batch_qids.append(question_ids[idx])
        batch_cids.append(col_ids[idx])
        batch_y.append(y[idx])
        assert len(col_ids[idx]) == len(y[idx])
        max_c_toks = max(max_c_toks, len(y[idx]))
        max_q_toks = max(max_q_toks, len(question_ids[idx]))

    for idx in range(len(idxs)):
        batch_qids[idx] += [0] * (max_q_toks - len(batch_qids[idx]))
        batch_cids[idx] += ([[0] * MAX_COL_TOK_LEN]) * (max_c_toks - len(batch_cids[idx]))
        batch_y[idx] += [-1] * (max_c_toks - len(batch_y[idx]))

    batch_qid_torch = torch.from_numpy(np.array(batch_qids))
    batch_cid_torch = torch.from_numpy(np.array(batch_cids))
    batch_y_torch = torch.from_numpy(np.array(batch_y))

    return (batch_qid_torch, batch_cid_torch), batch_y_torch


def extract_labels(example):
    example['y'] = [_get_schema_col_idx(query_tok, example['schema'])
                    for query_tok in example['query_toks_clean'] if '::' in query_tok]


def tokens_to_ids(dataset, vocab):
    question_ids, col_ids, ys_onehot = [], [], []
    for ex in dataset:
        question_id_seq = vocab.tok_seq_to_ids(ex['question_toks'])
        col_id_seqs = []
        y_onehot = [0] * len(ex['schema']['column_names'])
        for column in ex['schema']['column_names']:
            col_id_seq = vocab.tok_seq_to_ids(column[1].split())
            if len(col_id_seq) > MAX_COL_TOK_LEN:
                col_id_seq = col_id_seq[:MAX_COL_TOK_LEN]
            else:
                col_id_seq += [0] * (MAX_COL_TOK_LEN - len(col_id_seq))  # Zero-pad
            col_id_seqs.append(col_id_seq)
        question_ids.append(question_id_seq)
        col_ids.append(col_id_seqs)
        for y_idx in ex['y']:
            y_onehot[y_idx] = 1
        ys_onehot.append(y_onehot)
    return (question_ids, col_ids), ys_onehot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entity Link Questions To Columns for Spider Text-2-SQL dataset.')
    parser.add_argument('--epochs', type=int, default=100, help='Training Epochs')
    parser.add_argument('-debug', default=True, action='store_true')
    args = parser.parse_args()

    train_data, dev_data = link_schemas(max_n=32 if args.debug else 9999999)
    print('Linked db schemas for {} training examples and {} dev examples.'.format(len(train_data), len(dev_data)))

    [extract_labels(ex) for ex in train_data]
    [extract_labels(ex) for ex in dev_data]

    # Generate GloVe embedding matrix and instantiate vocabulary as intersection of dataset tokens and GloVe vectors
    # Make sure to include every column even if not in GloVE --> in this case, assign as random vector
    vocab, embed_matrix = generate_embedding_matrix(train_data)

    # convert to token ids for 'query_toks_clean' and 'column_names'
    train_x, train_y = tokens_to_ids(train_data, vocab)
    dev_x, dev_y = tokens_to_ids(dev_data, vocab)
    
    model = EntityLinker(embed_matrix=embed_matrix)

    # arange training sequence
    train_batch_idxs = np.arange(len(train_y))

    # pos_weight = torch.Tensor([10.0])
    loss_func = nn.BCEWithLogitsLoss()  # pos_weight=pos_weight)

    trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=0.001)

    for epoch_idx in range(args.epochs):
        model.train()

        # Randomize batches for epoch
        # np.random.shuffle(train_batch_idxs)
        num_batches = len(train_y) // BATCH_SIZE
        batches_by_idx = train_batch_idxs[:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE)

        batch_loss = 0.
        for batch_idx, batch_idx_set in enumerate(batches_by_idx):
            optimizer.zero_grad()
            X, y = tensorize_batch(train_x, train_y, batch_idx_set)
            output = model(X)

            y_flat = y.view(-1)
            output_flat = output.view(-1)

            valid_idxs = (y_flat >= 0).nonzero().squeeze(-1)
            y_valid = torch.index_select(y_flat, 0, valid_idxs).unsqueeze(-1)
            output_valid = torch.index_select(output_flat, 0, valid_idxs).unsqueeze(-1)
            loss = loss_func(output_valid, y_valid.float())
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
        print(batch_loss / float(num_batches))
