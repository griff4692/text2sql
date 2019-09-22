import argparse
import numpy as np
import torch
import torch.nn as nn

from load import link_schemas
from models.entity_linker import EntityLinker
from utils import tokens_to_ids
from vocab import generate_embedding_matrix


MAX_COL_TOK_LEN = 5  # Maximum # of words in each column for truncation (i.e. 'patient name id' has col_tok_len of 3)


def extract_labels(data):
    def _get_schema_col_idx(col_tok, schema):
        column_names_flat = list(map(lambda x: x[1], schema['column_names']))
        table, column = col_tok.split('::')
        # TODO: in case there is a db with the same col name across tables, disambiguate by using table name
        return column_names_flat.index(column)

    for example in data:
        col_tok_idxs = map(lambda x: _get_schema_col_idx(x, example['schema']),
                           filter(lambda x: '::' in x, example['query_toks_clean']))
        example['y'] = list(col_tok_idxs)


def generate_batch_idxs(data_len, batch_size, shuffle=False):
    batch_idxs = np.arange(data_len)
    if shuffle:
        np.random.shuffle(batch_idxs)
    num_batches = data_len // batch_size
    batches_by_idx = batch_idxs[:num_batches * batch_size].reshape(num_batches, batch_size)
    return batches_by_idx


def run_batch(X, y, batch_idxs):
    batch_X, batch_y = tensorize_batch(X, y, batch_idxs)
    output = model(batch_X)

    batch_y_flat = batch_y.view(-1)
    output_flat = output.view(-1)

    valid_idxs = (batch_y_flat >= 0).nonzero().squeeze(-1)
    batch_y_valid = torch.index_select(batch_y_flat, 0, valid_idxs).unsqueeze(-1)
    output_valid = torch.index_select(output_flat, 0, valid_idxs).unsqueeze(-1)

    return loss_func(output_valid, batch_y_valid.float())


def tensorize_batch(X, y, idxs):
    """
    :param X: question_ids, col_ids
    :param y: column indices indicating which columns in col_ids are used in the query for question_ids
    :param idxs: indices to extract for this batch --> len(batch_size)
    :return: (padded torch.question_ids, padded torch.col_ids, num question tokens for each question in batch,
    num cols for each question in batch), padded one-hot torch.y
    """
    question_ids, col_ids = X
    batch_qids, batch_cids, batch_y = [], [], []
    num_qs, num_cols = [], []
    max_q_toks = 0
    max_c_toks = 0
    for idx in idxs:
        batch_qids.append(question_ids[idx])
        batch_cids.append(col_ids[idx])
        batch_y.append(y[idx])
        assert len(col_ids[idx]) == len(y[idx])
        max_c_toks = max(max_c_toks, len(y[idx]))
        max_q_toks = max(max_q_toks, len(question_ids[idx]))
        num_qs.append(len(question_ids[idx]))
        num_cols.append(len(col_ids[idx]))

    for idx in range(len(idxs)):
        batch_qids[idx] += [0] * (max_q_toks - len(batch_qids[idx]))
        batch_cids[idx] += ([[0] * MAX_COL_TOK_LEN]) * (max_c_toks - len(batch_cids[idx]))
        batch_y[idx] += [-1] * (max_c_toks - len(batch_y[idx]))

    batch_qid_torch = torch.from_numpy(np.array(batch_qids))
    batch_cid_torch = torch.from_numpy(np.array(batch_cids))
    batch_y_torch = torch.from_numpy(np.array(batch_y))

    return (batch_qid_torch, batch_cid_torch, num_qs, num_cols), batch_y_torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entity Link Questions To Columns for Spider Text-2-SQL dataset.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--epochs', type=int, default=100, help='Training Epochs')
    args = parser.parse_args()

    train_data, dev_data = link_schemas(max_n=args.batch_size if args.debug else 9999999)
    print('Linked db schemas for {} training examples and {} dev examples.'.format(len(train_data), len(dev_data)))

    # Assign labels which are just the indices of column names used in the SQL query for a given question
    # i.e. SELECT COUNT(*) FROM people WHERE name = 'griffin'.  ['*', 'id', 'name'] --> y = [0, 2]
    extract_labels(train_data)
    extract_labels(dev_data)

    # Generate GloVe embedding matrix and instantiate vocabulary as intersection of dataset tokens and GloVe vectors
    # Make sure to include every column even if not in GloVE --> in this case, assign as random vector
    print('Generating embedding matrix and vocabulary.')
    vocab, embed_matrix = generate_embedding_matrix(train_data, dev_data)

    print('Converting tokens to ids.')
    train_x, train_y = tokens_to_ids(train_data, vocab, max_col_token_len=MAX_COL_TOK_LEN)
    dev_x, dev_y = tokens_to_ids(dev_data, vocab, max_col_token_len=MAX_COL_TOK_LEN)
    
    model = EntityLinker(embed_matrix=embed_matrix)

    # 10x fewer examples of positive case - a given column is in the query (i.e. most columns unused)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10.0]))

    trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=0.001)

    print('Starting Training...')
    for epoch_idx in range(args.epochs):
        model.train()

        train_batch_loss = 0.
        for batch_idx, batch_idx_set in enumerate(generate_batch_idxs(len(train_data), args.batch_size, shuffle=True)):
            optimizer.zero_grad()
            loss = run_batch(train_x, train_y, batch_idx_set)
            loss.backward()
            optimizer.step()
            train_batch_loss += loss.item()

            if epoch_idx == 0 and (batch_idx + 1) % 25 == 0:
                print('Epoch 1 Batch {} Running Avg Loss is {}'.format(
                    batch_idx + 1, train_batch_loss / float((batch_idx + 1))))
        print('Epoch {} Training Loss: {}'.format(epoch_idx + 1, train_batch_loss / float(batch_idx + 1)))

        if not args.debug:
            with torch.no_grad():
                model.eval()
                dev_batch_loss = 0.
                for batch_idx, batch_idx_set in enumerate(generate_batch_idxs(len(dev_data), args.batch_size)):
                    loss = run_batch(dev_x, dev_y, batch_idx_set)
                    dev_batch_loss += loss.item()
                print('Epoch {} Dev Loss: {}'.format(epoch_idx + 1, dev_batch_loss / float(batch_idx + 1)))
