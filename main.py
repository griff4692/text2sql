from load import link_schemas


def _get_schema_col_idx(col_tok, schema):
    colum_names_flat = list(map(lambda x: x[1], schema['column_names']))
    table, column = col_tok.split('::')
    # TODO in case there is a db with the same col name across tables, disambiguate by using table name
    return colum_names_flat.index(column)


def extract_labels(example):
    example['y'] = [_get_schema_col_idx(query_tok, example['schema'])
                    for query_tok in example['query_toks_clean'] if '::' in query_tok]


if __name__ == '__main__':
    train_data, dev_data = link_schemas()
    print('Linked db schemas for {} training examples and {} dev examples.'.format(len(train_data), len(dev_data)))

    [extract_labels(ex) for ex in train_data]
    [extract_labels(ex) for ex in dev_data]

    print('Illustrative example of data for train and dev...')
    print(train_data[0]['query_toks_clean'], train_data[0]['y'])
    print(dev_data[0]['query_toks_clean'], dev_data[0]['y'])
