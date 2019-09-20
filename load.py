import json
import os

from utils import clean_column_names, clean_table_names


def _augment_example(ex, db_id_to_schema):
    ex['schema'] = db_id_to_schema[ex['db_id']]
    ex['query_toks_clean'] = clean_table_names(ex['schema'], ex['query_toks'])
    ex['query_toks_clean'] = clean_column_names(ex['schema'], ex['query_toks_clean'])


def _get_schemas(schema_fn):
    schemas = json.load(open(schema_fn, 'r'))
    db_id_to_schema = {}
    for s in schemas:
        db_id_to_schema[s['db_id']] = s
    return db_id_to_schema


def link_schemas(data_dir='~/Desktop'):
    schema_fn = os.path.expanduser(os.path.join(data_dir, 'spider', 'tables.json'))
    db_id_to_schema = _get_schemas(schema_fn)

    train_data_fn = os.path.expanduser(os.path.join(data_dir, 'spider', 'train_spider.json'))
    train_data_other = os.path.expanduser(os.path.join(data_dir, 'spider', 'train_others.json'))
    train_data = json.load(open(train_data_fn, 'r')) + json.load(open(train_data_other, 'r'))
    dev_data = json.load(open(os.path.expanduser(os.path.join(data_dir, 'spider', 'dev.json')), 'r'))

    [_augment_example(ex, db_id_to_schema) for ex in train_data]
    [_augment_example(ex, db_id_to_schema) for ex in dev_data]

    return train_data, dev_data


if __name__ == '__main__':
    link_schemas()
