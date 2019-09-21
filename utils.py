from collections import defaultdict


def clean_column_names(schema, query_toks):
    # Get a map from alias to table name first
    alias_to_table = {}
    tables_aliased = []
    for i, tok in enumerate(query_toks):
        if tok == 'AS':
            tables_aliased.append(query_toks[i - 1])
            alias_to_table[query_toks[i + 1]] = query_toks[i - 1]
    tables_used = []
    for i, tok in enumerate(query_toks):
        if tok in ('FROM', 'JOIN'):
            tables_used.append(query_toks[i + 1])
    tables_used = list(set(tables_used + tables_aliased))

    column_to_table = defaultdict(list)
    for table_idx, column in schema['column_names']:
        if column == '*':
            column_to_table[column] += schema['table_names']
        else:
            column_to_table[column].append(schema['table_names'][table_idx])

    column_names = list(map(lambda x: x[1], schema['column_names']))
    column_names_original_lower = list(map(lambda x: x[1].lower(), schema['column_names_original']))
    return list(map(lambda column_name: _clean_column_name(column_name, column_names, column_names_original_lower,
                                                           alias_to_table, tables_used, column_to_table), query_toks))


def _clean_table_name(schema, candidate_table_name, table_names_original_lower):
    if candidate_table_name.lower() not in table_names_original_lower:
        return candidate_table_name
    return schema['table_names'][table_names_original_lower.index(candidate_table_name.lower())]


def clean_table_names(schema, query_toks):
    table_names_original_lower = list(map(lambda x: x.lower(), schema['table_names_original']))
    return list(map(lambda x: _clean_table_name(schema, x, table_names_original_lower), query_toks))


def _clean_column_name(candidate_column_name, column_names, column_names_original_lower, alias_to_table_map,
                       used_table_names, column_to_table):
    table_name, real_column_name = None, candidate_column_name
    alias_split = candidate_column_name.split('.')
    if len(alias_split) > 1 and alias_split[0] in alias_to_table_map.keys():
        table_name = alias_to_table_map[alias_split[0]]
        real_column_name = '.'.join(alias_split[1:])
    if real_column_name.lower() in column_names_original_lower:
        column_name = column_names[column_names_original_lower.index(real_column_name.lower())]
        if table_name is None:
            # TODO[disambiguate between table names if multiple and its ambiguous]
            table_names = list(set(used_table_names).intersection(set(column_to_table[column_name])))
            table_name = ','.join(table_names)
        return '{}::{}'.format(table_name, column_name)
    else:
        return candidate_column_name
