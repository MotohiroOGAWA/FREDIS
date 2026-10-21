"""Shared category selection and ordering for tables, figures, and exports."""
from functools import cmp_to_key
import math
import statistics


def fields_for(result):
    fields = [{'id': 'category', 'label': 'Category', 'type': 'text'},
              {'id': 'totalN', 'label': 'Original N', 'type': 'number'}]
    fields += [{'id': 'metadata:'+name, 'label': 'Metadata · '+name, 'type': 'text', 'cached': name in result.get('metadataColumns', [])} for name in result.get('availableMetadataColumns', result.get('metadataColumns', []))]
    for series in result['series']:
        for metric, label in [('retainedPercent', 'Retention (%)'), ('retainedN', 'Retained N'), ('excludedN', 'Excluded N')]:
            fields.append({'id': 'tool:'+series['name']+':'+metric, 'label': series['name']+' · '+label, 'type': 'number'})
    fields += [{'id': 'stats:'+name, 'label': 'All tools · '+name+' retention (%)', 'type': 'number'} for name in ['min', 'max', 'mean', 'median']]
    return fields


def select_view(result, view):
    fields = fields_for(result)
    types = {f['id']: f['type'] for f in fields}
    groups = {}
    lookup = {}
    metadata = {(g['scope'], g['category']): g.get('metadata', {}) for g in result.get('groups', [])}
    real_tools = {s['name'] for s in result['series'] if s['name'] != 'Common'}
    for row in result['rows']:
        key = (row['scope'], row['category'])
        group = groups.setdefault(key, {'scope': row['scope'], 'category': row['category'], 'totalN': row['totalN']})
        lookup[(*key, row['tool'])] = row
        for metric in ['retainedPercent', 'retainedN', 'excludedN']:
            group['tool:'+row['tool']+':'+metric] = row[metric]
    for key, group in groups.items():
        for column in result.get('metadataColumns', []):
            group['metadata:'+column] = metadata.get(key, {}).get(column, [])
        values = [group['tool:'+name+':retainedPercent'] for name in real_tools
                  if group.get('tool:'+name+':retainedPercent') is not None]
        for name, fn in [('min', min), ('max', max), ('mean', statistics.mean), ('median', statistics.median)]:
            group['stats:'+name] = fn(values) if values else None
    def text(value):
        return ' | '.join(str(v) for v in value) if isinstance(value, list) else str(value)
    def matches(group, rule):
        field = rule.get('field')
        if field not in types:
            raise ValueError('Unknown filter field: '+str(field)+'. Load metadata columns first.')
        actual = group.get(field)
        expected = rule.get('value', '')
        op = rule.get('op', '>=')
        if op not in ['>=', '<=', '>', '<', '==', '!=', 'contains', 'notContains']:
            raise ValueError('Unknown filter operator: '+str(op))
        if types[field] == 'number':
            try:
                expected = float(expected)
            except (TypeError, ValueError) as error:
                raise ValueError('Numeric filter requires a number.') from error
            if not math.isfinite(expected):
                raise ValueError('Numeric filter requires a finite number.')
            if actual is None:
                return False
        else:
            actual, expected = text(actual).casefold(), str(expected).casefold()
        if op == 'contains': return str(expected) in str(actual)
        if op == 'notContains': return str(expected) not in str(actual)
        if op == '==': return actual == expected
        if op == '!=': return actual != expected
        if op == '>=': return actual >= expected
        if op == '<=': return actual <= expected
        if op == '>': return actual > expected
        return actual < expected
    selected = [g for g in groups.values()
                if (view.get('showOverall', True) or g['scope'] != 'overall')
                and ('categories' not in view or g['scope'] == 'overall' or g['category'] in view['categories'])
                and g['totalN'] >= int(view.get('minN', 0))
                and all(matches(g, rule) for rule in view.get('filters', []))]
    sorts = view.get('sortRules')
    if sorts is None:
        legacy = view.get('sort', 'name')
        sorts = [{'field': {'name':'category','count':'totalN','retention':'stats:min'}.get(legacy,'category'),
                  'direction':'desc' if legacy == 'count' else 'asc'}]
    for rule in sorts:
        if rule.get('field') not in types or rule.get('direction', 'asc') not in ['asc', 'desc']:
            raise ValueError('Invalid sort rule: '+str(rule))
    def compare(a, b):
        # Overall is a reference row, always placed before categories when shown.
        if (a['scope'] == 'overall') != (b['scope'] == 'overall'):
            return -1 if a['scope'] == 'overall' else 1
        for rule in sorts:
            field = rule['field']
            left, right = a.get(field), b.get(field)
            left_missing, right_missing = left is None or left == [], right is None or right == []
            if left_missing != right_missing: return 1 if left_missing else -1
            if left_missing: continue
            if types[field] == 'text': left, right = text(left).casefold(), text(right).casefold()
            compared = (left > right)-(left < right)
            if compared: return -compared if rule.get('direction') == 'desc' else compared
        return (a['category'] > b['category'])-(a['category'] < b['category'])
    selected.sort(key=cmp_to_key(compare))
    series = [s for s in result['series'] if 'tools' not in view or s['name'] in view['tools']]
    rows = [lookup[(g['scope'],g['category'],s['name'])] for g in selected for s in series
            if (g['scope'],g['category'],s['name']) in lookup]
    return rows, selected, series, fields
