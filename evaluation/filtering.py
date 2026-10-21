"""SpecID retention audit before training-data preparation."""
from pathlib import Path
import base64
import io
import json
import pandas as pd
from .io import _plain_table


def inspect_metadata(config):
    frame = _plain_table(Path(config['metadata']).expanduser())
    return {'columns': [str(c) for c in frame.columns], 'rows': len(frame)}


def evaluate(config, output_dir=None):
    from matplotlib.colors import is_color_like

    source = Path(config['metadata']).expanduser()
    frame = (pd.read_csv(source, sep='\t' if source.suffix.lower() == '.tsv' else ',', dtype=str)
             if source.suffix.lower() in {'.csv', '.tsv'} else _plain_table(source)).copy()
    key = config.get('idColumn') or 'SpecID'
    column = config.get('groupColumn')
    if key not in frame or column not in frame:
        raise ValueError('Select valid SpecID and classification columns.')
    if frame[key].isna().any():
        raise ValueError('Metadata contains missing SpecIDs.')
    frame[key] = frame[key].astype(str).str.strip()
    if frame[key].eq('').any() or frame[key].duplicated().any():
        raise ValueError('Metadata SpecIDs must be nonempty and unique.')
    frame[column] = frame[column].fillna('(Missing)').astype(str)
    tools = config.get('tools', [])
    if not tools:
        raise ValueError('Add at least one tool.')
    names = [str(t.get('name', '')).strip() for t in tools]
    if any(not n for n in names) or len(set(names)) != len(names) or 'Common' in names:
        raise ValueError('Tool names must be unique, nonempty, and different from Common.')
    universe = set(frame[key])
    series = []
    diagnostics = []
    for name, tool in zip(names, tools):
        color = tool.get('color', '#36c5a2')
        if not is_color_like(color):
            raise ValueError(f'Invalid color for {name}.')
        lines = [line.strip() for line in Path(tool['path']).expanduser().read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        ids = set(lines)
        series.append((name, ids & universe, color))
        diagnostics.append({'tool': name, 'listedN': len(lines), 'uniqueN': len(ids), 'duplicateN': len(lines)-len(ids), 'matchedN': len(ids & universe), 'unknownN': len(ids-universe)})
    common = set.intersection(*(s[1] for s in series))
    series.insert(0, ('Common', common, '#666666'))
    groups = [('All', universe)] + [(str(category), set(group[key])) for category, group in frame.groupby(column, sort=True)]
    rows = []
    for index, (category, ids) in enumerate(groups):
        for name, kept, _ in series:
            n = len(ids & kept)
            rows.append({'scope': 'overall' if index == 0 else 'category', 'category': category, 'tool': name, 'totalN': len(ids), 'retainedN': n, 'excludedN': len(ids)-n, 'retainedPercent': 100*n/len(ids) if ids else None})
    result = {'rows': rows, 'diagnostics': diagnostics, 'sourceN': len(universe),
              'series': [{'name': n, 'color': c} for n, _, c in series],
              'title': config.get('title') or 'Training data filtering audit'}
    if output_dir:
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        (target/'results.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
        saved_config = {**config, 'metadata': str(source.resolve()), 'tools': [{**t, 'path': str(Path(t['path']).expanduser().resolve())} for t in tools], 'resultPath': 'results.json'}
        (target/'settings.evaltrainable.pft').write_text(json.dumps({'schema':'fredis.evaluation.config','schemaVersion':1,'kind':'filtering','config':saved_config}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        result['outputDir'] = str(target)
        result['settingsPath'] = str(target/'settings.evaltrainable.pft')
    return {**result, **present(result, config.get('view', {}))}


def present(result, view=None, output_dir=None, formats=()):
    """Render/export cached counts without reading metadata or SpecID lists."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    view = view or {}
    rows = list(result['rows'])
    if 'categories' in view:
        rows = [r for r in rows if r['scope'] == 'overall' or r['category'] in view['categories']]
    if view.get('showOverall', True) is False:
        rows = [r for r in rows if r['scope'] != 'overall']
    if 'tools' in view:
        rows = [r for r in rows if r['tool'] in view['tools']]
    rows = [r for r in rows if r['totalN'] >= int(view.get('minN', 0))]
    groups = list(dict.fromkeys((r['scope'], r['category']) for r in rows))
    mode = view.get('sort', 'name')
    def group_key(group):
        subset = [r for r in rows if (r['scope'], r['category']) == group]
        value = group[1] if mode == 'name' else (-subset[0]['totalN'] if mode == 'count' else min(r['retainedPercent'] or 0 for r in subset))
        return (group[0] != 'overall', value, group[1])
    groups.sort(key=group_key)
    series = [s for s in result['series'] if any(r['tool'] == s['name'] for r in rows)]
    lookup = {(r['scope'], r['category'], r['tool']): r for r in rows}
    rows = [lookup[(*g, s['name'])] for g in groups for s in series]
    fig, ax = plt.subplots(figsize=(12, max(4, len(groups)*max(1,len(series))*.18)))
    width = .8/max(1,len(series))
    for j, item in enumerate(series):
        selected = [lookup[(*g, item['name'])] for g in groups]
        positions = [i-.4+width/2+j*width for i in range(len(groups))]
        bars = ax.barh(positions, [r['retainedPercent'] or 0 for r in selected], height=width, label=item['name'], color=view.get('colors', {}).get(item['name'], item['color']))
        ax.bar_label(bars, labels=[f"{r['retainedN']}/{r['totalN']} ({(r['retainedPercent'] or 0):.1f}%)" for r in selected], fontsize=7, padding=3)
    ax.set_yticks(range(len(groups)), [f"{g[1]} (N={lookup[(*g,series[0]['name'])]['totalN']})" for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xlabel('Retained spectra (%) — denominator: metadata SpecIDs in category')
    ax.set_title(view.get('title') or result.get('title', 'Training data filtering audit'))
    if series:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    if output_dir:
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        if 'tsv' in formats:
            pd.DataFrame(rows, columns=['scope','category','tool','totalN','retainedN','excludedN','retainedPercent']).to_csv(target/'retention.tsv', sep='\t', index=False)
            pd.DataFrame(result['diagnostics']).to_csv(target/'id_diagnostics.tsv', sep='\t', index=False)
        if 'png' in formats:
            (target/'retention.png').write_bytes(buffer.getvalue())
        if 'svg' in formats:
            fig.savefig(target/'retention.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    return {'displayRows': rows, 'png': base64.b64encode(buffer.getvalue()).decode('ascii')}
