"""SpecID retention audit before training-data preparation."""
from pathlib import Path
import base64
import io
import json
import hashlib
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
    requested = [rule['field'][len('metadata:'):] for rule in [*config.get('view', {}).get('sortRules', []), *config.get('view', {}).get('filters', [])] if str(rule.get('field', '')).startswith('metadata:')]
    metadata_columns = list(dict.fromkeys([c for c in ['Kingdom','Superclass','Class','Subclass','DirectParent',column] if c in frame] + requested))
    if any(c not in frame for c in metadata_columns):
        raise ValueError('A selected classification column is missing from metadata.')
    grouped_metadata = {str(category): subset for category, subset in frame.groupby(column, sort=False)}
    group_metadata = []
    for index, (category, ids) in enumerate(groups):
        values = {} if index == 0 else {
            c: sorted(grouped_metadata[category][c].fillna('(Missing)').astype(str).unique().tolist())
            for c in metadata_columns}
        group_metadata.append({'scope':'overall' if index == 0 else 'category','category':category,'metadata':values})
    result = {'availableMetadataColumns': [str(c) for c in frame.columns if c != key],
              'metadataSource': {'metadata': str(source.resolve()), 'idColumn': key, 'groupColumn': column},
              'metadataFingerprint': metadata_fingerprint(frame, key, column), 'groups': group_metadata, 'metadataColumns': metadata_columns, 'rows': rows, 'diagnostics': diagnostics, 'sourceN': len(universe),
              'series': [{'name': n, 'color': c} for n, _, c in series],
              'title': config.get('title') or 'Training data filtering audit'}
    if output_dir:
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        (target/'results.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
        saved_config = {**config, 'metadata': str(source.resolve()), 'tools': [{**t, 'path': str(Path(t['path']).expanduser().resolve())} for t in tools], 'resultPath': 'results.json'}
        registry = json.loads((Path(__file__).resolve().parent/'file-formats.json').read_text(encoding='utf-8'))
        suffix = next(item['suffix'] for item in registry['formats'] if item['id'] == 'evaluation-trainable')
        settings_path = target/('settings'+suffix)
        settings_path.write_text(json.dumps({'schema':'fredis.evaluation.config','schemaVersion':1,'kind':'filtering','config':saved_config}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        result['outputDir'] = str(target)
        result['settingsPath'] = str(settings_path)
    return {**result, **present(result, config.get('view', {}))}


def present(result, view=None, output_dir=None, formats=(), output_file=None):
    """Render/export cached counts without reading metadata or SpecID lists."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output_file and (len(formats) != 1 or formats[0] not in {'png', 'svg'}):
        raise ValueError('A single output file requires PNG or SVG format.')
    view = view or {}
    from .filtering_view import select_view
    # List filters narrow the working list; graph membership is exclusively
    # controlled by category checkboxes (plus the independent Overall toggle).
    graph_view = {**view, 'filters': [], 'minN': 0}
    rows, category_rows, series, fields = select_view(result, graph_view)
    if not series:
        category_rows = []
    list_view = dict(view)
    list_view.pop('categories', None)
    list_tool_rows, list_rows, _, _ = select_view(result, list_view)
    metadata_fields = {rule.get('field') for rule in [*view.get('sortRules', []), *view.get('filters', [])]}
    visible_fields = [f for f in fields if (not f['id'].startswith('metadata:') or f['id'] in metadata_fields) and f.get('cached', True) and (not f['id'].startswith('tool:') or any(f['id'] == 'tool:'+s['name']+':'+metric for s in series for metric in ['retainedPercent','retainedN','excludedN']))]
    vertical = view.get('orientation', 'vertical') == 'vertical'
    def number(key, default, minimum, maximum):
        value = view.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum:
            raise ValueError(f'{key} must be between {minimum} and {maximum}.')
        return value

    group_width = number('groupWidth', 60, 1, 12000)
    group_padding = number('paddingWidth', 16, 0, 12000)
    group_count = len(category_rows)
    # Older settings store a total width; retain it until group sizing is chosen.
    width_px = view.get('width', 1200) if 'groupWidth' not in view else group_width*group_count + group_padding*(group_count+1)
    width_px = max(1, width_px)
    height_px = number('height', 700, 200, 12000)
    if isinstance(width_px, bool) or not isinstance(width_px, (int, float)) or not 1 <= width_px < 65536:
        raise ValueError('Calculated image width must be below 65536 px. Reduce group width, padding, or selected groups.')
    top_padding = number('topPadding', 80, 0, 2000) if 'groupWidth' in view and vertical else 0
    plot_height_px = height_px + top_padding
    title_size = number('titleSize', 14, 1, 96)
    category_size = number('categoryLabelSize', 10, 1, 96)
    tick_size = number('tickLabelSize', 10, 1, 96)
    axis_size = number('axisTitleSize', 11, 1, 96)
    value_size = number('valueLabelSize', 7, 1, 96)
    legend_size = number('legendSize', 10, 1, 96)
    fig, ax = plt.subplots(figsize=(width_px/150, plot_height_px/150), dpi=150)
    group_span = group_width/(group_width+group_padding) if 'groupWidth' in view else .8
    width = group_span/max(1,len(series))
    lookup = {(r['scope'], r['category'], r['tool']):r for r in rows}
    for j, item in enumerate(series):
        selected = [lookup[(g['scope'],g['category'],item['name'])] for g in category_rows]
        positions = [i-group_span/2+width/2+j*width for i in range(len(category_rows))]
        color = view.get('colors', {}).get(item['name'], item['color'])
        values = [r['retainedPercent'] or 0 for r in selected]
        bars = ax.bar(positions, values, width=width, label=item['name'], color=color) if vertical else ax.barh(positions, values, height=width, label=item['name'], color=color)
        if view.get('showValues', True):
            ax.bar_label(bars, labels=[f"{r['retainedN']}/{r['totalN']} ({(r['retainedPercent'] or 0):.1f}%)" for r in selected], fontsize=value_size, padding=3, rotation=90 if vertical else 0)
    labels = [(f"{g['category']} (N={g['totalN']})" if view.get('showCounts', True) else g['category'])
              if view.get('showCategoryLabels', True) else '' for g in category_rows]
    if vertical:
        ax.set_xticks(range(len(category_rows)), labels, rotation=60, ha='right')
        ax.set_ylim(0, 100*plot_height_px/height_px if 'groupWidth' in view else 140)
        if 'groupWidth' in view and group_count:
            edge = group_span/2 + group_padding/(group_width+group_padding)
            ax.set_xlim(-edge, group_count-1+edge)
        ax.set_ylabel('Retained spectra (%)' if view.get('showAxisTitle', True) else '', fontsize=axis_size)
    else:
        ax.set_yticks(range(len(category_rows)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100 if 'groupWidth' in view else 135)
        ax.set_xlabel('Retained spectra (%)' if view.get('showAxisTitle', True) else '', fontsize=axis_size)
    ax.tick_params(axis='x' if vertical else 'y', labelsize=category_size)
    ax.tick_params(axis='y' if vertical else 'x', labelsize=tick_size,
                   labelleft=view.get('showTickLabels', True) if vertical else True,
                   labelbottom=True if vertical else view.get('showTickLabels', True))
    if view.get('showTitle', True):
        ax.set_title(view.get('title') or result.get('title', 'Training data filtering audit'), fontsize=title_size)
    if series and view.get('showLegend', True):
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=legend_size)
    if 'groupWidth' in view:
        # Size the axes themselves in pixels. Add only the space actually needed
        # by tick labels, titles, bar annotations and the legend.
        import math
        ax.set_position([0, 0, 1, 1])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        plot_box = ax.get_window_extent(renderer)
        content_box = ax.get_tightbbox(renderer)
        left = math.ceil(max(0, plot_box.x0-content_box.x0)) + 8
        right = math.ceil(max(0, content_box.x1-plot_box.x1)) + 8
        bottom = math.ceil(max(0, plot_box.y0-content_box.y0)) + 8
        top = math.ceil(max(0, content_box.y1-plot_box.y1)) + 8
        canvas_width = math.ceil(width_px+left+right)
        canvas_height = math.ceil(plot_height_px+bottom+top)
        if max(canvas_width, canvas_height) >= 65536:
            plt.close(fig)
            raise ValueError('Calculated image width or height including labels must be below 65536 px.')
        fig.set_size_inches(canvas_width/150, canvas_height/150)
        ax.set_position([left/canvas_width, bottom/canvas_height,
                         width_px/canvas_width, plot_height_px/canvas_height])
    else:
        fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150)
    if output_dir:
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        if 'tsv' in formats:
            pd.DataFrame(rows, columns=['scope','category','tool','totalN','retainedN','excludedN','retainedPercent']).to_csv(target/'retention.tsv', sep='\t', index=False)
            pd.DataFrame([{k: ' | '.join(v) if isinstance(v, list) else v for k,v in g.items() if k == 'scope' or k in {f['id'] for f in visible_fields}} for g in category_rows], columns=['scope']+[f['id'] for f in visible_fields]).to_csv(target/'categories.tsv', sep='\t', index=False)
            pd.DataFrame(result['diagnostics']).to_csv(target/'id_diagnostics.tsv', sep='\t', index=False)
        if 'png' in formats:
            (target/'retention.png').write_bytes(buffer.getvalue())
        if 'svg' in formats:
            fig.savefig(target/'retention.svg', format='svg')
    if output_file:
        target = Path(output_file).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if formats[0] == 'png':
            target.write_bytes(buffer.getvalue())
        else:
            fig.savefig(target, format='svg')
    plt.close(fig)
    return {'displayRows': rows, 'categoryRows': category_rows, 'listRows': list_rows, 'listToolRows': list_tool_rows, 'fields': fields, 'png': base64.b64encode(buffer.getvalue()).decode('ascii')}


def metadata_fingerprint(frame, key, column):
    pairs = sorted(zip(frame[key].astype(str).str.strip(), frame[column].fillna('(Missing)').astype(str)))
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode('utf-8')).hexdigest()


def enrich_metadata(result, view, metadata_config=None):
    """Load only newly requested sort/filter columns, without recalculating retention."""
    wanted = list(dict.fromkeys(rule['field'][len('metadata:'):] for rule in
        [*view.get('sortRules', []), *view.get('filters', [])]
        if str(rule.get('field', '')).startswith('metadata:')))
    missing = [c for c in wanted if c not in result.get('metadataColumns', [])]
    if not missing:
        return result
    source = metadata_config or result.get('metadataSource')
    if not source or not source.get('metadata'):
        raise ValueError('Load the metadata source to use this sort/filter column.')
    path = Path(source['metadata']).expanduser()
    frame = (pd.read_csv(path, sep='\t' if path.suffix.lower() == '.tsv' else ',', dtype=str)
             if path.suffix.lower() in {'.csv','.tsv'} else _plain_table(path)).copy()
    key, column = source.get('idColumn') or 'SpecID', source.get('groupColumn')
    if any(c not in frame for c in [key, column, *missing]):
        raise ValueError('A requested metadata column is missing from the metadata source.')
    if frame[key].isna().any():
        raise ValueError('Metadata contains missing SpecIDs.')
    frame[key] = frame[key].astype(str).str.strip()
    frame[column] = frame[column].fillna('(Missing)').astype(str)
    if frame[key].eq('').any() or frame[key].duplicated().any():
        raise ValueError('Metadata SpecIDs must be nonempty and unique.')
    fingerprint = metadata_fingerprint(frame, key, column)
    if result.get('metadataFingerprint') and fingerprint != result['metadataFingerprint']:
        raise ValueError('Metadata SpecIDs or categories have changed. Evaluate again before using these columns.')
    expected = {r['category']:r['totalN'] for r in result['rows'] if r['scope']=='category'}
    grouped = {str(category):subset for category,subset in frame.groupby(column, sort=False)}
    if len(frame) != result['sourceN'] or {name:len(f) for name,f in grouped.items()} != expected:
        raise ValueError('Metadata categories do not match the saved evaluation. Evaluate again.')
    old = {(g['scope'],g['category']):g.get('metadata',{}) for g in result.get('groups', [])}
    groups = [{'scope':'overall','category':'All','metadata':old.get(('overall','All'),{})}]
    for name, subset in grouped.items():
        values = dict(old.get(('category',name),{}))
        values.update({c:sorted(subset[c].fillna('(Missing)').astype(str).unique().tolist()) for c in missing})
        groups.append({'scope':'category','category':name,'metadata':values})
    return {**result, 'groups':groups, 'metadataColumns':list(dict.fromkeys([*result.get('metadataColumns',[]),*missing])),
            'availableMetadataColumns':[str(c) for c in frame.columns if c!=key],
            'metadataSource':{**source,'metadata':str(path.resolve())},'metadataFingerprint':fingerprint}
