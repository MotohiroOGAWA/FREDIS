"""Auditable batch prediction artifacts using the official MS formats."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from msentity_similarity import load


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def valid_spectra(dataset, id_column):
    """Keep peaks untouched, recording original rows and rejecting ambiguous IDs."""
    ids = dataset.to_pandas()[id_column].map(lambda value: '' if pd.isna(value) else str(value).strip())
    if ids[ids != ''].duplicated().any():
        raise ValueError(f'Duplicate spectrum identifiers in {id_column}')
    selected, rejected = [], []
    for row, identifier in enumerate(ids):
        spectrum = dataset[row]
        mz, intensity = np.asarray(spectrum.mz), np.asarray(spectrum.intensity)
        reasons = []
        if not identifier: reasons.append('missing_id')
        if not len(mz): reasons.append('empty_spectrum')
        if not np.isfinite(mz).all() or not np.isfinite(intensity).all(): reasons.append('nonfinite_peaks')
        if (mz <= 0).any(): reasons.append('nonpositive_mz')
        if (intensity < 0).any(): reasons.append('negative_intensity')
        if not (intensity > 0).any(): reasons.append('no_positive_intensity')
        if reasons:
            rejected.append({'SpecID': identifier, 'SourceRow': row, 'Status': 'excluded', 'Reason': ';'.join(reasons)})
        else:
            selected.append(row)
    result = dataset.select(selected)
    column = 'ComparisonSourceRow'
    while column in result.columns: column += '_'
    result[column] = selected
    key = 'ComparisonSpecID'
    while key in result.columns: key += '_'
    result[key] = ids.iloc[selected].tolist()
    return result, rejected, {'row_column': column, 'id_column': key}


def native_fingerprints():
    fingerprints = {}
    for name in ('msentity', 'msentity-similarity'):
        distribution = importlib.metadata.distribution(name)
        fingerprints[name] = {
            str(file): sha256(distribution.locate_file(file))
            for file in distribution.files or ()
            if str(file).endswith(('.so', '.pyd', '.dll', '.dylib'))
            and distribution.locate_file(file).is_file()
        }
    return fingerprints


def finalize(directory, similarity, observed, predicted, excluded, mapping, settings, input_hash, expected_table=None):
    """Reload and verify official output before reporting success."""
    directory = Path(directory)
    path = directory / 'predicted.mssim'
    restored = load(str(path))
    frame = similarity.to_pandas()
    pd.testing.assert_frame_equal(restored.to_pandas()[frame.columns], frame)
    frame = restored.to_pandas()
    if expected_table is not None:
        pd.testing.assert_frame_equal(frame, expected_table)
    if restored.metadata != similarity.metadata:
        raise ValueError('MSSIM metadata changed during save')
    if restored.has_matched_data:
        # Check every embedded spectrum, including its metadata and original peaks.
        for side, source in ((1, observed), (2, predicted)):
            embedded = restored.matched_datasets[side - 1]
            indices = list(restored.source_indices(side))
            pd.testing.assert_frame_equal(embedded.to_pandas().reset_index(drop=True),
                                          source.to_pandas().iloc[indices].reset_index(drop=True))
            for local, original in enumerate(indices):
                np.testing.assert_array_equal(embedded[local].mz, source[original].mz)
                np.testing.assert_array_equal(embedded[local].intensity, source[original].intensity)
    records = frame.copy()
    records['Status'] = 'compared'
    records['Reason'] = ''
    for side, source in ((1, observed), (2, predicted)):
        meta = source.to_pandas()
        info = mapping[str(side)]
        records[f'SourceRow{side}'] = [meta.iloc[int(i)][info['row_column']] for i in frame[f'index{side}']]
        records[f'SpecID{side}'] = [meta.iloc[int(i)][info['id_column']] for i in frame[f'index{side}']]
    predicted_metadata = predicted.to_pandas()
    records['GeneratedSpecID'] = [predicted_metadata.iloc[int(i)]['SpecID'] for i in frame['index2']]
    records['Score'] = frame[restored.score_columns[0]]
    records = pd.concat([records, pd.DataFrame(excluded)], ignore_index=True)
    records.to_csv(directory / 'records.tsv', sep='\t', index=False, na_rep='')
    statistics = {}
    for method in restored.score_columns:
        scores = frame[method]
        statistics[method] = {'count': int(scores.count()), 'missing': int(scores.isna().sum()),
                              'mean': float(scores.mean()) if scores.count() else None,
                              'median': float(scores.median()) if scores.count() else None,
                              'min': float(scores.min()) if scores.count() else None,
                              'max': float(scores.max()) if scores.count() else None}
    plot_columns = [name for name in frame.columns if name in restored.score_columns or name.startswith('intensity_coverage') or name.startswith('modified_intensity_coverage') or name == 'theoretical_fragment_peak_intensity_coverage']
    plot_rows = frame[plot_columns]
    # Keep the result page responsive; the complete table is in MSSIM / TSV.
    if len(plot_rows) > 20000:
        plot_rows = plot_rows.iloc[np.linspace(0, len(plot_rows)-1, 20000, dtype=int)]
    report = {'schema': 'fredis.prediction-result', 'settings': settings,
              'comparison_count': len(frame), 'excluded_count': len(excluded),
              'score_statistics': statistics, 'methods': list(restored.score_columns),
              'observed_side': 1, 'similarity_file': 'predicted.mssim' if len(frame) else None,
              'plot_rows': json.loads(plot_rows.to_json(orient='records'))}
    (directory / 'result.prediction.pft').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    source = settings.get('input')
    if source and sha256(source) != input_hash:
        raise ValueError('Input file changed during prediction')
    if not len(frame):
        path.unlink(missing_ok=True)
    outputs = {str(p.relative_to(directory)): sha256(p) for p in directory.rglob('*')
               if p.is_file() and p.name not in ('summary.json', 'manifest.json') and 'tmp' not in p.relative_to(directory).parts}
    summary = {'comparison_count': len(frame), 'input_count': len(observed) + sum(r['Side'] == 1 and r['Reason'] != 'no_matching_spectrum' for r in excluded),
               'predicted_count': len(predicted) + sum(r['Side'] == 2 and r['Reason'] != 'no_matching_spectrum' for r in excluded), 'excluded_count': len(excluded),
               'exclusion_reasons': dict(Counter(reason for row in excluded for reason in row['Reason'].split(';'))),
               'primary_method': restored.score_columns[0], 'score_statistics': statistics,
               'settings': settings, 'output_sha256': outputs}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    outputs = {**outputs, 'summary.json': sha256(directory / 'summary.json')}
    manifest = {'status': 'complete' if len(frame) else 'no_comparable_pairs', 'input': source, 'input_sha256': input_hash,
                'model_sha256': sha256(settings['modelPath']), 'native_sha256': native_fingerprints(),
                'libraries': {name: importlib.metadata.version(name) for name in ('msentity', 'msentity-similarity')},
                'code_sha256': {name: sha256(Path(__file__).with_name(name)) for name in ('prediction_exports.py', 'predict_spectrum.py')}, 'settings': settings,
                'observed_side': 1, 'predicted_side': 2, 'output_sha256': outputs}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
