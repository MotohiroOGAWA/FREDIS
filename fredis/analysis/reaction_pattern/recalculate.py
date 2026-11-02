"""Rebuild coverage summaries from saved peak annotations, without chemistry."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .comparison import ConditionOutcome, compare_compound
from .compounds import CompoundGroup, SampleRecord
from .constants import assigned_column, count_column
from .hdf5_storage import ResultWriter, TABLE_SCHEMAS
from .summary import summarize
from .tree import PatternRegistry, TreeConfiguration, resolve_tree


def recalculate_coverage(source, output: Path, formula_only: bool, progress=None):
    """Write a temporary summary result; read only A/B, never original sources or C."""
    config = source.configuration()
    tree = resolve_tree(TreeConfiguration.from_dict(config['tree']), PatternRegistry([
        {'pattern_id': row['pattern_id'], 'definition': json.loads(row['definition_json'])}
        for row in source.table_records('patterns')
    ]))
    # Stored ids are the identity of the annotation columns.
    if [(n.node_id, n.condition_id) for n in tree.nodes] != [
        (n['node_id'], n['condition_id']) for n in source.nodes()
    ]:
        raise ValueError('Saved condition ids do not match the tree configuration.')
    conditions = defaultdict(list)
    for row in source.table_records('compound_condition'):
        conditions[row['compound_id']].append(row)
    applicable = defaultdict(dict)
    for row in source.table_records('compound_node_summary'):
        applicable[row['compound_id']][tree.nodes[row['node_row']].node_id] = row['applicable']
    compounds = source.table_records('compounds')
    comparisons = []
    for number, compound in enumerate(compounds):
        if progress:
            progress({'phase': 'coverage', 'message': f'Recalculating coverage: {number + 1}/{len(compounds)} compounds',
                      'percent': 100 * number / max(1, len(compounds))})
        path = source.root / compound['peak_summary_file']
        frame = pd.read_parquet(path)
        source.loaded_files.append(str(path))
        if formula_only and len(frame) and 'formula' not in frame:
            raise ValueError('This result has no saved peak formulas. Run formula assignment first.')
        group = CompoundGroup(compound['compound_id'], compound['canonical_smiles'], compound['original_smiles'])
        parts = []
        for sample_index, rows in frame.groupby('sample_index', sort=True):
            rows = rows.sort_values('peak_index')
            first = rows.iloc[0]
            group.samples.append(SampleRecord(
                dataset_id=int(first.dataset_id), record_index=int(first.record_index),
                smiles=str(first.smiles), adduct=str(first.adduct), collision_energy=float(first.collision_energy),
                precursor_mz=float(first.precursor_mz), mz=rows.mz.to_numpy(), intensity=rows.intensity.to_numpy(),
                relative_intensity=rows.relative_intensity.to_numpy(), sample_id=int(first.sample_id),
                sample_index=int(sample_index), is_precursor=rows.is_precursor.to_numpy(dtype=bool),
                formula_status=str(first.get('formula_status', '')),
                precursor_formula=str(first.get('precursor_formula', '')),
                formulas=(rows.formula.fillna('').tolist()
                          if 'formula' in rows and first.get('formula_status') == 'assigned' else None),
                intensity_coverage_formula_only=formula_only,
            ))
            parts.append(rows)
        positions = {sample.sample_index: index for index, sample in enumerate(group.samples)}
        outcomes = {}
        for row in conditions[group.compound_id]:
            cid = row['condition_id']
            outcomes[cid] = ConditionOutcome(
                status=row['status'], skip_category=row['skip_category'], reason=row['reason'], error=row['error'],
                kept=tuple(positions[int(index)] for index in row['kept'].split(',') if index and int(index) in positions),
                assigned=[rows[assigned_column(cid)].to_numpy(dtype=bool) for rows in parts],
                counts=[rows[count_column(cid)].to_numpy(dtype=np.int32) for rows in parts],
                file=row['detail_path'], seed_file=row['seed_path'],
                alias_of=None if row['alias_of'] < 0 else row['alias_of'],
            )
        comparisons.append(compare_compound(group, tree.nodes, outcomes, applicable[group.compound_id]))
    datasets = [{key: value for key, value in row.items()
                 if key not in ('compound_count', 'sample_count', 'peak_count')}
                for row in source.table_records('datasets')]
    tables, overall = summarize(comparisons, tree=tree, datasets=datasets)
    original_overall = source.overall_summary()
    overall = {**original_overall, **overall}
    writer = ResultWriter(output)
    try:
        writer.write_json('configuration', config)
        writer.write_overall_summary(overall)
        for name in TABLE_SCHEMAS:
            writer.write_table(name, tables.get(name))
        writer.write_indexes(len(comparisons), tables.tables)
        writer.finalize()
    except BaseException:
        writer.abort()
        raise
