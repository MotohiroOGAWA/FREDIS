"""Check, snapshot and export canonical molecule splits without generating structures."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile

from fredis.cli.chemistry_options import configure_chemistry_options, resolve_chemistry_options

import msentity as ms
from .context import create_preparation_context
from .dataset_plan import DEFAULT_MAPPING, SPLITS, normalize_entries, plan_datasets
from .datasets import canonical_keys, load_spectrum_dataset

SCHEMA = 'fredis.checked-dataset-split'


def fingerprint(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtimeNs=stat.st_mtime_ns, device=stat.st_dev, inode=stat.st_ino)


def validate_peaks(data):
    import numpy as np
    peaks = data
    if np.any(np.diff(data.offsets) == 0):
        raise ValueError('Some spectra have no peaks.')
    if not np.isfinite(peaks.mz).all() or np.any(peaks.mz <= 0):
        raise ValueError('Peak m/z values must be finite and positive.')
    if not np.isfinite(peaks.intensity).all() or np.any(peaks.intensity < 0):
        raise ValueError('Peak intensities must be finite and non-negative.')


def check_split(config, progress=None):
    chemistry = dict(fragmenter_params=config['fragmenterParams'], symbols=config.get('symbols'))
    if config.get('reactionModels'):
        from fredis.ml.training.fragment_tree_training.pattern_models import prepare_pattern_models
        chemistry = prepare_pattern_models(chemistry, config['reactionModels'])
    context = create_preparation_context(chemistry)
    entries = normalize_entries(config['datasets'])
    before = {entry['id']: fingerprint(entry['path']) for entry in entries}
    seed = config.get('validationSeed', 0)
    if type(seed) is not int:
        raise ValueError('Split seed must be an integer.')
    plan = plan_datasets(entries, config.get('datasetMode', 'automatic'), context.fragmenter,
                         chemistry['symbols'], float(config.get('validationRatio') or 0),
                         float(config.get('testRatio') or 0), seed, invalid_detail_limit=5000,
                         progress=progress,
                         validation_max_train_ratio=config.get("validationMaxTrainRatio"),
                         test_max_train_ratio=config.get("testMaxTrainRatio"))
    validate_peaks(plan['combined'])
    if before != {entry['id']: fingerprint(entry['path']) for entry in entries}:
        raise ValueError('Input files changed while checking. Run Check Split Summary again.')
    snapshot = dict(schema=SCHEMA, version=1, mode=config.get('datasetMode', 'automatic'),
                    seed=seed, validationRatio=config.get('validationRatio') or 0,
                    testRatio=config.get('testRatio') or 0,
                    validationMaxTrainRatio=config.get('validationMaxTrainRatio'),
                    testMaxTrainRatio=config.get('testMaxTrainRatio'), entries=plan['entries'],
                    fingerprints=before, validIndexes=plan['valid_indexes'],
                    assignments=plan['assignments'], splits=plan['summary'], sources=plan['sources'])
    return plan, snapshot


def write_snapshot(snapshot, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A failed check never exposes a partial checked plan.
    with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(snapshot, handle, allow_nan=False)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def restore_checked(snapshot, progress=None):
    import numpy as np
    if snapshot.get('schema') != SCHEMA or snapshot.get('version') != 1:
        raise ValueError('Unsupported checked dataset split.')
    entries = normalize_entries(snapshot['entries'])
    datasets = []
    for index, entry in enumerate(entries):
        def unchanged():
            if fingerprint(entry['path']) != snapshot['fingerprints'][entry['id']]:
                raise ValueError(f"{entry['path']}: input changed since checking. Run Check Split Summary again.")
        unchanged()
        if progress:
            progress(dict(stage='loading', percent=index * 65 / len(entries),
                          datasetIndex=index + 1, datasetsTotal=len(entries), path=entry['path']))
        data = load_spectrum_dataset(entry['path'])
        unchanged()
        data = data[snapshot['validIndexes'][entry['id']]].copy()
        columns = {DEFAULT_MAPPING[key]: data[column].copy() for key, column in entry['mapping'].items()}
        for column, values in columns.items():
            data[column] = values
        data['_fredis_dataset_path'] = entry['path']
        data['_fredis_source_record'] = snapshot['validIndexes'][entry['id']]
        datasets.append(data)
    combined = ms.concat(datasets) if len(datasets) > 1 else datasets[0]
    keys = canonical_keys(combined['SMILES'], progress=(lambda current, total: progress(dict(
        stage='restoring', percent=65 + 10 * current / max(1, total), current=current, total=total))) if progress else None)
    labels = np.array([snapshot['assignments'][key] for key in keys])
    splits = {name: combined[np.flatnonzero(labels == name)] for name in SPLITS}
    actual = {name: dict(records=len(data), molecules=len(set(key for key, label in zip(keys, labels) if label == name)))
              for name, data in splits.items()}
    if actual != snapshot['splits']:
        raise ValueError('Checked split counts no longer match. Run Check Split Summary again.')
    return splits


def filtered_input_jobs(snapshot, progress=None):
    """Select source rows without renaming columns or adding preparation metadata."""
    import numpy as np

    entries = normalize_entries(snapshot['entries'])
    jobs, counts, molecules = [], dict.fromkeys(SPLITS, 0), {name: set() for name in SPLITS}
    for index, entry in enumerate(entries):
        if fingerprint(entry['path']) != snapshot['fingerprints'][entry['id']]:
            raise ValueError('Input changed since checking. Run Check Split Summary again.')
        data = load_spectrum_dataset(entry['path'])
        if fingerprint(entry['path']) != snapshot['fingerprints'][entry['id']]:
            raise ValueError('Input changed since checking. Run Check Split Summary again.')
        rows = snapshot['validIndexes'][entry['id']]
        structures = data[entry['mapping']['smilesColumn']]
        keys = canonical_keys([structures[row] for row in rows])
        labels = np.array([snapshot['assignments'][key] for key in keys])
        for name in SPLITS:
            selected = np.flatnonzero(labels == name)
            if not len(selected):
                continue
            original_rows = [rows[i] for i in selected]
            filename = f"{name}/{index + 1:03d}_{Path(entry['path']).stem}.msds"
            jobs.append((filename, data[original_rows].copy()))
            counts[name] += len(selected)
            molecules[name].update(keys[i] for i in selected)
        if progress:
            progress(dict(stage='restoring', percent=75 * (index + 1) / len(entries)))
    actual = {name: dict(records=counts[name], molecules=len(molecules[name])) for name in SPLITS}
    if actual != snapshot['splits']:
        raise ValueError('Checked split counts no longer match. Run Check Split Summary again.')
    return jobs


def export_checked(snapshot, output_dir, progress=None, splits=None, preserve_input_metadata=False):
    output = Path(output_dir).resolve()
    if preserve_input_metadata:
        jobs = filtered_input_jobs(snapshot, progress)
    else:
        splits = restore_checked(snapshot, progress) if splits is None else splits
        jobs = [(name + '.msds', data) for name, data in splits.items() if len(data)]
    reserved = [output / name for name, _ in jobs] + [output / 'dataset_split_manifest.json']
    if not preserve_input_metadata:
        reserved.extend(output / (name + '.msds') for name in SPLITS)
    if any(file.exists() for file in reserved):
        raise FileExistsError('Output already contains split datasets. Choose another directory.')
    output.mkdir(parents=True, exist_ok=True)
    written = []
    # Stage every file before publishing and never overwrite existing user data.
    with tempfile.TemporaryDirectory(prefix='.fredis-split-', dir=output) as directory:
        staging = Path(directory)
        for index, (name, data) in enumerate(jobs):
            if progress:
                progress(dict(stage='writing', split=name, percent=75 + 24 * index / len(jobs),
                              current=index, total=len(jobs)))
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            data.save(str(staging / name))
        manifest = {key: value for key, value in snapshot.items() if key != 'validIndexes'}
        manifest['exportMode'] = 'preserve-input-metadata' if preserve_input_metadata else 'combined'
        (staging / 'dataset_split_manifest.json').write_text(json.dumps(manifest, indent=2))
        try:
            for name in [name for name, _ in jobs] + ['dataset_split_manifest.json']:
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(staging / name, target)
                written.append(target)
        except BaseException:
            for file in written:
                file.unlink(missing_ok=True)
            raise
    if progress:
        progress(dict(stage='complete', percent=100))
    return dict(outputDir=str(output), files={(name if preserve_input_metadata else Path(name).stem): str(output / name) for name, _ in jobs},
                splits=snapshot['splits'])


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--datasets-json', help='JSON list of input dataset paths, mappings and roles.')
    inputs.add_argument('--config-stdin', action='store_true', help='Read Workbench dataset configuration JSON from stdin.')
    inputs.add_argument('--checked-plan', help='Export a previously checked split, without repeating validation.')
    parser.add_argument('--dataset-mode', choices=('automatic', 'per-dataset', 'assigned'), default='automatic')
    parser.add_argument("--validation-max-train-ratio", type=float, help="Assigned mode: maximum validation/train molecule ratio; omitted means unlimited.")
    parser.add_argument("--test-max-train-ratio", type=float, help="Assigned mode: maximum test/train molecule ratio; omitted means unlimited.")
    parser.add_argument('--validation-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plan-output', help='Save a checked plan for subsequent export.')
    parser.add_argument('--output-dir', help='Write non-empty train.msds / validation.msds / test.msds here.')
    parser.add_argument('--preserve-input-metadata', action='store_true',
                        help='Export each input separately per split, filtering rows only and preserving original metadata and peaks.')
    configure_chemistry_options(parser)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    def progress(event):
        print('FREDIS_PROGRESS ' + json.dumps(event), file=sys.stderr, flush=True)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.checked_plan:
                if not args.output_dir:
                    raise ValueError('--checked-plan requires --output-dir.')
                snapshot = json.loads(Path(args.checked_plan).read_text())
                result = export_checked(snapshot, args.output_dir, progress, preserve_input_metadata=args.preserve_input_metadata)
            else:
                if args.config_stdin:
                    config = json.load(sys.stdin)
                else:
                    chemistry = resolve_chemistry_options(args)
                    config = dict(datasets=json.loads(args.datasets_json), datasetMode=args.dataset_mode,
                                  validationRatio=args.validation_ratio, testRatio=args.test_ratio, validationSeed=args.seed,
                                  validationMaxTrainRatio=args.validation_max_train_ratio, testMaxTrainRatio=args.test_max_train_ratio,
                                  fragmenterParams=chemistry['fragmenter_params'], symbols=chemistry['symbols'])
                plan, snapshot = check_split(config, lambda event: progress({**event, 'percent':event['percent'] * 0.7 if args.output_dir else event['percent'] * 0.95}))
                if args.plan_output:
                    progress(dict(stage='saving-plan', percent=96 if not args.output_dir else 72))
                    write_snapshot(snapshot, args.plan_output)
                result = dict(valid=True, splits=plan['summary'], sources=plan['sources'])
                if args.plan_output:
                    result['checkedPlan'] = str(Path(args.plan_output).resolve())
                if args.output_dir:
                    result.update(export_checked(snapshot, args.output_dir, progress, splits=plan['splits'], preserve_input_metadata=args.preserve_input_metadata))
                else:
                    progress(dict(stage='complete', percent=100))
        print(json.dumps(dict(ok=True, result=result)), flush=True)
    except Exception as error:
        print(json.dumps(dict(ok=False, error=str(error))), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
