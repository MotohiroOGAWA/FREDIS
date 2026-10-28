"""Reconcile search progress and construct checkpoint-safe continuation commands."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def plan_signature(trials):
    return hashlib.sha256(json.dumps([t['args'] for t in trials], separators=(',', ':')).encode()).hexdigest()


def read_progress(output, plan, result_name, plan_name):
    """Check identity before trusting a completed row or touching trial outputs."""
    report_file = output / result_name
    if not report_file.exists():
        report_file = next((output / name for name in ('fragment-tree.tuning.pft', 'fragment-tree.tuning-result.pft') if (output / name).exists()), report_file)
    report = json.loads(report_file.read_text()) if report_file.exists() else {}
    saved_file = next((output / name for name in (plan_name, 'tuning-plan.json') if (output / name).exists()), None)
    signature = plan_signature(plan['trials'])
    if saved_file:
        saved = json.loads(saved_file.read_text())
        if 'trials' not in saved:
            from .fragment_tree import expand_search
            saved = expand_search(saved)
        if plan_signature(saved['trials']) != signature:
            raise ValueError('Existing tuning plan differs. Restore the original plan or choose a new output directory.')
    if report.get('planSignature', signature) != signature:
        raise ValueError('Existing tuning results belong to a different plan. Choose a new output directory.')
    records = {}
    for record in report.get('trials', []):
        index = int(record['trial'])
        if index in records or not 1 <= index <= len(plan['trials']):
            raise ValueError('Invalid or duplicate trial in existing tuning results')
        if record.get('args') != plan['trials'][index - 1]['args']:
            raise ValueError(f'Trial {index} arguments differ from the existing results')
        records[index] = record
    scores = {}
    score_file = output / 'tuning-scores.tsv'
    if score_file.exists():
        with score_file.open() as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                index = int(row['trial'])
                if index in scores or not 1 <= index <= len(plan['trials']):
                    raise ValueError('Invalid or duplicate trial in tuning-scores.tsv')
                config = json.loads(row['config'])
                if config != plan['trials'][index - 1]['config']:
                    raise ValueError(f'Trial {index} configuration differs from tuning-scores.tsv')
                scores[index] = row
    if not saved_file and not report and (scores or any(Path(t['config']['outputDir']).exists() for t in plan['trials'])):
        raise ValueError('Existing trial outputs have no saved tuning plan; refusing to overwrite them')
    return report, records, scores


def completed(record, row, directory):
    """Presence alone is insufficient: running/failed rows must be retried."""
    if row and row.get('status') == 'completed' and str(row.get('exitCode')) == '0':
        return True
    if record and record.get('status') == 'completed' and record.get('exitCode') == 0:
        return True
    file = directory / 'result.training.pft'
    if file.exists():
        result = json.loads(file.read_text())
        return result.get('status') == 'completed' and result.get('completedEpochs', 0) > 0
    return False


def continuation(trial, directory):
    """Keep common runtime flags; inherit architecture and training state from disk.

    The ordinary training CLI interprets --epochs as *additional* epochs on
    resume. Subtract the saved epoch to preserve the search plan's total budget.
    """
    if not directory.exists() or not any(directory.iterdir()):
        return trial['args'], None, False
    from ...cli.base import namespace_argv
    from ..training.fragment_tree_training.shared_training import architecture_destinations, build_arg_parser as shared_parser
    from ..training.fragment_tree_training.training import build_arg_parser as detailed_parser
    import torch

    parser = shared_parser() if trial['args'][3] == 'fragment-tree' else detailed_parser()
    try:
        args = parser.parse_args(trial['args'][4:])
    except SystemExit as error:
        raise ValueError('Cannot parse the saved trial arguments for continuation') from error
    checkpoints = [directory / 'last.pt', *sorted((directory / 'checkpoints').glob('epoch_*.pt'), reverse=True)]
    checkpoint = None
    errors = []
    for file in checkpoints:
        if not file.exists():
            continue
        try:
            saved = torch.load(file, map_location='cpu', weights_only=False)
            for key in ('model_config', 'model_state_dict', 'optimizer_state_dict', 'epoch', 'training_settings'):
                if key not in saved:
                    raise ValueError('missing ' + key)
            epoch = saved['epoch']
            if type(epoch) is not int or epoch < 0 or epoch > args.epochs:
                raise ValueError('checkpoint epoch is outside the planned budget')
            checkpoint = (file, saved)
            break
        except Exception as error:
            errors.append(f'{file.name}: {error}')
    if checkpoint is None:
        if directory.exists() and any(directory.iterdir()):
            detail = '; '.join(errors) if errors else 'no last.pt or periodic checkpoint'
            raise ValueError('Cannot resume existing trial without a valid checkpoint: ' + detail)
        return trial['args'], None, False
    file, saved = checkpoint
    remaining = args.epochs - saved['epoch']
    patience = args.early_stopping_patience
    stopped = patience > 0 and saved.get('bad_epochs', 0) >= patience
    if remaining == 0 or stopped:
        # A kill can land after the final checkpoint and before the final report.
        # Repair the live report without performing an unwanted additional epoch.
        from ..common.paths import resolve_paths
        from ..training.fragment_tree_training.report import write_training_report
        report_file = directory / 'result.training.pft'
        previous = json.loads(report_file.read_text()) if report_file.exists() else {}
        write_training_report(
            directory, status='completed',
            settings=resolve_paths(saved['training_settings'], file.parent.resolve()),
            datasets=resolve_paths(previous.get('datasets', {}), directory.resolve()),
            completed_epochs=saved['epoch'], global_step=saved.get('global_step', 0),
            latest_validation=f"spectrum_validation/epoch_{saved['epoch']}.json",
        )
        return None, {'checkpoint': str(file), 'epoch': saved['epoch'], 'remainingEpochs': 0}, True
    excluded = architecture_destinations() | {'overwrite', 'overwrite_confirmed', 'validate_before_training'}
    runtime = argparse.Namespace(**{k: v for k, v in vars(args).items() if k not in excluded})
    runtime.resume = str(file.resolve())
    runtime.epochs = remaining
    argv = ['-m', 'fredis.cli', 'train', 'fragment-tree-detailed', *namespace_argv(detailed_parser(), runtime)]
    return argv, {'checkpoint': str(file), 'epoch': saved['epoch'], 'remainingEpochs': remaining}, False


def score_summaries(row):
    result = {}
    for key, value in (row or {}).items():
        metric, _, statistic = key.rpartition('.')
        if not metric.startswith('validation') or statistic not in ('max', 'min', 'last') or not value:
            continue
        number = float(value)
        if math.isfinite(number):
            result.setdefault(metric, {})[statistic] = number
    return result
