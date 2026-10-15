"""Run normal training CLI processes and retain all validation observations."""
import csv
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

RESULT_NAME = 'fragment-tree.tuning.pft'


def validation_history(directory):
    file = Path(directory) / 'metrics.tsv'
    if not file.exists():
        return []
    history = []
    with file.open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            if row.get('event') != 'epoch_end':
                continue
            numeric = {}
            for key, value in row.items():
                try:
                    number = float(value)
                    if math.isfinite(number):
                        numeric[key] = number
                except (TypeError, ValueError):
                    pass
            history.append(numeric)
    # Distribution quartiles are measured across validation spectra at each epoch,
    # rather than across epoch means. Merge only observations of completed epochs.
    distributions = Path(directory) / 'metric_distributions.tsv'
    if distributions.exists():
        indexed = {(row.get('epoch'), row.get('global_step')): row for row in history}
        with distributions.open() as stream:
            for item in csv.DictReader(stream, delimiter='\t'):
                if item.get('split') != 'validation':
                    continue
                try:
                    row = indexed.get((float(item['epoch']), float(item['global_step'])))
                    value = float(item['value'])
                except (ValueError, KeyError):
                    continue
                if row is not None and math.isfinite(value):
                    row['validation_' + item['metric']] = value
    return history


def summarize_trial(trial):
    """Retain every comparison column, independent of any ranking preference."""
    history = trial.get('validationHistory', [])
    metrics = sorted({key for row in history for key in row if key.startswith('validation')})
    result = {}
    for metric in metrics:
        observations = [(row.get('epoch'), row[metric]) for row in history
                        if isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric])]
        if not observations:
            continue
        maximum = max(observations, key=lambda pair: pair[1])
        summary = {'max': maximum[1], 'maxEpoch': maximum[0]}
        # Last means the final completed epoch, never the last non-missing value.
        summary.update(last=history[-1].get(metric), lastEpoch=history[-1].get('epoch'))
        if 'loss' in metric:
            minimum = min(observations, key=lambda pair: pair[1])
            summary.update(min=minimum[1], minEpoch=minimum[0])
        result[metric] = summary
    return result


def expand_search(spec):
    """Public CLI search format: normal CLI arguments plus flag candidate lists."""
    base = spec['baseArgs']
    search = spec['parameters']
    if not isinstance(base, list) or not all(isinstance(v, str) for v in base):
        raise ValueError('baseArgs must be a list of ordinary training CLI arguments')
    if not search or any(not key.startswith('--') or not isinstance(values, list) or not values
                         for key, values in search.items()):
        raise ValueError('parameters must map CLI flags to non-empty candidate lists')
    trials = []
    for index, values in enumerate(itertools.product(*search.values()), 1):
        args = list(base)
        parameters = dict(zip(search, values))
        for flag, value in parameters.items():
            if flag in ('--output-dir', '--resume', '--overwrite'):
                raise ValueError('Output, resume and overwrite cannot be tuned')
            if flag in args:
                pos = args.index(flag)
                args[pos + 1] = str(value)
            else:
                args.extend([flag, str(value)])
        directory = str(Path(spec['outputDir']) / f'trial-{index:03d}')
        if '--output-dir' in args:
            args[args.index('--output-dir') + 1] = directory
        else:
            args.extend(['--output-dir', directory])
        trials.append({'config': {'outputDir': directory, 'parameters': parameters},
                       'args': ['-m', 'fredis.cli', 'train', 'fragment-tree', *args]})
    return {**spec, 'trials': trials}


def run(plan):
    if 'trials' not in plan:
        plan = expand_search(plan)
    output = Path(plan['outputDir'])
    trials = plan['trials']
    if not trials:
        raise ValueError('Supply at least one trial')
    directories = []
    for trial in trials:
        directory = Path(trial['config']['outputDir']).resolve()
        args = trial['args']
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args) or args[:4] != ['-m', 'fredis.cli', 'train', 'fragment-tree']:
            raise ValueError('Trials must invoke the normal fragment-tree training CLI')
        if '--output-dir' not in args or Path(args[args.index('--output-dir') + 1]).resolve() != directory:
            raise ValueError('Trial output argument does not match its configuration')
        if '--resume' in args or '--overwrite' in args:
            raise ValueError('Tuning trials must use fresh output directories')
        if not directory.is_relative_to(output.resolve()) or directory == output.resolve():
            raise ValueError('Trial outputs must be inside the search output directory')
        if directory.exists() or directory in directories:
            raise ValueError('Trial output already exists or is duplicated: ' + str(directory))
        directories.append(directory)
    output.mkdir(parents=True, exist_ok=True)
    report_file = output / RESULT_NAME
    if report_file.exists() or (output / 'tuning-results.json').exists():
        raise ValueError('Tuning results already exist. Choose a new output directory.')
    results = []
    report = {'schema': 'fredis.tuning-result', 'version': 2, 'status': 'running',
              'createdAt': datetime.now(timezone.utc).isoformat(), 'searchMethod': 'grid',
              'totalTrials': len(trials), 'trials': results}

    def publish():
        for trial in results:
            trial['scoreSummaries'] = summarize_trial(trial)
        report.update(updatedAt=datetime.now(timezone.utc).isoformat())
        columns = sorted({f'{metric}.{stat}' for trial in results
                          for metric, summary in trial['scoreSummaries'].items()
                          for stat in summary if stat in ('max', 'last', 'min')})
        report['scoreColumns'] = columns
        table = output / 'tuning-scores.tsv.tmp'
        with table.open('w') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['trial', 'status', 'exitCode', 'config', *columns])
            for trial in results:
                values = {f'{metric}.{stat}': value
                          for metric, summary in trial['scoreSummaries'].items()
                          for stat, value in summary.items()}
                writer.writerow([trial['trial'], trial['status'], trial.get('exitCode', ''),
                                 json.dumps(trial['config']), *[values.get(c) for c in columns]])
        table.replace(output / 'tuning-scores.tsv')
        temp = report_file.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temp.replace(report_file)
    publish()
    (output / 'tuning-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    try:
        for index, trial in enumerate(trials, 1):
            directory = Path(trial['config']['outputDir'])
            record = {'trial': index, 'outputDir': str(directory.relative_to(output)),
                      'status': 'running', 'config': trial['config'], 'args': trial['args'],
                      'startedAt': datetime.now(timezone.utc).isoformat(), 'validationHistory': []}
            results.append(record)
            publish()
            print(f'Tuning trial {index}/{len(trials)}: {directory}', flush=True)
            try:
                record['exitCode'] = subprocess.call([sys.executable, *trial['args']])
                record['status'] = 'completed' if record['exitCode'] == 0 else 'failed'
            except OSError as error:
                record.update(exitCode=1, status='failed', error=str(error))
            record.update(validationHistory=validation_history(directory), finishedAt=datetime.now(timezone.utc).isoformat())
            publish()
            print(f'Trial {index}: {record["status"]}', flush=True)
    except BaseException:
        if results and results[-1]['status'] == 'running':
            results[-1].update(status='interrupted', validationHistory=validation_history(trials[len(results)-1]['config']['outputDir']))
        report['status'] = 'interrupted'
        publish()
        raise
    report['status'] = 'completed' if all(r['exitCode'] == 0 for r in results) else 'failed'
    publish()
    print(f'Tuning results: {report_file}', flush=True)
    return 0 if report['status'] == 'completed' else 1
