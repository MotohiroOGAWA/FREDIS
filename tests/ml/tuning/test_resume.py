"""Recovery must reconcile completed scores, missing rows, and checkpoint state."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from fredis.ml.tuning.fragment_tree import PLAN_NAME, RESULT_NAME, expand_search, run
from fredis.ml.tuning.resume import continuation


def plan(root, count=3):
    return expand_search({'outputDir': str(root),
                          'baseArgs': ['--dataset-dir', 'prepared', '--epochs', '3', '--device', 'cpu', '--batch-size', '2', '--early-stopping-patience', '10'],
                          'parameters': {'--lr': [0.001 * (i + 1) for i in range(count)]}})


def checkpoint(directory, epoch=1, bad_epochs=0, file='last.pt'):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / file
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_config': {}, 'model_state_dict': {}, 'optimizer_state_dict': {},
                'training_settings': {}, 'epoch': epoch, 'bad_epochs': bad_epochs}, target)
    return target


def metrics(directory, epoch=1):
    directory.mkdir(exist_ok=True)
    (directory / 'metrics.tsv').write_text('event\tepoch\tglobal_step\tvalidation_loss\n' + f'epoch_end\t{epoch}\t{epoch}\t0.5\n')


def test_missing_row_resumes_checkpoint_and_starts_unattempted_trial(tmp_path):
    search = plan(tmp_path)
    calls = []
    def interrupt(argv):
        directory = Path(argv[argv.index('--output-dir') + 1])
        calls.append(argv)
        metrics(directory)
        if len(calls) == 2:
            checkpoint(directory)
            raise KeyboardInterrupt
        return 0
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call', side_effect=interrupt):
        with pytest.raises(KeyboardInterrupt):
            run(search)
    # Simulate a kill before the second trial's row/result was committed.
    report = json.loads((tmp_path / RESULT_NAME).read_text())
    report['trials'] = report['trials'][:1]
    (tmp_path / RESULT_NAME).write_text(json.dumps(report))
    rows = (tmp_path / 'tuning-scores.tsv').read_text().splitlines(keepends=True)
    (tmp_path / 'tuning-scores.tsv').write_text(''.join(rows[:2]))
    retried = []
    def resume(argv):
        retried.append(argv)
        return 0
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call', side_effect=resume):
        assert run(search) == 0
    assert len(retried) == 2
    assert retried[0][4] == 'fragment-tree-detailed'
    assert retried[0][retried[0].index('--resume') + 1] == str(tmp_path / 'trial-002' / 'last.pt')
    assert retried[0][retried[0].index('--epochs') + 1] == '2'
    assert retried[0][retried[0].index('--batch-size') + 1] == '2'
    assert '--resume' not in retried[1]
    with (tmp_path / 'tuning-scores.tsv').open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    assert [row['trial'] for row in rows] == ['1', '2', '3']
    assert all(row['status'] == 'completed' for row in rows)
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call') as train:
        assert run(search) == 0
        train.assert_not_called()


def test_running_row_is_not_a_completed_trial(tmp_path):
    search = plan(tmp_path, 1)
    def interrupt(argv):
        checkpoint(tmp_path / 'trial-001')
        raise KeyboardInterrupt
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call', side_effect=interrupt):
        with pytest.raises(KeyboardInterrupt):
            run(search)
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call', return_value=0) as train:
        assert run(search) == 0
        assert train.call_count == 1
        assert '--resume' in train.call_args.args[0]


def test_score_only_completion_retains_values_and_skips_training(tmp_path):
    search = plan(tmp_path, 1)
    def train(argv):
        metrics(tmp_path / 'trial-001')
        return 0
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call', side_effect=train):
        assert run(search) == 0
    (tmp_path / RESULT_NAME).unlink()
    (tmp_path / 'trial-001' / 'metrics.tsv').unlink()
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call') as train:
        assert run(search) == 0
        train.assert_not_called()
    with (tmp_path / 'tuning-scores.tsv').open() as stream:
        row = next(csv.DictReader(stream, delimiter='\t'))
    assert float(row['validation_loss.last']) == 0.5


@pytest.mark.parametrize('epoch,bad_epochs', [(3, 0), (1, 10)])
def test_finished_checkpoint_does_not_add_extra_epochs(tmp_path, epoch, bad_epochs):
    search = plan(tmp_path, 1)
    directory = tmp_path / 'trial-001'
    checkpoint(directory, epoch=epoch, bad_epochs=bad_epochs)
    argv, state, finished = continuation(search['trials'][0], directory)
    assert finished and argv is None and state['remainingEpochs'] == 0
    assert json.loads((directory / 'result.training.pft').read_text())['status'] == 'completed'


def test_corrupt_last_checkpoint_falls_back_to_periodic(tmp_path):
    search = plan(tmp_path, 1)
    directory = tmp_path / 'trial-001'
    file = checkpoint(directory, file='checkpoints/epoch_000001.pt')
    (directory / 'last.pt').write_bytes(b'interrupted write')
    argv, state, finished = continuation(search['trials'][0], directory)
    assert not finished
    assert argv[argv.index('--resume') + 1] == str(file)
    assert state['epoch'] == 1


def test_detailed_resume_does_not_pass_architecture_flags(tmp_path):
    search = plan(tmp_path, 1)
    trial = search['trials'][0]
    trial['args'][3] = 'fragment-tree-detailed'
    trial['args'] += ['--model-architecture-json', '{"model_config": {}}', '--pattern-hidden-dim', '0=32']
    directory = tmp_path / 'trial-001'
    checkpoint(directory)
    argv, _, _ = continuation(trial, directory)
    assert '--pattern-hidden-dim' not in argv
    assert '--model-architecture-json' not in argv
    assert '--resume' in argv


def test_changed_plan_is_rejected_even_if_plan_file_was_edited(tmp_path):
    search = plan(tmp_path, 1)
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call', return_value=0):
        assert run(search) == 0
    search['trials'][0]['args'] += ['--batch-size', '8']
    (tmp_path / PLAN_NAME).write_text(json.dumps(search))
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call') as train:
        with pytest.raises(ValueError, match='different plan'):
            run(search)
        train.assert_not_called()


def test_output_lock_prevents_concurrent_searches(tmp_path):
    import fcntl
    with (tmp_path / '.tuning.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='Another tuning process'):
            run(plan(tmp_path, 1))


def test_missing_checkpoint_keeps_files_and_continues_later_trials(tmp_path):
    search = plan(tmp_path, 2)
    def interrupt(argv):
        directory = tmp_path / 'trial-001'
        directory.mkdir()
        (directory / 'evidence.txt').write_text('preserve')
        raise KeyboardInterrupt
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call', side_effect=interrupt):
        with pytest.raises(KeyboardInterrupt):
            run(search)
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call', return_value=0) as train:
        assert run(search) == 1
        assert train.call_count == 1
        assert 'trial-002' in train.call_args.args[0][train.call_args.args[0].index('--output-dir') + 1]
    assert (tmp_path / 'trial-001' / 'evidence.txt').read_text() == 'preserve'
    report = json.loads((tmp_path / RESULT_NAME).read_text())
    assert 'without a valid checkpoint' in report['trials'][0]['error']
    assert report['trials'][1]['status'] == 'completed'


def test_completed_training_report_recovers_missing_score_and_result(tmp_path):
    search = plan(tmp_path, 1)
    def interrupt_after_training(argv):
        directory = tmp_path / 'trial-001'
        metrics(directory, 2)
        (directory / 'result.training.pft').write_text(json.dumps({'status': 'completed', 'completedEpochs': 2}))
        raise KeyboardInterrupt
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call', side_effect=interrupt_after_training):
        with pytest.raises(KeyboardInterrupt):
            run(search)
    (tmp_path / RESULT_NAME).unlink()
    (tmp_path / 'tuning-scores.tsv').unlink()
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call') as train:
        assert run(search) == 0
        train.assert_not_called()
    report = json.loads((tmp_path / RESULT_NAME).read_text())
    assert report['trials'][0]['scoreSummaries']['validation_loss']['last'] == 0.5


@pytest.mark.parametrize('legacy_name', ['fragment-tree.tuning.pft', 'fragment-tree.tuning-result.pft'])
def test_legacy_result_name_resumes_without_retraining(tmp_path, legacy_name):
    search = plan(tmp_path, 1)
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call', return_value=0):
        assert run(search) == 0
    assert RESULT_NAME == 'result.tuning.pft'
    legacy = tmp_path / legacy_name
    (tmp_path / RESULT_NAME).rename(legacy)
    (tmp_path / 'tuning-scores.tsv').unlink()
    with patch('fredis.ml.tuning.fragment_tree.subprocess.call') as train:
        assert run(search) == 0
        train.assert_not_called()
    result = json.loads((tmp_path / RESULT_NAME).read_text())
    assert result['status'] == 'completed'
    assert len(result['trials']) == 1
