import json

import pytest

from fredis.domain.mass.tolerance import DaTolerance
from fredis.ml.training.fragment_tree_training.report import REPORT_NAME, write_training_report
from fredis.ml.training.fragment_tree_training.training import (
    _mean_training_interval, _read_metric_events, _write_metric_artifacts,
)
from fredis.ml.training.fragment_tree_training.spectrum_validation import (
    TOP_N_VALUES, assignment_score, distribution, matched_cosine, top_n_assignment_metrics, _report_metrics)


def test_assignment_and_cosine_without_intensity_prediction():
    tolerance = DaTolerance(0.01)
    observed = [{'mz': 100., 'intensity': 9.}, {'mz': 200., 'intensity': 1.}]
    assert assignment_score([100.], observed, tolerance) == pytest.approx(.9)
    assert matched_cosine([100.], [1.], observed, tolerance) == pytest.approx(9 / (82 ** .5))


def test_top_n_metrics_reward_recall_but_penalize_padding_with_junk_candidates():
    # A model that generates a huge pool of near-zero-intensity junk nodes (the
    # user's concern: max_fragment_nodes~100 makes "generate everything" look
    # attractive) must not get free recall/precision for that -- only the
    # candidates it actually ranks highly should count.
    tolerance = DaTolerance(0.5)
    observed = [{'mz': 100., 'intensity': 10., 'precursor': False, 'assigned': True},
                {'mz': 200., 'intensity': 5., 'precursor': False, 'assigned': True},
                {'mz': 300., 'intensity': 1., 'precursor': False, 'assigned': True},
                {'mz': 999., 'intensity': 50., 'precursor': False, 'assigned': False}]
    generated = ([{'mz': 100., 'intensity': 9.}, {'mz': 200., 'intensity': 4.}] +
                 [{'mz': 500. + i, 'intensity': .01} for i in range(30)])
    metrics = top_n_assignment_metrics(generated, observed, tolerance)
    assert metrics['peak_recall_top1'] == pytest.approx(1 / 3)
    assert metrics['peak_precision_top1'] == pytest.approx(1.0)
    assert metrics['peak_recall_top5'] == pytest.approx(2 / 3)
    assert metrics['peak_precision_top5'] == pytest.approx(2 / 5)
    # Recall cannot improve past what was actually generated (300 was never
    # produced), but precision keeps dropping as N grows past real hits --
    # exactly the signal that a large candidate pool doesn't buy a free ride.
    assert metrics['peak_recall_top20'] == pytest.approx(2 / 3)
    assert metrics['peak_precision_top20'] == pytest.approx(2 / 20)


def test_top_n_metrics_are_none_without_any_assigned_ground_truth():
    tolerance = DaTolerance(0.01)
    observed = [{'mz': 1., 'intensity': 1., 'precursor': False, 'assigned': False}]
    generated = [{'mz': 1., 'intensity': 1.}]
    metrics = top_n_assignment_metrics(generated, observed, tolerance)
    for n in TOP_N_VALUES:
        assert metrics[f'peak_recall_top{n}'] is None
        assert metrics[f'peak_precision_top{n}'] is None


def test_top_n_metrics_use_suffix_and_report_metrics_facets_without_depth():
    tolerance = DaTolerance(0.01)
    observed = [{'mz': 100., 'intensity': 1., 'precursor': False, 'assigned': True}]
    generated = [{'mz': 100., 'intensity': 1.}]
    metrics = top_n_assignment_metrics(generated, observed, tolerance, suffix='_without_precursor')
    assert set(metrics) == {f'peak_{kind}_top{n}_without_precursor' for kind in ('recall', 'precision') for n in TOP_N_VALUES}
    records = [{'collision_energy_bin': 'Low', 'main_adduct': '[M+H]+', 'depth_metrics': {},
                'peak_recall_top1': 1.0, 'peak_precision_top1': 1.0,
                'peak_recall_top1_without_precursor': None, 'peak_precision_top1_without_precursor': None}]
    report = _report_metrics(records)
    assert report['peak_recall_top1'] == [1.0]
    assert report['peak_recall_top1@collision_energy:Low'] == [1.0]
    assert report['peak_recall_top1@main_adduct:[M+H]+'] == [1.0]
    assert not any(key.startswith('peak_recall_top1@depth:') for key in report)


def test_report_quantiles_and_pft_manifest(tmp_path):
    stats = distribution(range(11))
    assert stats['q10'] == 1
    assert stats['q25'] == 2.5
    assert stats['median'] == 5
    assert stats['q75'] == 7.5
    assert stats['q90'] == 9
    target = write_training_report(tmp_path / 'project' / 'runs' / 'run-1', status='running',
                                   completed_epochs=2, global_step=40,
                                   latest_validation='spectrum_validation/epoch_2.json')
    assert target.name == REPORT_NAME == 'training.pft'
    payload = json.loads(target.read_text())
    assert payload['schema'] == 'fredis.training-report'
    assert payload['kind'] == 'training-result'
    assert payload['completedEpochs'] == 2
    assert payload['artifacts']['metrics'] == 'metrics.tsv'
    assert 'metricsJson' not in payload['artifacts']


def test_live_train_and_epoch_metrics_round_trip_through_tsv(tmp_path):
    report = {
        'train_step_history': [
            {'epoch': 1, 'global_step': 100, 'train_loss': 2.5,
             'train/samples': 400, 'train/batches': 100},
        ],
        'history': [
            {'epoch': 1, 'global_step': 120, 'train_loss': 2.0,
             'validation_loss': 1.5},
        ],
    }
    _write_metric_artifacts(tmp_path, report)
    rows = _read_metric_events(tmp_path / 'metrics.tsv')
    assert [row['event'] for row in rows] == ['train_step', 'epoch_end']
    assert rows[0]['global_step'] == 100
    assert rows[0]['train_loss'] == pytest.approx(2.5)
    assert rows[0]['train_samples'] == 400
    assert not (tmp_path / 'metrics.json').exists()


def test_training_interval_metrics_are_means():
    row = _mean_training_interval(
        totals={'loss': 11.0, 'branch_loss': 7.0},
        batch_metrics={'gradient_norm': [1.0, 3.0], 'learning_rate': [.1, .2]},
        samples=4, batches=2, epoch=1, global_step=100, elapsed_seconds=5.0,
    )
    assert row['train_loss'] == pytest.approx(2.75)
    assert row['train/branch_loss'] == pytest.approx(1.75)
    assert row['gradient_norm'] == pytest.approx(2.0)
    assert row['learning_rate'] == pytest.approx(.15)
