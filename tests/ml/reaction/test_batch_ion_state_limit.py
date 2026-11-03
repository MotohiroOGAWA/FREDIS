import csv
import random
import tempfile
import unittest
from pathlib import Path

from fredis.ml.training.fragment_tree_training.memory_estimate import (
    _ChunkPlan,
    fit_memory_model,
    predict_dynamic,
)
from fredis.ml.training.fragment_tree_training.prepared_dataset import prepared_piece_sequence
from fredis.ml.training.fragment_tree_training.training import plan_batch_groups


def write_dataset(root: Path, *, per_sample_ions: bool):
    # a: 4 samples with very different ion states; b: 2 small samples.
    ions = {'a.preft.pt': [100, 900, 50, 5000], 'b.preft.pt': [10, 20]}
    for name in ions:
        (root / name).touch()
    with (root / 'manifest.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('file', 'num_valid_samples', 'num_ion_states',
                                                    'num_transition_states', 'status'), delimiter='\t')
        writer.writeheader()
        for name, values in ions.items():
            writer.writerow({'file': name, 'num_valid_samples': len(values), 'num_ion_states': sum(values),
                             'num_transition_states': 7, 'status': 'completed'})
    fields = ['structure_file', 'sample_index', 'assignment_score', 'assignment_score_without_precursor']
    if per_sample_ions:
        fields.insert(2, 'num_ion_states')
    with (root / 'assignment_scores.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for name, values in ions.items():
            for index, value in enumerate(values):
                row = {'structure_file': name, 'sample_index': index, 'assignment_score': 1,
                       'assignment_score_without_precursor': 1}
                if per_sample_ions:
                    row['num_ion_states'] = value
                writer.writerow(row)


class TestBatchIonStateLimit(unittest.TestCase):
    def sequence(self, root, limit, **kwargs):
        return prepared_piece_sequence(root, max_samples=128, minimum_assignment_score=0,
                                       minimum_assignment_score_without_precursor=0,
                                       cache_in_memory=False, max_batch_ion_states=limit, **kwargs)

    def test_unlimited_keeps_one_chunk_per_file_with_exact_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); write_dataset(root, per_sample_ions=True)
            pieces = self.sequence(root, 0)
            self.assertTrue(pieces.ion_state_counts_exact)
            self.assertEqual([pieces.piece(i).ion_states for i in range(len(pieces))], [6050, 30])
            self.assertEqual(pieces.transition_state_count(0), 7)

    def test_limit_splits_chunks_and_excludes_oversized_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); write_dataset(root, per_sample_ions=True)
            pieces = self.sequence(root, 1000)
            chunks = [pieces.piece(i) for i in range(len(pieces))]
            self.assertEqual([c.sample_indices for c in chunks], [(0, 1), (2,), (0, 1)])
            self.assertTrue(all(c.ion_states <= 1000 for c in chunks))
            self.assertEqual(pieces.oversized_samples, 1)  # the 5000-ion sample

    def test_limit_with_filtering_still_reads_per_sample_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); write_dataset(root, per_sample_ions=True)
            pieces = prepared_piece_sequence(
                root, max_samples=128, minimum_assignment_score=0.5,
                minimum_assignment_score_without_precursor=0, cache_in_memory=False,
                max_batch_ion_states=1000)
            self.assertTrue(pieces.ion_state_counts_exact)
            self.assertEqual(pieces.oversized_samples, 1)

    def test_without_per_sample_counts_uses_file_average(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); write_dataset(root, per_sample_ions=False)
            pieces = self.sequence(root, 4000)
            self.assertFalse(pieces.ion_state_counts_exact)
            # a averages 1512.5 ions/sample -> two samples per chunk.
            self.assertEqual([pieces.piece(i).sample_indices for i in range(len(pieces))],
                             [(0, 1), (2, 3), (0, 1)])

    def test_batches_close_before_exceeding_the_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); write_dataset(root, per_sample_ions=True)
            pieces = self.sequence(root, 1000)
            groups = plan_batch_groups(pieces, False, batch_size=4, max_samples=128, max_batch_ion_states=1000)
            totals = [sum(pieces.ion_state_count(i) for i in group) for group in groups]
            self.assertTrue(all(total <= 1000 for total in totals))
            # Without the limit, batch_size alone packs all three chunks together.
            self.assertEqual(len(plan_batch_groups(pieces, False, batch_size=4, max_samples=128)), 1)

    def test_chunk_plan_mirrors_limit_without_loading_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); write_dataset(root, per_sample_ions=True)
            pieces = self.sequence(root, 0)
            plan, dropped = _ChunkPlan.from_pieces(pieces, 2000)
            self.assertEqual(dropped, 0)
            self.assertTrue(all(c.ion_states <= 2000 for c in plan.chunks))
            self.assertEqual(sum(c.samples for c in plan.chunks), 6)
            random.seed(0)
            self.assertEqual(sorted(plan.ordered_indices(True)), list(range(len(plan))))


class TestMemoryFit(unittest.TestCase):
    def test_two_term_fit_recovers_coefficients(self):
        points = [dict(ion_states=i, transition_states=t, dynamic_bytes=1_000 + 10 * i + 500 * t)
                  for i, t in ((100, 5), (400, 3), (900, 20), (1600, 8), (2500, 40))]
        fit = fit_memory_model(points)
        self.assertAlmostEqual(fit['bytes_per_ion_state'], 10, places=4)
        self.assertAlmostEqual(fit['bytes_per_transition_state'], 500, places=3)
        self.assertAlmostEqual(fit['r2'], 1.0, places=6)
        self.assertAlmostEqual(predict_dynamic(fit, 1000, 10), 1_000 + 10_000 + 5_000, delta=1)

    def test_negative_coefficient_is_dropped(self):
        points = [dict(ion_states=i, transition_states=t, dynamic_bytes=10 * i)
                  for i, t in ((100, 50), (400, 30), (900, 10), (1600, 5))]
        fit = fit_memory_model(points)
        self.assertGreaterEqual(fit['bytes_per_transition_state'], 0)
        self.assertGreater(fit['bytes_per_ion_state'], 0)

    def test_single_point_is_proportional(self):
        fit = fit_memory_model([dict(ion_states=100, transition_states=4, dynamic_bytes=5000)])
        self.assertEqual(fit['bytes_per_ion_state'], 50)
        self.assertIsNone(fit['r2'])


if __name__ == '__main__':
    unittest.main()
