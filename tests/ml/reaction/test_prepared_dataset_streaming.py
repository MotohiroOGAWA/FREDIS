import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fredis.ml.training.fragment_tree_training.prepared_dataset import prepared_piece_sequence


class TestPreparedDatasetStreaming(unittest.TestCase):
    def dataset(self, root: Path):
        for name in ('a.preft.pt', 'b.preft.pt'):
            (root / name).touch()
        with (root / 'manifest.tsv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=('file', 'num_valid_samples', 'status'), delimiter='\t')
            writer.writeheader()
            writer.writerows((
                {'file': 'a.preft.pt', 'num_valid_samples': 3, 'status': 'completed'},
                {'file': 'b.preft.pt', 'num_valid_samples': 1, 'status': 'completed'},
            ))
        with (root / 'assignment_scores.tsv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                'structure_file', 'sample_index', 'assignment_score',
                'assignment_score_without_precursor'), delimiter='\t')
            writer.writeheader()
            writer.writerows((
                {'structure_file': 'a.preft.pt', 'sample_index': 0, 'assignment_score': .9,
                 'assignment_score_without_precursor': .9},
                {'structure_file': 'a.preft.pt', 'sample_index': 1, 'assignment_score': .1,
                 'assignment_score_without_precursor': .9},
                {'structure_file': 'a.preft.pt', 'sample_index': 2, 'assignment_score': .8,
                 'assignment_score_without_precursor': .8},
                {'structure_file': 'b.preft.pt', 'sample_index': 0, 'assignment_score': .7,
                 'assignment_score_without_precursor': .7},
            ))

    def structures(self):
        return {
            'a.preft.pt': SimpleNamespace(num_samples=3, spectrum_batch=object()),
            'b.preft.pt': SimpleNamespace(num_samples=1, spectrum_batch=object()),
        }

    def test_streaming_is_lazy_and_retains_only_the_most_recent_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.dataset(root); structures = self.structures()
            with patch('fredis.ml.training.fragment_tree_training.prepared_dataset.SourceReactionStructure.load',
                       side_effect=lambda path: structures[path.name]) as load, \
                 patch('fredis.ml.training.fragment_tree_training.prepared_dataset.select_samples',
                       side_effect=lambda data, indices: (data, tuple(indices))):
                pieces = prepared_piece_sequence(root, max_samples=2, minimum_assignment_score=0,
                    minimum_assignment_score_without_precursor=0, cache_in_memory=False)
                self.assertEqual(load.call_count, 0)
                self.assertEqual(len(pieces), 3)
                pieces[0]; pieces[1]
                self.assertEqual(load.call_count, 1)
                pieces[2]; pieces[0]
                self.assertEqual(load.call_count, 3)

    def test_ram_cache_also_loads_on_first_use_and_then_retains_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.dataset(root); structures = self.structures()
            with patch('fredis.ml.training.fragment_tree_training.prepared_dataset.SourceReactionStructure.load',
                       side_effect=lambda path: structures[path.name]) as load, \
                 patch('fredis.ml.training.fragment_tree_training.prepared_dataset.select_samples',
                       side_effect=lambda data, indices: (data, tuple(indices))):
                pieces = prepared_piece_sequence(root, max_samples=2, minimum_assignment_score=0,
                    minimum_assignment_score_without_precursor=0, cache_in_memory=True)
                self.assertEqual(load.call_count, 0)
                pieces[0]; pieces[2]; pieces[0]
                self.assertEqual(load.call_count, 2)

    def test_assignment_filter_and_lazy_subset_use_sidecar_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.dataset(root); structures = self.structures()
            with patch('fredis.ml.training.fragment_tree_training.prepared_dataset.SourceReactionStructure.load',
                       side_effect=lambda path: structures[path.name]) as load, \
                 patch('fredis.ml.training.fragment_tree_training.prepared_dataset.select_samples',
                       side_effect=lambda data, indices: (data, tuple(indices))) as select:
                pieces = prepared_piece_sequence(root, max_samples=8, minimum_assignment_score=.75,
                    minimum_assignment_score_without_precursor=.75, cache_in_memory=False)
                self.assertEqual(pieces.total_unfiltered_samples, 4)
                self.assertEqual(pieces.total_samples, 2)
                subset = pieces.sample_subset([1])
                self.assertEqual(load.call_count, 0)
                subset[0]
                self.assertEqual(select.call_args.args[1], (2,))


if __name__ == '__main__':
    unittest.main()
