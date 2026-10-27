"""Dataset manifest compatibility without importing the training runtime."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

file = Path(__file__).resolve().parents[3] / 'fredis/ml/training/fragment_tree_training/sources.py'
spec = importlib.util.spec_from_file_location('dataset_sources_manifest', file)
sources = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sources)


class DatasetManifestNames(unittest.TestCase):
    def test_current_manifest(self):
        config = {'fragmenterParams': {'example': True}, 'symbols': ['C', 'H']}
        for name in ['fragment-tree.training-data.pft']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / name).write_text(json.dumps(config))
                self.assertEqual(sources.dataset_chemistry_config(tmp)['fragmenter_params'], {'example': True})
