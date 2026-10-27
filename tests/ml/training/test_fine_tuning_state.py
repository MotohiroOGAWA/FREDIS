"""initialize_from_base() requires every base parameter, including pattern atom encoders."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from fredis.ml.training.fragment_tree_training.fine_tuning import initialize_from_base


def _toy_generator():
    generator = nn.Module()
    generator.feature_model = nn.Module()
    generator.feature_model.reaction_instance_encoder = nn.Module()
    reaction_instance_encoder = generator.feature_model.reaction_instance_encoder
    generator.feature_model.mol_encoder = nn.Linear(4, 4)
    reaction_instance_encoder.reactant_encoders = nn.ModuleDict({'0': nn.Linear(4, 4, bias=False)})
    return generator


class TestPatternFineTuning(unittest.TestCase):
    def _assert_missing_raises(self, missing_key):
        old_state = _toy_generator().state_dict()
        broken_state = {key: value for key, value in old_state.items() if key != missing_key}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / 'base.pt'
            torch.save({'model_state_dict': broken_state}, checkpoint_path)
            new = _toy_generator()
            with self.assertRaisesRegex(ValueError, 'Base checkpoint is missing'):
                initialize_from_base(new, {'fine_tuning': {'source_checkpoint': str(checkpoint_path), 'pattern_mapping': [[0, 0]]}})

    def test_missing_base_parameter_raises(self):
        self._assert_missing_raises('feature_model.mol_encoder.bias')

    def test_missing_reaction_encoder_raises(self):
        self._assert_missing_raises('feature_model.reaction_instance_encoder.reactant_encoders.0.weight')

    def test_complete_base_checkpoint_is_restored(self):
        old = _toy_generator()
        nn.init.normal_(old.feature_model.reaction_instance_encoder.reactant_encoders['0'].weight)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / 'base.pt'
            torch.save({'model_state_dict': old.state_dict()}, checkpoint_path)
            new = _toy_generator()
            initialize_from_base(new, {'fine_tuning': {'source_checkpoint': str(checkpoint_path), 'pattern_mapping': [[0, 0]]}})
            self.assertTrue(torch.equal(new.feature_model.reaction_instance_encoder.reactant_encoders['0'].weight,
                                        old.feature_model.reaction_instance_encoder.reactant_encoders['0'].weight))


if __name__ == '__main__':
    unittest.main()
