"""Check explicit fragment training options without starting training."""
import contextlib
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fredis.ml.training.fragment_tree_training.training import build_arg_parser, training_model_config, MODEL_OPTIONS, MOL_ENCODER_OPTIONS
parser = build_arg_parser()
base = ['--train-dir', 'train', '--val-dir', 'val', '--output-dir', 'out']
argv = list(base)
for flag, (_, _, kind, _) in MODEL_OPTIONS.items():
    argv += ['--' + flag, str(0.25 if kind is float else 64)]
config = training_model_config(parser.parse_args(argv))
assert 'fragmenter_params' not in config
for _, (section, key, kind, _) in MODEL_OPTIONS.items():
    assert config[section][key] == (0.25 if kind is float else 64)
assert config['reaction_model_params']['reaction_neighborhood_mode'] == 'hop_pooling'
assert config['reaction_model_params']['reaction_neighborhood_max_hop'] == 3
assert 'max_roles' not in config['reaction_model_params']
none_config = training_model_config(parser.parse_args(argv + ['--reaction-neighborhood-mode', 'none']))
assert none_config['reaction_model_params']['reaction_neighborhood_mode'] == 'none'
with contextlib.redirect_stderr(io.StringIO()):
    try:
        parser.parse_args(argv + ['--reaction-neighborhood-mode', 'gnn'])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError('--reaction-neighborhood-mode gnn')
hop_config = training_model_config(parser.parse_args(argv + ['--reaction-neighborhood-max-hop', '2']))
assert hop_config['reaction_model_params']['reaction_neighborhood_max_hop'] == 2
for flag, (key, _, default) in MOL_ENCODER_OPTIONS.items():
    assert config['mol_encoder_params'][key] == default
print('Explicit joint fragment CLI options passed.')
