"""Check explicit fragment training options without starting training."""
import contextlib
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fredis.ml.training.fragment_tree_training.training import build_arg_parser, training_model_config, MODEL_OPTIONS, MOL_ENCODER_OPTIONS
parser = build_arg_parser()
base = ['--dataset-dir', 'dataset', '--output-dir', 'out']
argv = list(base) + ['--dropout', '0.25']
for flag, (_, _, kind, _) in MODEL_OPTIONS.items():
    value = 'competitive' if flag=='spectrum-prediction-mode' else 'mlp' if flag=='collision-energy-encoding' else '32,16' if flag.endswith('fc-dims') else str(0.25 if kind is float else 64)
    argv += ['--' + flag, value]
config = training_model_config(parser.parse_args(argv))
assert 'fragmenter_params' not in config
for _, (section, key, kind, _) in MODEL_OPTIONS.items():
    expected = 'competitive' if key=='prediction_mode' else 'mlp' if key=='collision_energy_encoding' else [32,16] if key.endswith('fc_dims') else 0.25 if kind is float else 64
    assert config[section][key] == expected
assert 'reaction_neighborhood_mode' not in config['reaction_model_params']
assert 'reaction_neighborhood_max_hop' not in config['reaction_model_params']
for flag, (key, _, default) in MOL_ENCODER_OPTIONS.items():
    assert config['mol_encoder_params'][key] == default
assert config['spectrum_model_params']['adduct_embedding_dim'] == 64
assert 'ce_dropout' not in config['spectrum_model_params']
with contextlib.redirect_stderr(io.StringIO()):
    try:
        parser.parse_args(base + ['--ce-dropout', '0.1'])
    except SystemExit:
        pass
    else:
        raise AssertionError('Removed CE dropout flag must be rejected')
print('Explicit joint fragment CLI options passed.')
