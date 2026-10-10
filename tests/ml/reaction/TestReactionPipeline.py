"""Shared compact model configuration for fragment-tree integration tests."""
import json
from pathlib import Path


def config() -> dict:
    value=json.loads(Path('fredis/presets/spectrum_generator_params/source_anchored_pos_model_config.json').read_text())
    value['mol_encoder_params'].update(node_dim=16,graph_dim=16,num_layers=1,num_heads=4,dropout=0.)
    value['reaction_model_params'].update(hidden_dim=16,branch_main_adduct_dim=16,num_heads=4,state_num_layers=1)
    value['spectrum_model_params'].update(hidden_dim=16,state_hidden_dim=16,num_layers=1,num_heads=4,dropout=0.)
    value['generation_params'].update(branch_path_threshold=0.,max_fragment_nodes=32)
    return value
