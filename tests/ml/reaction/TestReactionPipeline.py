"""Shared compact chemistry and model configuration for fragment-tree integration tests."""

import json
from pathlib import Path

FRAGMENTER = Path("fredis/presets/fragment/single_bond_cleavage_pos.fragmenter.pft")
SYMBOLS = ["C", "N", "O", "P", "S", "F", "Cl", "Br", "I"]


def chemistry() -> dict:
    """Prepared-data chemistry: the Fragmenter and atom-feature symbols."""
    return {
        "fragmenter_params": json.loads(FRAGMENTER.read_text()),
        "symbols": list(SYMBOLS),
    }


def config() -> dict:
    """A small model configuration over the same chemistry."""
    return {
        "architecture": "fragment-tree-competitive-ion",
        "dropout": 0.0,
        "fragmenter_params": chemistry()["fragmenter_params"],
        "mol_encoder_params": dict(
            symbols=list(SYMBOLS),
            node_dim=16,
            graph_dim=16,
            num_layers=1,
            num_heads=4,
            max_degree=4,
            max_spatial_dist=3,
            max_edge_dist=3,
            dropout=0.0,
            external_feature_dim=0,
        ),
        "reaction_model_params": dict(
            hidden_dim=16, branch_main_adduct_dim=16, num_heads=4
        ),
        "spectrum_model_params": dict(fragment_fc_dims=[16], state_fc_dims=[16]),
        "generation_params": dict(branch_path_threshold=0.0, max_fragment_nodes=32),
        # Neutral-loss forms resolve to a registered main adduct during prediction.
        "adduct_type_strs": [
            "[M+H]+",
            "[M+Na]+",
            "[M+H4N]+",
            "[2M+H]+",
            "[M+H-H2O]+",
            "[M+H-2H2O]+",
            "[M+H-C6H6]+",
        ],
    }
