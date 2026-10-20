from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


APP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_ROOT))
backend_path = Path(__file__).resolve().parents[1] / "src/features/spectrum-prediction/backend.py"
spec = importlib.util.spec_from_file_location("spectrum_prediction_backend", backend_path)
backend = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(backend)


reaction = object()
sequence = SimpleNamespace(reactions=(reaction,))
transition = SimpleNamespace(
    added_reaction=reaction,
    parent_reaction_sequence=None,
    reaction_sequence=sequence,
    is_seed=False,
)
edge = SimpleNamespace(index=0, source_index=0, target_index=1, transitions=(transition,))
tree = SimpleNamespace(num_edges=1, get_edge=lambda index: edge)
formula = SimpleNamespace(exact_mass=32.0262, __str__=lambda self: "CH4O")
fragments = SimpleNamespace(
    node_sample_index=torch.tensor([0, 0]),
    trees=(tree,),
    compounds=(
        SimpleNamespace(smiles="CCO", formula=SimpleNamespace(exact_mass=46.0419, __str__=lambda self: "C2H6O")),
        SimpleNamespace(smiles="CO", formula=formula),
    ),
)
spectrum_batch = SimpleNamespace(
    state_node_index=torch.tensor([1]),
    state_adducts=("[M+H]+",),
    state_sample_index=torch.tensor([0]),
    state_mz=torch.tensor([33.0335]),
    state_formula_tensor=torch.tensor([[1, 4, 1]]),
)
spectra = SimpleNamespace(
    ion_probability=torch.tensor([0.9]),
    intensity=torch.tensor([1.0]),
)
result = SimpleNamespace(fragments=fragments, spectrum_batch=spectrum_batch, spectra=spectra)
tensorizer = SimpleNamespace(tensor_to_formula=lambda row: "CH5O+")

serialized = backend._build_tree(result, 0, tensorizer, 0.5, (reaction,))
assert serialized["edges"][0]["transitions"] == [{
    "addedReaction": 0,
    "parentReactions": [],
    "targetReactions": [0],
    "seed": False,
}]
assert serialized["nodes"][0]["reactionSets"] == [[]]
assert serialized["nodes"][1]["reactionSets"] == [[0]]
assert serialized["nodes"][1]["annotations"][0]["ionFormula"] == "CH5O+"

from fredis.domain.fragment.fragmenter import Fragmenter
from fredis.libs.mmkit.mmkit import Compound

fragmenter = Fragmenter.from_dict(json.loads((APP_ROOT / "fredis/presets/fragment/single_bond_cleavage_pos.fragmenter.pft").read_text()))
source = Compound.from_smiles("CCOC(=O)C")
real_reactions = tuple(fragmenter.fragment_ion_tree_builder.create_reaction_instances(source))
registry = backend._reaction_registry(SimpleNamespace(fragmenter=fragmenter), real_reactions)
assert len(registry) == len(real_reactions) > 0
assert all({"id", "reactionPatternId", "reactionRuleId", "sourceAtomMaps", "retainedAtomMaps", "cutBondMaps"} <= row.keys() for row in registry)

print("Single-spectrum prediction tree preserves selectable edge transitions and reaction histories.")
