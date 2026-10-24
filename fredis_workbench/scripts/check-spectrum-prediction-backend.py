from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

APP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_ROOT))
backend_path = (
    Path(__file__).resolve().parents[1] / "src/features/spectrum-prediction/backend.py"
)
spec = importlib.util.spec_from_file_location(
    "spectrum_prediction_backend", backend_path
)
backend = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(backend)


from tests.ml.reaction.test_canonical_workflow import prepared
from fredis.libs.mmkit.mmkit import Compound, Adduct

model, _ = prepared()
model.eval()
source = Compound.from_smiles("CCO")
result = model.predict([source], [Adduct.parse("[M+H]+")], [20.0])
serialized = backend._build_tree(result, 0, model.tensorizer, 0.0)
node_ids = {node["id"] for node in serialized["nodes"]}
assert len(node_ids) > 1 and serialized["edges"]
assert all(
    edge["source"] in node_ids and edge["target"] in node_ids and edge["transitions"]
    for edge in serialized["edges"]
)
assert any(
    node["precursor"] and [] in node["reactionSets"] for node in serialized["nodes"]
)
assert any(node["annotations"] for node in serialized["nodes"])
json.dumps(serialized)

from fredis.domain.fragment.fragmenter import Fragmenter
from fredis.libs.mmkit.mmkit import Compound

fragmenter = Fragmenter.from_dict(
    json.loads(
        (
            APP_ROOT / "fredis/presets/fragment/single_bond_cleavage_pos.fragmenter.pft"
        ).read_text()
    )
)
source = Compound.from_smiles("CCOC(=O)C")
real_reactions = tuple(
    fragmenter.fragment_ion_tree_builder.create_reaction_instances(source)
)
registry = backend._reaction_registry(
    SimpleNamespace(fragmenter=fragmenter), real_reactions
)
assert len(registry) == len(real_reactions) > 0
assert all(
    {
        "id",
        "reactionPatternId",
        "reactionRuleId",
        "sourceAtomMaps",
        "retainedAtomMaps",
        "cutBondMaps",
    }
    <= row.keys()
    for row in registry
)

print(
    "Single-spectrum prediction tree preserves selectable edge transitions and reaction histories."
)

# The single-prediction UI defaults to zero and accepts a numeric intensity cutoff.
from dataclasses import replace
from tempfile import NamedTemporaryFile
from unittest.mock import patch

model.set_generation(replace(model.generation, peak_intensity_threshold=0.5))
with NamedTemporaryFile(suffix=".pt") as checkpoint:
    request = {
        "modelPath": checkpoint.name,
        "smiles": "CCO",
        "ce": "20",
        "adductType": "[M+H]+",
    }
    with patch.object(backend, "load_generator", return_value=model):
        unfiltered = backend.predict(request)
        filtered = backend.predict({**request, "peakIntensityThreshold": 0.5})
    assert any(p["intensity"] <= 0.5 for p in unfiltered["peaks"])
    assert all(p["intensity"] > 0.5 for p in filtered["peaks"])
    assert len(filtered["peaks"]) < len(unfiltered["peaks"])
print("Single-spectrum intensity threshold defaults to zero and accepts numeric values.")
