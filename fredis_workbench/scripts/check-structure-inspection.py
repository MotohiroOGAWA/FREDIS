"""Regression check for preparation-only objects in inspection metadata."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

APP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_ROOT))
from fredis.domain.fragment.reaction import ReactionInstance, ReactionSequence

spec = importlib.util.spec_from_file_location(
    'structure_inspection', APP_ROOT / 'fredis_workbench/src/features/fragment-tree-result/backend.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
reaction = ReactionInstance(0, 0, (1,), frozenset({1}), frozenset(), frozenset(), frozenset(), product_molecule_id=0)
sequence = ReactionSequence((reaction,))
metadata = {'smiles': 'C', 'record_indexes': [7],
            'reaction_sequence_compounds': [(sequence, '[CH4:1]')]}
from fredis.ml.input.reaction_context_data import ReactionContextData, TeacherData
structure = SimpleNamespace(
    num_samples=0, num_branch_groups=0, sample_annotations=(),
    reactions=SimpleNamespace(reaction_type=torch.empty((0,3))),
    states=ReactionContextData(), teacher=TeacherData(),
    candidate_reactions=lambda: (torch.zeros(1,dtype=torch.long), torch.empty(0,dtype=torch.long)),
    state_reaction_count=torch.empty(0,dtype=torch.long),
    collision_energy_eV=torch.empty(0), branch_group_adduct_index=torch.empty(0,dtype=torch.long), sample_branch_group_index=torch.empty(0,dtype=torch.long),
    spectrum_batch=SimpleNamespace(decoded=SimpleNamespace(trees=(),compounds=()),
        state_node_index=torch.empty(0),state_adduct_index=torch.empty(0)),
)
with patch.object(backend.torch, 'load', return_value={'structure': structure, 'metadata': metadata}), \
     patch.object(backend, '_node_view', return_value=([], '', {}, None)):
    result = backend.inspect_structure(Path('fixture.preft.pt'))
    reply = json.loads(json.dumps({'ok': True, 'result': result}))
assert reply['result']['metadata'] == {'smiles': 'C', 'record_indexes': [7]}
assert metadata['reaction_sequence_compounds'][0][0] is sequence
print('Structure inspection serializes metadata and preserves preparation seeds.')

# The same combined label can arise from different base/shift decompositions.
positive = backend._display_ion_adducts(dict(fragmentIonAdduct='[M+H]+', hydrogenShift=-2, adduct='[M+H]+'))
negative = backend._display_ion_adducts(dict(fragmentIonAdduct='[M-H]+', hydrogenShift=0, adduct='[M-H]+'))
assert positive['adduct'] == negative['adduct'] == '[M-H]+'
assert positive['fragmentIonAdduct'] != negative['fragmentIonAdduct']
sodium = dict(fragmentIonAdduct='[M+Na]+', hydrogenShift=-2, adduct='[M-2H+Na]+')
assert backend._display_ion_adducts(sodium)['adduct'] == '[M+Na-2H]+'
assert sodium['adduct'] == '[M-2H+Na]+'
assert backend._display_ion_adducts(dict(adduct='[M-2H+Na]+'))['adduct'] == '[M+Na-2H]+'
assert 'fragmentIonAdduct' not in backend._display_ion_adducts(dict(adduct='[M-H]+'))
print('Ion labels use current canonical formatting and preserve base/shift distinctions.')
