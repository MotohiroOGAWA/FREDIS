import math
import pytest

import torch
from torch_geometric.data import Batch,Data

from fredis.domain.fragment.ion_tree import FragmentIonAdductRuleSet
from fredis.libs.mmkit.mmkit import Compound
from fredis.ml.mol.mol_encoder import MolEncoder


def rule_set():
    return FragmentIonAdductRuleSet.from_dict({"fragment_ion_adduct_rule_set":{
        "name":"positive_default","hydrogen_shift_min":-5,"hydrogen_shift_max":0,
        "adduct_rules":[{"name":"protonated","adduct_type":"[M+H]+",
                         "fragment_ion_adducts":["[M+H]+","[M-H]+"]}]}})


def test_ion_rule_schema_and_shared_hydrogen_range():
    rules=rule_set();assert rules.hydrogen_shifts==tuple(range(-5,1))
    encoded=rules.to_dict();assert encoded["hydrogen_shift_min"]==-5
    assert encoded["adduct_rules"][0]["fragment_ion_adducts"]==["[M+H]+","[M-H]+"]
    assert set(encoded["adduct_rules"][0])=={"name","adduct_type","fragment_ion_adducts"}




def test_mol_encoder_external_features():
    model = MolEncoder(("C", "H", "O"), 8, 16, 1, 2, external_feature_dim=2).eval()
    batch = Batch.from_data_list([
        model.encode_components(Compound.from_smiles("CCO")),
        model.encode_components(Compound.from_smiles("CC")),
    ])
    features = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    output = model(batch, features)
    assert output.shape == (2, 16)
    changed = model(batch, features + 1)
    assert not torch.equal(output, changed)
    torch.testing.assert_close(model.encode_with_nodes(batch, features).x,
                               model.encode_with_nodes(batch, features + 1).x)
    output.square().mean().backward()
    assert features.grad.abs().sum() > 0
    assert model.external_feature_projection[0].weight.grad.abs().sum() > 0
    batch.external_features = features.detach()
    torch.testing.assert_close(model(batch), output)
    del batch.external_features
    for invalid in (None, torch.zeros(1, 2), torch.zeros(2, 3), torch.full((2, 2), float("nan"))):
        with pytest.raises(ValueError, match="external_features"):
            model(batch, invalid)
    plain = MolEncoder(("C", "H", "O"), 8, 16, 1, 2)
    assert plain.external_feature_projection is None
    assert plain(batch).shape == (2, 16)
    with pytest.raises(ValueError, match="external_feature_dim"):
        plain(batch, features)
    for width in (-1, True, 1.5):
        with pytest.raises(ValueError, match="external_feature_dim"):
            MolEncoder(("C", "H", "O"), 8, 16, 1, 2, external_feature_dim=width)


def test_workbench_descriptor_choices_match_supported_descriptors():
    import re
    from pathlib import Path
    from fredis.domain.molecule.descriptors import SUPPORTED_DESCRIPTOR_NAMES
    text=Path("fredis_workbench/src/workbench/mol-descriptors.js").read_text()
    assert tuple(re.findall(r"'([A-Za-z0-9]+)'",text))==SUPPORTED_DESCRIPTOR_NAMES
