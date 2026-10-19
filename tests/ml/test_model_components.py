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




def test_mol_encoder_concatenates_selected_descriptors():
    from fredis.domain.molecule.descriptors import SUPPORTED_DESCRIPTOR_NAMES
    default=MolEncoder(("C","H","O"),8,16,1,2)
    assert default.descriptors==("FormalCharge",)
    model=MolEncoder(("C","H","O"),8,16,1,2,descriptors=("ExactMolWt","TPSA"))
    graph=model.encode_components(Compound.from_smiles("CCO"))
    assert graph.descriptors.shape==(1,len(SUPPORTED_DESCRIPTOR_NAMES))
    batch=Batch.from_data_list([graph,model.encode_components(Compound.from_smiles("CC"))])
    embedded=model(batch)
    assert embedded.shape==(2,16) and torch.isfinite(embedded).all()
    embedded.square().mean().backward()
    assert model.descriptor_projection[0].weight.grad is not None
    assert MolEncoder(("C","H","O"),8,16,1,2,descriptors=()).descriptor_projection is None
    with pytest.raises(ValueError,match="Unsupported descriptors"):
        MolEncoder(("C","H","O"),8,16,1,2,descriptors=("NotADescriptor",))






def test_workbench_descriptor_choices_match_supported_descriptors():
    import re
    from pathlib import Path
    from fredis.domain.molecule.descriptors import SUPPORTED_DESCRIPTOR_NAMES
    text=Path("fredis_workbench/src/workbench/mol-descriptors.js").read_text()
    assert tuple(re.findall(r"'([A-Za-z0-9]+)'",text))==SUPPORTED_DESCRIPTOR_NAMES
