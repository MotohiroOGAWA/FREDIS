import math
import pytest

import torch
from torch_geometric.data import Batch,Data

from fredis.domain.fragment.ion_tree import FragmentIonAdductRuleSet
from fredis.libs.mmkit.mmkit import Compound
from fredis.ml.mol.mol_encoder import MolEncoder
from fredis.ml.reaction import EdgeAwareGCN,LocalContextGraphBuilder,LocalContextEncoder,ReactantEncoderRegistry
from fredis.ml.specgen.fragment_spectrum_model import sum_explanation_log_probabilities
from fredis.ml.training.reaction_preference import combine_teacher_and_preference_weights


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


def test_equivalent_explanation_log_probabilities_are_plain_sum():
    logits=torch.log(torch.tensor([.5,.25,.8]))
    result=sum_explanation_log_probabilities(logits,torch.tensor([0,2,3]))
    torch.testing.assert_close(result,torch.tensor([math.log(.125),math.log(.8)]))


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


def test_boundary_anchor_is_zero_and_bond_features_affect_messages():
    model=MolEncoder(("C","H","O","N"),8,16,1,2)
    context=LocalContextGraphBuilder(model.graph_builder,radius=1).build(Compound.from_smiles("CC(O)N"),[1])
    assert context.boundary_anchor_mask.sum()==1
    assert torch.count_nonzero(context.x[context.boundary_anchor_mask])==0
    encoder=EdgeAwareGCN(model.atom_dim,model.bond_dim,8,1).eval()
    first=encoder(context.x,context.edge_index,context.edge_attr)
    second=encoder(context.x,context.edge_index,torch.zeros_like(context.edge_attr))
    assert not torch.equal(first,second)


def test_pattern_registry_selects_independent_encoder_and_preference_is_opt_in():
    registry=ReactantEncoderRegistry({1:2,2:3},4,2,3,8,1)
    assert registry.encoders["1"] is not registry.encoders["2"]
    base=torch.tensor([.25,1.]);preference=torch.tensor([2.,3.])
    torch.testing.assert_close(combine_teacher_and_preference_weights(base,preference),base)
    torch.testing.assert_close(combine_teacher_and_preference_weights(base,preference,enabled=True),base*preference)


def test_workbench_descriptor_choices_match_supported_descriptors():
    import re
    from pathlib import Path
    from fredis.domain.molecule.descriptors import SUPPORTED_DESCRIPTOR_NAMES
    text=Path("fredis_workbench/src/workbench/mol-descriptors.js").read_text()
    assert tuple(re.findall(r"'([A-Za-z0-9]+)'",text))==SUPPORTED_DESCRIPTOR_NAMES
