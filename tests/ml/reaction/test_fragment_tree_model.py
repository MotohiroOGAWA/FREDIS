import math
from unittest.mock import patch
import pytest
import torch

from fredis.libs.mmkit.mmkit import Compound,Adduct
from fredis.domain.fragment.reaction import ReactionSequence
from fredis.domain.fragment.reaction.ReactionRelations import ReactionRelations
from fredis.domain.fragment.ion_tree import FragmentIonTree
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.input.source_reaction_structure import SourceReactionStructure
from fredis.ml.input.source_reaction_structure import prepare_source_reactions
from fredis.ml.mol.graph_builder import MolGraphBuilder
from fredis.ml.specgen import materialization
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.specgen.fragment_spectrum_model import SpectrumLossSettings
from fredis.ml.specgen.components.reaction.reaction_compatibility import ReactionCompatibilityEngine
from fredis.ml.training.fragment_tree_training.model import ReactionFragmentTreeTrainingModel,smooth_max_mil
from .TestReactionPipeline import config


def prepared_two_ce():
    generator=create_spectrum_generator(config())
    source=Compound.from_smiles('CC(O)N');adduct=Adduct.parse('[M+H]+')
    data,_=ReactionStructureBuilder(generator).build(source,[adduct,adduct],[10.,40.],
        [[18.033826,44.049476],[18.033826,44.049476]],[[.5,1.],[1.,.5]])
    return generator,data


def test_ce_spectra_share_one_branch_group_and_tree():
    generator,data=prepared_two_ce()
    assert data.num_samples==2 and data.num_branch_groups==1
    assert data.sample_branch_group_index.tolist()==[0,0]
    assert len(data.spectrum_batch.decoded.trees)==1
    generator.eval();source=Compound.from_smiles('CC(O)N');adduct=Adduct.parse('[M+H]+')
    output=generator.predict([source,source],[adduct,adduct],[10.,40.])
    assert len(output.fragments.trees)==1
    assert len(output.spectrum_batch.decoded.trees)==1
    first=output.spectra.intensity[output.spectra.sample_index==0]
    second=output.spectra.intensity[output.spectra.sample_index==1]
    assert first.shape==second.shape and not torch.equal(first,second)


def test_log_path_probability_and_zero_threshold():
    scores=torch.logit(torch.tensor([.8,.5]))
    log_probability=torch.nn.functional.logsigmoid(scores).sum()
    assert log_probability.item()==pytest.approx(math.log(.4))
    generator=create_spectrum_generator(config())
    assert generator.feature_model.decoder.branch_log_threshold==-math.inf


def test_candidate_mask_requires_mutual_reaction_center_retention():
    engine=ReactionCompatibilityEngine(max_reaction_count=3)
    # A is selected. B preserves A and is preserved by A. C is invalidated by
    # A; D invalidates A. Because reactions are simultaneous, both directions
    # make the composite invalid rather than establishing an execution order.
    invalidation=torch.zeros((1,4,4),dtype=torch.bool)
    invalidation[0,0,2]=True
    invalidation[0,3,0]=True
    expansion=engine.expand(
        state_reaction_index=torch.tensor([[0,-1,-1]]),state_sample_index=torch.tensor([0]),
        pool_valid=torch.ones((1,4),dtype=torch.bool),
        pool_conflict=torch.zeros((1,4,4),dtype=torch.bool),
        pool_invalidation=invalidation,
        pool_retained=torch.full((1,4,1),15,dtype=torch.long),
        source_atom_valid=torch.full((1,1),15,dtype=torch.long))
    assert expansion.valid.tolist()==[False,True,False,False]


def test_positive_mil_is_normalized_and_smooth():
    paths=torch.tensor([math.log(.8),math.log(.4)])
    value=smooth_max_mil(paths,torch.tensor([0,2]),temperature=.1)
    expected=.1*(torch.logsumexp(paths/.1,0)-math.log(2))
    torch.testing.assert_close(value,expected[None])


def test_multi_depth_states_and_cross_ce_negative_protection():
    generator=create_spectrum_generator(config())
    source=Compound.from_smiles('CC(O)N')
    reactions=generator.fragmenter.fragment_ion_tree_builder.create_reaction_instances(source)
    relations=ReactionRelations.from_reactions(reactions)
    pair=None
    for first in range(len(reactions)):
        for second in range(len(reactions)):
            if first==second or relations.rejection((first,second)):continue
            sequence=ReactionSequence((reactions[first],reactions[second]))
            if sequence.retained_atom_maps and len(sequence.reactions)==2:
                try:
                    if len(sequence.compile(source).run(source))==1:
                        pair=(first,second,sequence);break
                except Exception:pass
        if pair:break
    assert pair is not None
    first,second,sequence=pair
    prefix=ReactionSequence((reactions[first],))
    shallow=((None,None),(prefix,reactions[first]))
    deep=((*shallow,(sequence,reactions[second])))
    data=prepare_source_reactions(source=source,reactions=reactions,graph_builder=generator.mol_encoder.graph_builder,
        condition_features=torch.tensor([[0.,10.],[0.,40.]]),max_reaction_count=3,
        sample_branch_group_index=torch.tensor([0,0]),branch_group_adduct_index=torch.tensor([0]),
        teacher_paths=[[[shallow]],[[deep]]])
    assert {0,1,2}<=set(data.teacher_node_reaction_count.tolist())
    prefix_row=next(row for row,depth in enumerate(data.teacher_node_reaction_count.tolist()) if depth==1)
    start,stop=data.weak_negative_reaction_ptr[prefix_row:prefix_row+2]
    assert second not in data.weak_negative_reaction_index[start:stop].tolist()
    assert torch.all(data.state_transition_next_state_index>=0)


def test_fragment_node_budget_is_shared_across_collision_energies():
    value=config();value['generation_params']['max_fragment_nodes']=2
    generator=create_spectrum_generator(value).eval();source=Compound.from_smiles('CC(O)N');adduct=Adduct.parse('[M+H]+')
    outputs=list(generator.predict_batches([source,source],[adduct,adduct],[10.,40.],max_samples=1))
    assert len(outputs)==1
    output=outputs[0]
    assert len(output.fragments.trees)==1
    assert output.selection.decoded.state_sample_index.numel()<=2


def test_condition_states_keep_equivalent_formulas_separate_until_output():
    generator, data = prepared_two_ce()
    batch = data.spectrum_batch
    rows = ((batch.state_sample_index == 0)
            & (batch.state_node_index == 0)).nonzero().flatten()
    conditions = generator.spectrum_model.main_adduct_conditions['[M+H]+']
    assert rows.numel() == len(conditions)
    pairs = list(zip(
        (batch.ion_type_strs[int(i)] for i in batch.state_adduct_index[rows]),
        batch.state_hydrogen_shift[rows].tolist(),
    ))
    assert pairs == list(conditions)
    assert torch.unique(batch.state_formula_tensor[rows], dim=0).shape[0] < rows.numel()

def test_shared_hydrogen_shift_range_is_used_for_every_adduct():
    generator,_=prepared_two_ce()
    rule_set=generator.fragmenter.adduct_rule_set
    expected=set(range(rule_set.hydrogen_shift_min,rule_set.hydrogen_shift_max+1))
    tree=generator.fragmenter.build_fragment_ion_tree(Compound.from_smiles('CCO'))
    for adduct_type in rule_set.adduct_types:
        assert set(tree.get_formula_candidate_group(adduct_type).hydrogen_shifts.tolist())==expected


def test_multiple_physical_ions_and_training_step_use_no_chemistry(tmp_path):
    generator,data=prepared_two_ce();spectrum_batch=data.spectrum_batch
    per_node=torch.bincount(spectrum_batch.state_node_index)
    assert int(per_node.max())>1
    path=tmp_path/'strict.preft.pt';data.save(path);data=SourceReactionStructure.load(path)
    trainer=ReactionFragmentTreeTrainingModel(generator.feature_model,spectrum_model=generator.spectrum_model)
    optimizer=torch.optim.Adam(trainer.parameters(),lr=1e-4)
    with patch.object(Compound,'from_smiles',side_effect=AssertionError('RDKit in training')), \
         patch.object(ReactionSequence,'compile',side_effect=AssertionError('fragment materialization in training')), \
         patch.object(FragmentIonTree,'from_fragment_tree',side_effect=AssertionError('ion chemistry in training')), \
         patch.object(MolGraphBuilder,'build',side_effect=AssertionError('graph construction in training')), \
         patch.object(materialization,'materialize_reaction_states',side_effect=AssertionError('fragment materialization in training')):
        optimizer.zero_grad();result=trainer(data);result.loss.backward();optimizer.step()
    assert torch.isfinite(result.loss)
    assert {'branch_positive_mil_loss','branch_negative_loss','num_ion_states'}<=result.metrics.keys()


def test_prepared_payload_has_schema_without_version(tmp_path):
    _,data=prepared_two_ce();path=tmp_path/'data.preft.pt';data.save(path)
    payload=torch.load(path,weights_only=False)
    assert set(payload)=={'schema','structure'}
    assert payload['schema']=='fredis.fragment-tree-training'
    loaded=SourceReactionStructure.load(path)
    assert loaded.num_branch_groups==1
