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
from fredis.ml.specgen.fragment_spectrum_model import SpectrumLossSettings,normalized_attention_pool,balanced_binary_loss
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
    assert output.spectrum_batch.tree_graph.num_graphs==1
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
        pool_dominance=torch.zeros((1,4,4),dtype=torch.bool),
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


def test_physical_candidates_preserve_equivalent_explanations_without_sum_bias():
    _,data=prepared_two_ce();spectrum_batch=data.spectrum_batch
    counts=spectrum_batch.physical_candidate_explanation_ptr[1:]-spectrum_batch.physical_candidate_explanation_ptr[:-1]
    assert torch.any(counts>1)
    candidate=int((counts>1).nonzero()[0]);start,stop=spectrum_batch.physical_candidate_explanation_ptr[candidate:candidate+2]
    pairs=set(zip(spectrum_batch.explanation_adduct_index[start:stop].tolist(),spectrum_batch.explanation_hydrogen_shift[start:stop].tolist()))
    assert len(pairs)==int(counts[candidate])
    values=torch.tensor([[2.,-1.],[2.,-1.]])
    attention=torch.nn.Linear(2,1,bias=False);torch.nn.init.zeros_(attention.weight)
    one=normalized_attention_pool(values[:1],torch.tensor([0]),1,attention)
    duplicate=normalized_attention_pool(values,torch.tensor([0,0]),1,attention)
    torch.testing.assert_close(one,duplicate)


def test_balanced_fragment_and_state_bce_exact_values_and_empty_classes():
    logits=torch.tensor([0.,math.log(3.),-math.log(3.)])
    labels=torch.tensor([True,True,False]);weights=torch.tensor([.25,1.,0.])
    loss,positive,negative=balanced_binary_loss(logits,labels,weights)
    expected_positive=(.25*math.log(2)+math.log(4/3))/1.25
    assert positive.item()==pytest.approx(expected_positive)
    assert negative.item()==pytest.approx(math.log(4/3))
    assert loss.item()==pytest.approx(.5*(expected_positive+math.log(4/3)))
    positive_only=balanced_binary_loss(logits[:2],torch.ones(2,dtype=torch.bool),weights[:2])
    assert positive_only[0].item()==pytest.approx(positive_only[1].item())
    negative_only=balanced_binary_loss(logits[:2],torch.zeros(2,dtype=torch.bool))[0]
    assert torch.isfinite(negative_only)


def test_prepared_fragment_and_state_labels_keep_peak_weights_and_negative_states():
    _,data=prepared_two_ce();spectrum_batch=data.spectrum_batch
    assert spectrum_batch.fragment_supervision_node_index.shape==spectrum_batch.fragment_is_positive.shape
    for sample in range(data.num_samples):
        rows=spectrum_batch.fragment_supervision_sample_index==sample
        observed=spectrum_batch.fragment_positive_weight[rows][spectrum_batch.fragment_is_positive[rows]]
        assert set(observed.tolist())=={.5,1.}
    counts=spectrum_batch.physical_candidate_explanation_ptr[1:]-spectrum_batch.physical_candidate_explanation_ptr[:-1]
    owners=torch.repeat_interleave(torch.arange(counts.numel()),counts)
    negative_fragment_nodes=set(spectrum_batch.fragment_supervision_node_index[~spectrum_batch.fragment_is_positive].tolist())
    negative_state_nodes=spectrum_batch.physical_candidate_node_index[owners[~spectrum_batch.explanation_is_positive]]
    assert negative_fragment_nodes.intersection(negative_state_nodes.tolist())
    assert spectrum_batch.explanation_positive_weight[spectrum_batch.explanation_is_positive].max().item()==1.


def test_repeated_peak_matches_use_maximum_relative_intensity_weight():
    from fredis.libs.mmkit.mmkit import Formula
    generator=create_spectrum_generator(config());source=Compound.from_smiles('CCO');adduct=Adduct.parse('[M+H]+')
    mz=Formula.parse('C2H7O+').exact_mass
    data,_=ReactionStructureBuilder(generator).build(source,[adduct],[20.],[[mz,mz]],[[.2,1.]])
    state_weights=data.spectrum_batch.explanation_positive_weight[data.spectrum_batch.explanation_is_positive]
    fragment_weights=data.spectrum_batch.fragment_positive_weight[data.spectrum_batch.fragment_is_positive]
    assert state_weights.numel() and fragment_weights.numel()
    assert torch.all(state_weights==1.) and torch.all(fragment_weights==1.)


def test_physical_probability_is_fragment_times_state_and_all_heads_receive_gradients():
    generator,data=prepared_two_ce()
    trainer=ReactionFragmentTreeTrainingModel(generator.feature_model,spectrum_model=generator.spectrum_model)
    captured={}
    fragment_hook=generator.spectrum_model.fragment_score.register_forward_hook(lambda _m,_i,value:captured.setdefault('fragment',value))
    output=trainer(data);fragment_hook.remove();spectrum_batch=output.spectrum_output
    torch.testing.assert_close(spectrum_batch.log_physical_probability,
        spectrum_batch.fragment_probability.log()+spectrum_batch.state_probability.log())
    torch.testing.assert_close(spectrum_batch.ion_probability,
        spectrum_batch.fragment_probability*spectrum_batch.state_probability)
    counts=data.spectrum_batch.physical_candidate_explanation_ptr[1:]-data.spectrum_batch.physical_candidate_explanation_ptr[:-1]
    owners=torch.repeat_interleave(torch.arange(counts.numel()),counts)
    per_state_fragment_log_probability=spectrum_batch.fragment_probability[owners].log()
    torch.testing.assert_close(spectrum_batch.state_log_physical_probability,
        per_state_fragment_log_probability+torch.nn.functional.logsigmoid(spectrum_batch.state_logit))
    torch.testing.assert_close(spectrum_batch.state_log_physical_probability.exp(),
        spectrum_batch.fragment_probability[owners]*spectrum_batch.state_logit.sigmoid())
    expected=(SpectrumLossSettings().fragment_loss_weight*spectrum_batch.metrics['fragment_classification_loss']+
              SpectrumLossSettings().state_loss_weight*spectrum_batch.metrics['state_classification_loss']+
              SpectrumLossSettings().intensity_loss_weight*spectrum_batch.metrics['intensity_loss'])
    torch.testing.assert_close(spectrum_batch.loss,expected)
    powered_fragment=balanced_binary_loss(captured['fragment'].squeeze(-1)[data.spectrum_batch.fragment_supervision_node_index],
        data.spectrum_batch.fragment_is_positive,data.spectrum_batch.fragment_positive_weight.pow(SpectrumLossSettings().intensity_power))[0]
    powered_state=balanced_binary_loss(spectrum_batch.state_logit,data.spectrum_batch.explanation_is_positive,
        data.spectrum_batch.explanation_positive_weight.pow(SpectrumLossSettings().intensity_power))[0]
    torch.testing.assert_close(spectrum_batch.metrics['fragment_classification_loss'],powered_fragment)
    torch.testing.assert_close(spectrum_batch.metrics['state_classification_loss'],powered_state)
    output.loss.backward()
    for head in (generator.spectrum_model.fragment_score,generator.spectrum_model.adduct_score,
                 generator.spectrum_model.hydrogen_shift_score,generator.spectrum_model.intensity_head[-1]):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all() and torch.any(head.weight.grad!=0)
    required={'fragment_classification_loss','fragment_positive_loss','fragment_negative_loss',
              'fragment_positive_count','fragment_negative_count','fragment_positive_probability_mean',
              'fragment_negative_probability_mean','state_classification_loss','state_positive_loss',
              'state_negative_loss','state_positive_count','state_negative_count',
              'state_positive_probability_mean','state_negative_probability_mean','fragment_precision',
              'fragment_recall','fragment_f1','state_precision','state_recall','state_f1',
              'mean_physical_probability','intensity_loss','intensity_mae','full_spectrum_cosine',
              'precursor_free_cosine'}
    assert required<=spectrum_batch.metrics.keys()


def test_shared_hydrogen_shift_range_is_used_for_every_adduct():
    generator,_=prepared_two_ce()
    rule_set=generator.fragmenter.adduct_rule_set
    expected=set(range(rule_set.hydrogen_shift_min,rule_set.hydrogen_shift_max+1))
    tree=generator.fragmenter.build_fragment_ion_tree(Compound.from_smiles('CCO'))
    for adduct_type in rule_set.adduct_types:
        assert set(tree.get_formula_candidate_group(adduct_type).hydrogen_shifts.tolist())==expected


def test_multiple_physical_ions_and_training_step_use_no_chemistry(tmp_path):
    generator,data=prepared_two_ce();spectrum_batch=data.spectrum_batch
    per_node=torch.bincount(spectrum_batch.physical_candidate_node_index)
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
    assert {'branch_positive_mil_loss','branch_negative_loss','num_physical_ion_candidates','num_ion_explanations'}<=result.metrics.keys()


def test_prepared_payload_has_explicit_fredis_schema_version(tmp_path):
    _,data=prepared_two_ce();path=tmp_path/'data.preft.pt';data.save(path)
    payload=torch.load(path,weights_only=False)
    assert set(payload)=={'schema','schema_version','structure'}
    assert payload['schema']=='fredis.fragment-tree-training'
    assert payload['schema_version']==2
    loaded=SourceReactionStructure.load(path)
    assert loaded.num_branch_groups==1
