"""Preparation-only reaction teachers from original spectra and Source chemistry."""
from __future__ import annotations
from dataclasses import replace
from collections.abc import Sequence
from types import SimpleNamespace
import torch
from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.domain.fragment.reaction import ReactionSequence, FragmentTreeLimitExceeded
from .source_reaction_structure import SourceReactionStructure, prepare_source_reactions, UnresolvedPrecursorError
from ..specgen.source_anchored_spectrum_predictor import SourceAnchoredFragmentSpectrumGenerator
from ..specgen.components.reaction.reaction_decoder import ReactionDecoderOutput
from ..specgen.materialization import materialize_reaction_states
from ..specgen.fragment_spectrum_model import prepare_spectrum_model_batch


def _transition_chains(transitions_by_sequence, sequence):
    """Every recorded transition chain from Source to one reaction combination.

    The combination is applied to Source at once; the chains only supply the
    one-reaction-at-a-time steps the branching decoder is trained on.
    """
    if sequence is None:
        return (((None,None),),)
    chains=[]
    for transition in transitions_by_sequence.get(sequence,()):
        for prefix in _transition_chains(transitions_by_sequence,transition.parent_reaction_sequence):
            chains.append((*prefix,(sequence,transition.added_reaction)))
    return tuple(dict.fromkeys(chains))


def _transitions_by_sequence(tree):
    result={}
    for edge_index in range(tree.num_edges):
        for transition in tree.get_edge(edge_index).transitions:
            result.setdefault(transition.reaction_sequence,[]).append(transition)
    return result


class ReactionStructureBuilder:
    def __init__(self, generator: SourceAnchoredFragmentSpectrumGenerator, *,
                 max_unique_fragment_smiles: int = -1, max_reaction_combinations: int = -1) -> None:
        self.generator = generator
        self.max_unique_fragment_smiles = max_unique_fragment_smiles
        self.max_reaction_combinations = max_reaction_combinations
        # Fragment-tree search statistics of the most recent build() call,
        # including one that stopped at a search limit or failed after the
        # tree was built; None when the tree search never ran.
        self.last_search_stats: dict[str, int] | None = None
        self.last_reaction_sequence_compounds = {}
        self.last_reactions = ()

    def build(self, source: Compound, precursor_types: Sequence[Adduct], collision_energy: Sequence[float],
              peaks_mz: Sequence[Sequence[float]], peaks_intensity: Sequence[Sequence[float]], *,
              seed_compounds: dict | None = None,
              addable_pattern_ids: frozenset[int] | None = None,
              precomputed_reactions: Sequence | None = None) -> tuple[SourceReactionStructure, tuple[int, ...]]:
        """Return the built structure plus which input sample indices it kept.

        A sample whose precursor ion is unreachable under the configured
        reaction patterns (e.g. an fragment-ion adduct the patterns cannot produce) is
        dropped rather than failing the whole group: real MS2 fragmentation
        never happens on an unreachable precursor, so such a sample has
        nothing to supervise. The caller uses the returned indices to report
        which of its original records were actually kept.
        """
        if not precursor_types or not len(precursor_types)==len(collision_energy)==len(peaks_mz)==len(peaks_intensity):
            raise ValueError("Every spectrum needs precursor, CE, peaks and intensities")
        for mz,intensity in zip(peaks_mz,peaks_intensity):
            if len(mz)!=len(intensity):
                raise ValueError("Peak m/z and intensity lengths differ")
        self.last_search_stats=None
        self.last_reaction_sequence_compounds={}
        self.last_reactions=()
        generator=self.generator
        fragmenter=generator.fragmenter
        # precomputed_reactions are the Source reactions this build may add: all
        # of them for a full build, or only the addable ones when seeded.
        if precomputed_reactions is not None:
            precomputed_reactions=tuple(sorted(set(precomputed_reactions),key=lambda reaction:reaction.key))
        if addable_pattern_ids is None:
            reactions=(precomputed_reactions if precomputed_reactions is not None else
                     fragmenter.fragment_ion_tree_builder.create_reaction_instances(source))
        else:
            added=(precomputed_reactions if precomputed_reactions is not None else
                   fragmenter.fragment_ion_tree_builder.create_reaction_instances(source,pattern_ids=addable_pattern_ids))
            reactions=tuple(sorted(set(added)|{reaction for sequence in (seed_compounds or {})
                                              for reaction in sequence.reactions},key=lambda reaction:reaction.key))
        self.last_reactions=tuple(reactions)
        # Full chemistry is permitted here, never inside neural forward.
        try:
            tree=fragmenter.build_fragment_ion_tree(source,max_unique_fragment_smiles=self.max_unique_fragment_smiles,
                max_reaction_combinations=self.max_reaction_combinations,
                seed_reaction_sequences=tuple(seed_compounds or ()), seed_compounds=seed_compounds,
                addable_pattern_ids=addable_pattern_ids,
                precomputed_reactions=precomputed_reactions)
        except FragmentTreeLimitExceeded as error:
            self.last_search_stats=error.stats
            raise
        self.last_search_stats=tree.search_stats
        self.last_reaction_sequence_compounds=tree.reaction_sequence_compounds
        # Resolve the observed precursor against the same already-built tree so
        # unreachable records can be dropped before teacher assignment. These
        # paths are validation metadata, not decoder seeds: branching starts at
        # Source and is conditioned by normalized main adduct.
        precursor_reachable=[bool(fragmenter.resolve_precursor_reactions(tree,precursor_type)) for precursor_type in precursor_types]
        kept=tuple(index for index,reachable in enumerate(precursor_reachable) if reachable)
        if not kept:
            raise UnresolvedPrecursorError("No sample in this group has a valid precursor reaction sequence from Source")
        if len(kept)!=len(precursor_types):
            precursor_types=[precursor_types[index] for index in kept]
            collision_energy=[collision_energy[index] for index in kept]
            peaks_mz=[peaks_mz[index] for index in kept]
            peaks_intensity=[peaks_intensity[index] for index in kept]
        assignments=fragmenter.annotate_peak_sets(tree,zip(precursor_types,peaks_mz))
        transitions_by_sequence=_transitions_by_sequence(tree)
        teacher_paths=[]
        for peak_annotations in assignments:
            peaks=[]
            for explanations in peak_annotations:
                paths=[]
                for sequence in dict.fromkeys(annotation.reaction_sequence for annotation in explanations):
                    paths.extend(_transition_chains(transitions_by_sequence,sequence))
                peaks.append(list(dict.fromkeys(paths)))
            teacher_paths.append(peaks)
        mains=[fragmenter._resolve_main_adduct_type(adduct) for adduct in precursor_types]
        conditions=torch.tensor([[generator.adduct_type_strs.index(str(main)),ce]
                                 for main,ce in zip(mains,collision_energy)],dtype=torch.float32)
        group_keys={};sample_groups=[]
        for main in mains:sample_groups.append(group_keys.setdefault(str(main),len(group_keys)))
        group_adduct=torch.tensor([generator.adduct_type_strs.index(key) for key in group_keys],dtype=torch.long)
        data=prepare_source_reactions(source=source,reactions=reactions,graph_builder=generator.mol_encoder.graph_builder,
            condition_features=conditions,max_reaction_count=fragmenter.tree_max_reaction_count,teacher_paths=teacher_paths,
            sample_branch_group_index=torch.tensor(sample_groups),branch_group_adduct_index=group_adduct,
            valid_reaction_sequences=frozenset(self.last_reaction_sequence_compounds))
        # Materialization adds its own Source bookkeeping root. Teacher depth
        # zero is the shared empty reaction state for each branch group.
        rows=[tuple(data.teacher_node_reaction_index[a:b].tolist()) for a,b in zip(data.teacher_node_reaction_ptr[:-1],data.teacher_node_reaction_ptr[1:])]
        samples=data.teacher_node_branch_group_index.tolist()
        dense_rows=[];dense_sample=[];parents=[];added=[];terminal=[];teacher_dense={}
        width=fragmenter.tree_max_reaction_count
        for sample in range(data.num_branch_groups):
            source_row=len(dense_rows)
            dense_rows.append(tuple([-1]*width));dense_sample.append(sample);parents.append(-1);added.append(-1);terminal.append(False)
            for node,(owner,ids) in enumerate(zip(samples,rows)):
                if owner!=sample:continue
                parent=int(data.teacher_node_parent_index[node])
                teacher_dense[node]=len(dense_rows)
                dense_rows.append((*ids,*([-1]*(width-len(ids)))))
                dense_sample.append(sample);parents.append(teacher_dense[parent] if parent>=0 else source_row)
                added.append(int(data.teacher_node_added_reaction_index[node]));terminal.append(True)
        synthetic=ReactionDecoderOutput(torch.tensor(dense_sample),torch.tensor(dense_rows,dtype=torch.long),
            torch.zeros(len(dense_rows)),torch.tensor(terminal),torch.tensor(parents),torch.tensor(added),
            SimpleNamespace(reaction_index=torch.arange(len(reactions)).expand(data.num_branch_groups,-1)))
        decoded=materialize_reaction_states(synthetic,(source,),(reactions,),data.branch_group_tree_index,generator.mol_encoder.graph_builder)
        node_by_state={(dense_sample[row],tuple(i for i in dense_rows[row] if i>=0)):node
                       for row,node in zip(decoded.materialized_state_index.tolist(),decoded.materialized_node_index.tolist())}
        state_nodes=torch.tensor([node_by_state.get((sample,tuple(row)),-1) for sample,row in zip(samples,rows)],dtype=torch.long)
        annotations=[]
        # (node, str(adduct)) -> maximum matched relative peak intensity.  This
        # chemistry/assignment work is prepared once here; the training loop
        # only consumes tensors and never reruns Fragmenter or RDKit.
        positive_node_adducts=[]
        for sample,(peak_annotations,adduct,energy,mzs,intensities) in enumerate(zip(assignments,precursor_types,collision_energy,peaks_mz,peaks_intensity)):
            branch_group=sample_groups[sample]
            main=fragmenter._resolve_main_adduct_type(adduct)
            precursor_mz=adduct.apply_to_formula(source.formula).normalized.exact_mass
            peaks=[]
            positive={}
            base_peak=max((max(0.,float(value)) for value in intensities),default=0.)
            for peak_index,(mz,intensity,explanations) in enumerate(zip(mzs,intensities,peak_annotations)):
                matches=[]
                for annotation in explanations:
                    sequence=annotation.reaction_sequence
                    state=(branch_group,tuple(sorted(reactions.index(reaction) for reaction in sequence.reactions)) if sequence else ())
                    if state not in node_by_state:continue
                    node=node_by_state[state]
                    relative=max(0.,float(intensity))/base_peak if base_peak>0 else 0.
                    key=(node,str(annotation.adduct));positive[key]=max(positive.get(key,0.),relative)
                    formula=annotation.formula
                    match=dict(smiles=annotation.smiles,
                               formula=str(formula),theoreticalMz=float(formula.exact_mass),
                               massErrorPpm=(float(mz)-formula.exact_mass)/formula.exact_mass*1e6,
                               adduct=str(annotation.adduct),fragmentIonAdduct=str(annotation.fragment_ion_adduct),
                               hydrogenShift=annotation.hydrogen_shift)
                    existing=next((item for item in matches if {k:v for k,v in item.items() if k!='nodeIndices'}==match),None)
                    if existing is None:matches.append(dict(nodeIndices=[node],**match))
                    elif node not in existing['nodeIndices']:existing['nodeIndices']=sorted((*existing['nodeIndices'],node))
                peaks.append(dict(index=peak_index,mz=float(mz),intensity=float(intensity),
                                  precursor=bool(fragmenter.mass_tolerance.within(float(mz),precursor_mz)),matches=matches))
            positive_node_adducts.append(positive)
            def score(selected):
                total=sum(peak['intensity'] for peak in selected)
                return sum(peak['intensity'] for peak in selected if peak['matches'])/total if total>0 else None
            annotations.append(dict(adduct=str(adduct),mainAdduct=str(main),collisionEnergy=float(energy),
                                    precursorMz=float(precursor_mz),peaks=peaks,
                                    assignmentScore=score(peaks),
                                    assignmentScoreWithoutPrecursor=score([peak for peak in peaks if not peak['precursor']])))
        spectrum_batch=prepare_spectrum_model_batch(decoded,fragmenter,precursor_types,generator.tensorizer,
            state_positive_weights=positive_node_adducts,sample_branch_group_index=data.sample_branch_group_index,
            ion_type_strs=getattr(generator,'ion_type_strs',None))
        target=[]
        for sample,mz in zip(spectrum_batch.physical_candidate_sample_index.tolist(),spectrum_batch.physical_candidate_mz.tolist()):
            matches=[float(intensity) for observed,intensity in zip(peaks_mz[sample],peaks_intensity[sample])
                     if fragmenter.mass_tolerance.within(observed,mz)]
            target.append(max(matches,default=0.))
        return replace(data,state_fragment_node_index=state_nodes,sample_annotations=tuple(annotations),
                       spectrum_batch=replace(spectrum_batch,target_intensity=torch.tensor(target))),kept
