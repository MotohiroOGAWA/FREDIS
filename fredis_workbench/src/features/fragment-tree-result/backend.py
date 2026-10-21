"""Backend for inspecting prepared branch-group/physical-ion structures."""
from __future__ import annotations

import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import torch
from fredis.analysis.prepared_result import restore_reaction_registry


def _display_ion_adducts(match: dict) -> dict:
    """Render saved explanations with the current Adduct formatter.

    The fragment-ion adduct precedes the neutral hydrogen shift. Recompute
    the combined name from those two components, including cancellation of H.
    """
    from fredis.libs.mmkit.mmkit import Adduct
    result = dict(match)
    base = match.get("fragmentIonAdduct")
    if base:
        adduct = Adduct.parse(base)
        result["fragmentIonAdduct"] = str(adduct)
        shift = match.get("hydrogenShift")
        if shift is not None:
            result["adduct"] = str(adduct.add_prefer_self(Adduct.from_dict({"H": int(shift)})))
        elif match.get("adduct"):
            result["adduct"] = str(Adduct.parse(match["adduct"]))
    elif match.get("adduct"):
        # Old annotations may lack the decomposition; do not guess a base.
        result["adduct"] = str(Adduct.parse(match["adduct"]))
    return result


def _teacher_states(structure) -> list[dict]:
    """One row per (sample, normalized reaction-set) teacher state, with its
    positive branching reactions and, if materialized, the resulting fragment SMILES."""
    state_reaction_ptr = structure.teacher_node_reaction_ptr.tolist()
    state_reaction_index = structure.teacher_node_reaction_index.tolist()
    positive_ptr = structure.teacher_positive_reaction_ptr.tolist()
    positive_index = structure.teacher_positive_reaction_index.tolist()
    eos = structure.teacher_node_observed.tolist()
    sample_index = structure.teacher_node_branch_group_index.tolist()
    fragment_node_index = structure.state_fragment_node_index.tolist()
    compounds = structure.spectrum_batch.decoded.compounds if structure.spectrum_batch is not None else None

    rows = []
    for row in range(len(sample_index)):
        reactions = state_reaction_index[state_reaction_ptr[row]:state_reaction_ptr[row + 1]]
        next_reactions = positive_index[positive_ptr[row]:positive_ptr[row + 1]]
        node = fragment_node_index[row]
        rows.append({
            "branchGroup": sample_index[row],
            "reactions": reactions,
            "terminal": not next_reactions,
            "observed": bool(eos[row]),
            "reactionCount": int(structure.teacher_node_reaction_count[row]),
            "fragmentSmiles": str(compounds[node].smiles) if compounds is not None and node >= 0 else None,
            "positiveBranchReactions": next_reactions,
        })
    return rows


def _transitions(structure) -> list[dict]:
    parents = structure.transition_parent_state_index.tolist()
    children = structure.transition_child_state_index.tolist()
    added = structure.transition_added_reaction_index.tolist()
    return [
        {"parent": parent, "child": child, "addedReaction": reaction}
        for parent, child, reaction in zip(parents, children, added)
    ]


def _node_view(structure, samples, source_smiles, file_path):
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from fredis.libs.mmkit.mmkit import Compound
    source=Compound.from_smiles(source_smiles)
    mol=source.mapped_mol
    drawer=rdMolDraw2D.MolDraw2DSVG(620,360)
    drawer.DrawMolecule(mol);drawer.FinishDrawing()
    registry=[];original_reactions=[];params={};reaction_patterns={}
    for parent in file_path.parents:
        config=parent/'preparation_config.json'
        if config.exists():
            params=json.loads(config.read_text());break
    fragmenter_params=params.get('chemistry_config',{}).get('fragmenter_params')
    if fragmenter_params:
        from fredis.domain.fragment.fragmenter import Fragmenter
        fragmenter=Fragmenter.from_dict(fragmenter_params)
        builder=fragmenter.fragment_ion_tree_builder
        original_reactions=list(builder.create_reaction_instances(source))
        reaction_patterns={pattern.pattern_id:pattern for pattern in builder.reaction_pattern_set}
        if len(original_reactions)!=len(structure.reaction_type): original_reactions=[]
        registry=restore_reaction_registry(structure,source_smiles,fragmenter_params)
    def reaction_id(reaction):
        if reaction is None:return None
        if original_reactions:
            try:return original_reactions.index(reaction)
            except ValueError:pass
        for entry in registry:
            if (entry['reactionPatternId'],entry['reactionRuleId'],entry['productMoleculeId'])==(reaction.reaction_pattern_id,reaction.reaction_rule_id,reaction.product_molecule_id) and entry['sourceAtomMaps']==list(reaction.source_atom_maps):return entry['id']
        raise ValueError('Saved transition cannot be matched to a reaction registry entry.')
    decoded=structure.spectrum_batch.decoded if structure.spectrum_batch is not None else None
    offset=0
    totals=dict(nodes=0,edges=0,edgeTransitions=0)
    if decoded:
        for sample,tree in zip(samples,decoded.trees):
            tree_nodes=[tree.get_node(index) for index in range(tree.num_nodes)]
            ids=[node.index for node in tree_nodes]
            states_for_node={}
            for state in sample['states']:
                global_node=int(structure.state_fragment_node_index[state['id']].item())
                states_for_node.setdefault(global_node,[]).append(state)
            nodes=[]
            for node in tree_nodes:
                histories=[];eos=False;next_reactions=set()
                for state in states_for_node.get(offset+node.index,[]):
                    if state['reactions'] not in histories:histories.append(state['reactions'])
                    eos=eos or state['observed'];next_reactions.update(state['positiveBranchReactions'])
                precursor=node.index==0
                nodes.append(dict(id=ids[node.index],smiles=node.smiles,reactionSets=histories,precursor=precursor,observed=eos,positiveBranchReactions=sorted(next_reactions)))
            edges=[]
            for edge in (tree.get_edge(index) for index in range(tree.num_edges)):
                transitions=[]
                for transition in edge.transitions:
                    transitions.append(dict(addedReaction=reaction_id(transition.added_reaction),
                        parentReactions=[reaction_id(reaction) for reaction in transition.parent_reaction_sequence.reactions] if transition.parent_reaction_sequence else [],
                        targetReactions=[reaction_id(reaction) for reaction in transition.reaction_sequence.reactions]))
                edges.append(dict(id=edge.index,source=ids[edge.source_index],target=ids[edge.target_index],transitions=transitions))
            for peak in sample.get('peaks',[]):
                for match in peak['matches']:
                    match['nodeIds']=[ids[index-offset] for index in match.get('nodeIndices',[]) if offset<=index<offset+len(nodes)]
            sample.update(nodes=nodes,edges=edges,nodeIdSource='local fragment node index',
                          nodeCount=len(nodes),edgeCount=len(edges),edgeTransitionCount=sum(len(edge['transitions']) for edge in edges))
            totals['nodes']+=len(nodes);totals['edges']+=len(edges);totals['edgeTransitions']+=sample['edgeTransitionCount']
            offset+=len(nodes)
    return registry,drawer.GetDrawingText(),totals,fragmenter_params


def inspect_structure(file_path: Path) -> dict:
    payload = torch.load(file_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "structure" not in payload:
        raise TypeError("This .preft.pt file has no saved structure.")
    structure = payload["structure"]
    metadata = dict(payload.get("metadata") or {})
    # Incremental preparation seeds contain ReactionSequence objects. They are
    # persisted for reuse by preparation, but are not part of the viewer data.
    metadata.pop("reaction_sequence_compounds", None)
    required = ("teacher_node_reaction_ptr", "sample_branch_group_index", "reaction_type")
    if not all(hasattr(structure, name) for name in required):
        raise TypeError("This .preft.pt file is not a prepared fragment-tree training structure.")

    source_smiles = str(metadata.get("smiles", ""))
    conditions = structure.condition_features.tolist()
    states = _teacher_states(structure)
    transitions = _transitions(structure)

    states_by_sample: dict[int, list[dict]] = {}
    for local_id, state in enumerate(states):
        state = {**state, "id": local_id}
        states_by_sample.setdefault(state["branchGroup"], []).append(state)
    transitions_by_sample: dict[int, list[dict]] = {}
    for transition in transitions:
        sample = states[transition["parent"]]["branchGroup"] if transition["parent"] < len(states) else None
        if sample is not None:
            transitions_by_sample.setdefault(sample, []).append(transition)

    num_samples = int(structure.num_samples)
    annotations=getattr(structure,'sample_annotations',()) or metadata.get('sample_annotations',[])
    samples = []
    sample_groups=structure.sample_branch_group_index.tolist()
    for sample_id in range(num_samples):
        adduct_index, collision_energy = (conditions[sample_id] if sample_id < len(conditions) else (None, None))
        if sample_id<len(annotations): collision_energy=annotations[sample_id]["collisionEnergy"]
        display_ce=str(Decimal(str(collision_energy if sample_id<len(annotations) else round(collision_energy,5))).quantize(Decimal("0.1"),rounding=ROUND_HALF_UP)) if collision_energy is not None else "Unavailable"
        samples.append({
            **(annotations[sample_id] if sample_id<len(annotations) else {}),
            "id": sample_id,
            "adductIndex": adduct_index,
            "collisionEnergy": collision_energy,
            "collisionEnergyDisplay":display_ce,
            "branchGroup": sample_groups[sample_id],
            "states": states_by_sample.get(sample_groups[sample_id], []),
            "transitions": transitions_by_sample.get(sample_groups[sample_id], []),
        })

    groups=[dict(id=group,states=states_by_sample.get(group,[]),transitions=transitions_by_sample.get(group,[]),peaks=[])
            for group in range(structure.num_branch_groups)]
    reactions,source_svg,totals,fragmenter_params=_node_view(structure,groups,source_smiles,file_path)
    group_offsets=[];offset=0
    for tree in structure.spectrum_batch.decoded.trees:
        group_offsets.append(offset);offset+=tree.num_nodes
    for sample in samples:
        view=groups[sample['branchGroup']]
        for key in ('nodes','edges','nodeCount','edgeCount','edgeTransitionCount','nodeIdSource'):
            if key in view:sample[key]=view[key]
        if 'peaks' in sample:
            sample['peaks'] = [{**peak, 'matches': [_display_ion_adducts(match) for match in peak.get('matches', [])]}
                               for peak in sample['peaks']]
        start=group_offsets[sample['branchGroup']];ids=[node['id'] for node in view.get('nodes',[])]
        for peak in sample.get('peaks',[]):
            for match in peak.get('matches',[]):
                match['nodeIds']=[ids[index-start] for index in match.get('nodeIndices',[]) if start<=index<start+len(ids)]
    if fragmenter_params:
        from fredis.domain.fragment.fragmenter import Fragmenter
        from fredis.libs.mmkit.mmkit import Adduct
        fragmenter=Fragmenter.from_dict(fragmenter_params)
        for sample in samples:
            if not sample.get('adduct'):
                for parent in file_path.parents:
                    config=parent/'preparation_config.json'
                    if not config.exists():continue
                    chemistry=json.loads(config.read_text()).get('chemistry_config',{})
                    types=chemistry.get('adduct_type_strs',[str(adduct) for adduct in fragmenter.adduct_types])
                    index=int(sample['adductIndex'])
                    if 0<=index<len(types):
                        sample['adduct']=types[index]
                        sample['mainAdduct']=str(fragmenter._resolve_main_adduct_type(Adduct.parse(types[index])))
                    break
    return {
        "reactions":reactions,"sourceSvg":source_svg,"fragmenterParams":fragmenter_params,
        "file": str(file_path),
        "metadata": metadata,
        "sourceSmiles": source_smiles,
        "summary": {
            **totals,
            "samples": num_samples,
            "primitiveReactions": int(structure.reaction_type.shape[0]),
            "teacherNodes": len(states),
            "positiveBranches": structure.transition_added_reaction_index.numel(),
            "branchGroups": int(structure.num_branch_groups),
            "ionStates": int(structure.spectrum_batch.state_node_index.numel()) if structure.spectrum_batch else 0,
            "maxTeacherReactionCount": int(structure.teacher_node_reaction_count.max()) if states else 0,
            "transitions": len(transitions),
        },
        "samples": samples,
    }


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        print(json.dumps({"ok": True, "result": inspect_structure(Path(request["path"]).resolve())}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
