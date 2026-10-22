"""Inspect canonical Compound graphs, branch labels, and CE-specific peak assignments."""

from __future__ import annotations
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import torch
from fredis.analysis.prepared_result import (
    restore_reaction_registry,
    find_fragmenter_params,
    state_reaction_combinations,
)
from fredis.ml.input.source_reaction_structure import rows


def _display_ion_adducts(match):
    from fredis.libs.mmkit.mmkit import Adduct

    result = dict(match)
    base = match.get("fragmentIonAdduct")
    if base:
        adduct = Adduct.parse(base)
        result["fragmentIonAdduct"] = str(adduct)
        result["adduct"] = str(
            adduct.add_prefer_self(
                Adduct.from_dict({"H": int(match.get("hydrogenShift", 0))})
            )
        )
    elif match.get("adduct"):
        result["adduct"] = str(Adduct.parse(match["adduct"]))
    return result


def _source_svg(smiles):
    from fredis.libs.mmkit.mmkit import Compound
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DSVG(620, 360)
    drawer.DrawMolecule(Compound.from_smiles(smiles).mapped_mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def inspect_structure(file_path):
    payload = torch.load(file_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "structure" not in payload:
        raise TypeError("This file has no saved structure")
    data = payload["structure"]
    data.validate()
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("reaction_sequence_compounds", None)
    source_smiles = str(metadata.get("smiles") or data.source_smiles[0])
    params = find_fragmenter_params(file_path, metadata)
    registry = (
        restore_reaction_registry(data, source_smiles, params)
        if params
        else [
            dict(
                id=i,
                reactionPatternId=p,
                reactionRuleId=r,
                productMoleculeId=m,
                sourceAtomMaps=[],
                retainedAtomMaps=[],
                discardedAtomMaps=[],
                cutBondMaps=[],
                changedBondMaps=[],
                reactionPatternName="",
                reactantSmarts="",
            )
            for i, (p, r, m) in enumerate(data.reactions.reaction_type.tolist())
        ]
    )
    combinations = state_reaction_combinations(data)
    graph = data.compound_graph
    ctx = data.reaction_contexts
    groups = []
    names = {1: "positive", 0: "weak_negative", -1: "ignored"}
    for group, edge_ids in enumerate(rows(ctx.edge_ptr, ctx.edge_index)):
        root = int(
            data.source_compounds.compound_index[ctx.root_source_compound_index[group]]
        )
        nodes = {root}
        edges = []
        positive = {}
        for j, eid in enumerate(edge_ids):
            src, dst, r = (
                int(graph.source_compound_index[eid]),
                int(graph.target_compound_index[eid]),
                int(graph.reaction_index[eid]),
            )
            label = (
                int(ctx.edge_label[int(ctx.edge_ptr[group]) + j])
                if ctx.edge_label is not None
                else -1
            )
            nodes.update((src, dst))
            if label == 1:
                positive.setdefault(src, []).append(r)
            transitions = []
            for before in combinations.get((group, src), []):
                after = tuple(sorted((*before, r)))
                if r not in before and after in combinations.get((group, dst), []):
                    transitions.append(
                        dict(
                            addedReaction=r,
                            parentReactions=list(before),
                            targetReactions=list(after),
                        )
                    )
            edges.append(
                dict(
                    id=eid,
                    source=src,
                    target=dst,
                    reactionIndex=r,
                    label=label,
                    labelName=names[label],
                    transitions=transitions,
                )
            )
        node_rows = [
            dict(
                id=cid,
                compoundIndex=cid,
                smiles=data.compounds.smiles[cid],
                precursor=cid == root,
                observed=False,
                reactionSets=[list(r) for r in combinations.get((group, cid), [])],
                positiveBranchReactions=sorted(set(positive.get(cid, []))),
            )
            for cid in sorted(nodes)
        ]
        main = data.adducts.main_adducts[
            int(data.branch_groups.main_adduct_index[group])
        ]
        groups.append(
            dict(
                id=group,
                mainAdduct=main,
                sourceCompoundIndex=int(ctx.root_source_compound_index[group]),
                rootCompoundIndex=root,
                nodes=node_rows,
                edges=edges,
                nodeCount=len(nodes),
                edgeCount=len(edges),
                edgeTransitionCount=sum(len(e["transitions"]) for e in edges),
                nodeIdSource="CompoundBatch row ID",
                positiveEdges=sum(e["label"] == 1 for e in edges),
                weakNegativeEdges=sum(e["label"] == 0 for e in edges),
                ignoredEdges=sum(e["label"] == -1 for e in edges),
            )
        )
    samples = []
    for i, annotation in enumerate(data.sample_annotations):
        group = int(data.sample_branch_group_index[i])
        view = groups[group]
        detected = {
            m["compoundIndex"]
            for peak in annotation.get("peaks", [])
            for m in peak.get("matches", [])
        }
        peaks = [
            dict(peak, matches=[_display_ion_adducts(m) for m in peak["matches"]])
            for peak in annotation.get("peaks", [])
        ]
        samples.append(
            dict(
                annotation,
                **{k: v for k, v in view.items() if k not in ("id", "nodes")},
                id=i,
                branchGroup=group,
                collisionEnergyDisplay=str(
                    Decimal(str(annotation["collisionEnergy"])).quantize(
                        Decimal("0.1"), rounding=ROUND_HALF_UP
                    )
                ),
                adductIndex=int(data.branch_groups.main_adduct_index[group]),
                peaks=peaks,
                nodes=[
                    dict(
                        node,
                        observed=node["id"] in detected,
                        observedIntensity=max(
                            (
                                peak["intensity"]
                                for peak in peaks
                                if any(
                                    m["compoundIndex"] == node["id"]
                                    for m in peak["matches"]
                                )
                            ),
                            default=0.0,
                        ),
                    )
                    for node in view["nodes"]
                ],
            )
        )
    return dict(
        file=str(file_path),
        metadata=metadata,
        sourceSmiles=source_smiles,
        sourceSvg=_source_svg(source_smiles),
        fragmenterParams=params,
        reactions=registry,
        branchGroups=groups,
        samples=samples,
        summary=dict(
            samples=data.num_samples,
            nodes=len(data.compounds.compounds),
            edges=graph.reaction_index.numel(),
            edgeTransitions=sum(g["edgeTransitionCount"] for g in groups),
            primitiveReactions=data.reactions.reaction_type.shape[0],
            labelledEdges=sum(
                g["positiveEdges"] + g["weakNegativeEdges"] for g in groups
            ),
            positiveBranches=sum(g["positiveEdges"] for g in groups),
            branchGroups=data.num_branch_groups,
            ionStates=data.spectrum_explanations.formula_explanation_index.numel(),
            maxHistoryReactionCount=max(
                (len(r) for rs in combinations.values() for r in rs), default=0
            ),
            transitions=sum(g["edgeCount"] for g in groups),
        ),
    )


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        print(
            json.dumps(
                dict(
                    ok=True, result=inspect_structure(Path(request["path"]).resolve())
                ),
                ensure_ascii=False,
            )
        )
    except Exception as error:
        print(
            json.dumps(
                dict(ok=False, error=f"{type(error).__name__}: {error}"),
                ensure_ascii=False,
            )
        )
