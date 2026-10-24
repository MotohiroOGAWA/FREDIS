"""Backend for predicting a single MS/MS spectrum from SMILES + CE + adduct."""

from __future__ import annotations


import contextlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from fredis.domain.fragment.reaction.ReactionRelations import ReactionRelations

import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from fredis.libs.mmkit.mmkit import Adduct, Compound, Formula
from fredis.domain.mass.parse_ce import parse_ce_to_ev
from fredis.ml.specgen.predict_spectrum import load_generator


def _reaction_id(reactions, reaction) -> int | None:
    if reaction is None:
        return None
    try:
        return reactions.index(reaction)
    except ValueError as exc:
        raise ValueError(
            "Predicted tree transition references an unknown reaction."
        ) from exc


def _reaction_registry(generator, reactions) -> list[dict[str, Any]]:
    """Describe the same primitive Source reactions used during prediction."""
    patterns = {
        pattern.pattern_id: pattern
        for pattern in generator.fragmenter.fragment_ion_tree_builder.reaction_pattern_set
    }
    registry = []
    for number, reaction in enumerate(reactions):
        pattern = patterns.get(reaction.reaction_pattern_id)
        rule = (
            next(
                (
                    item
                    for item in pattern.reaction_rules
                    if item.id == reaction.reaction_rule_id
                ),
                None,
            )
            if pattern
            else None
        )
        registry.append(
            {
                "id": number,
                "reactionPatternId": reaction.reaction_pattern_id,
                "reactionPatternName": pattern.name if pattern else "",
                "reactionRuleId": reaction.reaction_rule_id,
                "reactionRuleName": rule.source_rule.name if rule else "",
                "reactantSmarts": pattern.reactant_smarts if pattern else "",
                "reactionRuleProductSmarts": rule.source_rule.smarts if rule else "",
                "productMoleculeId": reaction.product_molecule_id,
                "sourceAtomMaps": list(reaction.source_atom_maps),
                "retainedAtomMaps": sorted(reaction.retained_atom_maps),
                "discardedAtomMaps": sorted(reaction.discarded_atom_maps),
                "cutBondMaps": sorted(reaction.cut_bond_maps),
                "changedBondMaps": sorted(reaction.changed_bond_maps),
            }
        )
    return registry


def _build_tree(
    result, sample: int, tensorizer, threshold: float | None, reactions=()
) -> dict[str, Any]:
    """Fragment-tree nodes/edges explored for this prediction, annotated with
    per-node neutral formula/exact mass plus any predicted-peak ion states."""
    fragments = result.fragments
    core = result.spectrum_batch.canonical_data
    group = int(core.sample_branch_group_index[sample])
    execution = core.execution_graph
    root = int(
        core.source_compounds.compound_index[
            core.reaction_contexts.root_source_compound_index[group]
        ]
    )
    reaction_sets_by_node = {}
    for cid, sid in zip(
        execution.compound_index[execution.branch_group_index == group].tolist(),
        execution.reaction_set_index[execution.branch_group_index == group].tolist(),
    ):
        reaction_sets_by_node.setdefault(cid, []).append(core.reaction_sets.rows()[sid])
    node_offsets = [root] + sorted(set(reaction_sets_by_node) - {root})
    local_index = {cid: cid for cid in node_offsets}
    depth = {cid: min(map(len, sets)) for cid, sets in reaction_sets_by_node.items()}
    edges = []
    first, last = core.reaction_contexts.edge_ptr[group : group + 2]
    graph = core.compound_graph
    for eid in core.reaction_contexts.edge_index[first:last].tolist():
        src, dst, reaction = (
            int(graph.source_compound_index[eid]),
            int(graph.target_compound_index[eid]),
            int(graph.reaction_index[eid]),
        )
        transitions = []
        for before in reaction_sets_by_node.get(src, []):
            after = sorted(before + [reaction])
            if reaction not in before and after in reaction_sets_by_node.get(dst, []):
                transitions.append(
                    dict(
                        addedReaction=reaction,
                        parentReactions=before,
                        targetReactions=after,
                        seed=False,
                    )
                )
        if transitions:
            edges.append(
                dict(
                    id=eid,
                    source=src,
                    target=dst,
                    reactionIndex=reaction,
                    transitions=transitions,
                )
            )

    spectrum_batch = result.spectrum_batch
    spectra = result.spectra
    candidate_node_index = spectrum_batch.state_node_index.detach().cpu().tolist()
    ion_probability = spectra.ion_probability.detach().cpu().tolist()
    candidate_adduct = spectrum_batch.state_adducts
    candidate_sample_index = spectrum_batch.state_sample_index.detach().cpu().tolist()
    candidate_mz = spectrum_batch.state_mz.detach().cpu().tolist()
    candidate_formula_tensor = spectrum_batch.state_formula_tensor
    intensity = spectra.intensity.detach().cpu().tolist()

    # Apply the optional intensity filter consistently in the node inspector.
    annotations_by_node: dict[int, list[dict[str, Any]]] = {}
    # Each physical candidate row is its own formula.
    for entry, node in enumerate(candidate_node_index):
        if node not in local_index or candidate_sample_index[entry] != sample:
            continue
        probability = float(ion_probability[entry])
        if (
            (threshold is not None and intensity[entry] <= threshold)
            or not bool((candidate_formula_tensor[entry, :-1] >= 0).all())
            or candidate_mz[entry] <= 0
        ):
            continue
        annotations_by_node.setdefault(local_index[node], []).append(
            {
                "peakMz": float(candidate_mz[entry]),
                "peakIntensity": float(intensity[entry]),
                "adduct": (
                    candidate_adduct[entry] if entry < len(candidate_adduct) else ""
                ),
                "ionFormula": str(
                    tensorizer.tensor_to_formula(candidate_formula_tensor[entry])
                ),
                "probability": probability,
            }
        )

    nodes = []
    for global_index in node_offsets:
        compound = fragments.compounds[global_index]
        neutral_formula = compound.formula
        local = local_index[global_index]
        nodes.append(
            {
                "id": local,
                "smiles": compound.smiles,
                "depth": depth[local],
                "precursor": global_index == root,
                "compoundIndex": global_index,
                "formula": str(neutral_formula),
                "exactMass": neutral_formula.exact_mass,
                "annotations": annotations_by_node.get(local, []),
                "reactionSets": reaction_sets_by_node.get(local, []),
            }
        )

    return {"nodes": nodes, "edges": edges}


def list_adducts(request: dict[str, Any]) -> dict[str, Any]:
    """Main adduct types (the measurement-condition adducts) that this
    checkpoint was trained to condition on."""
    model_path = str(request.get("modelPath", "")).strip()
    if not model_path:
        raise ValueError("Model checkpoint path is required.")
    if not Path(model_path).exists():
        raise ValueError(f"Model checkpoint was not found: {model_path}")

    device = torch.device(str(request.get("device") or "cpu"))
    generator = load_generator(
        model_path=model_path, params_path=None, device=device, strict=False
    )
    return {"adducts": list(generator.adduct_type_strs)}


def predict(request: dict[str, Any]) -> dict[str, Any]:
    smiles = str(request.get("smiles", "")).strip()
    if not smiles:
        raise ValueError("SMILES is required.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES.")

    adduct_type = str(request.get("adductType", "")).strip()
    if not adduct_type:
        raise ValueError("Adduct type is required.")
    adduct = Adduct.parse(adduct_type)

    ce = str(request.get("ce", "")).strip()
    if not ce:
        raise ValueError("Collision energy is required.")

    model_path = str(request.get("modelPath", "")).strip()
    if not model_path:
        raise ValueError("Model checkpoint path is required.")
    if not Path(model_path).exists():
        raise ValueError(f"Model checkpoint was not found: {model_path}")

    device = torch.device(str(request.get("device") or "cpu"))

    generator = load_generator(
        model_path=model_path, params_path=None, device=device, strict=False
    )
    source = Compound.from_smiles(smiles)
    neutral_formula = Formula.from_mol(mol).plain
    precursor_mz = adduct.apply_to_mz(neutral_formula.exact_mass)
    ce_ev = parse_ce_to_ev(ce, precursor_mz)
    if ce_ev is None:
        raise ValueError(f"Could not interpret collision energy: {ce!r}")

    reactions = tuple(
        generator.fragmenter.fragment_ion_tree_builder.create_reaction_instances(source)
    )
    with torch.no_grad():
        result = generator.predict(
            [source],
            [adduct],
            [ce_ev],
            precomputed={
                source.smiles: {
                    "reactions": reactions,
                    "relations": ReactionRelations.from_reactions(reactions),
                }
            },
        )

    if not result.spectra.mz.numel():
        raise ValueError("The model did not produce any peaks for this input.")
    intensity_threshold = (
        generator.generation.peak_intensity_threshold
        if request.get("filterLowIntensityPeaks", False)
        else None
    )
    node_indices = result.spectrum_batch.state_node_index.detach().cpu().tolist()
    node_smiles = result.spectrum_batch.node_smiles
    core = result.spectrum_batch.canonical_data
    root = int(
        core.source_compounds.compound_index[
            core.reaction_contexts.root_source_compound_index[
                core.sample_branch_group_index[0]
            ]
        ]
    )
    ion_adducts = result.spectrum_batch.state_adducts
    peaks = sorted(
        (
            {
                "mz": float(mz),
                "intensity": float(intensity),
                "formula": str(generator.tensorizer.tensor_to_formula(formula_row)),
                "confidence": float(confidence),
                "nodeId": int(node_indices[index]),
                "smiles": node_smiles[node_indices[index]],
                "adduct": ion_adducts[index] if index < len(ion_adducts) else "",
                "precursor": int(node_indices[index]) == root,
            }
            for index, (mz, intensity, formula_row, confidence) in enumerate(
                zip(
                    result.spectra.mz.detach().cpu().tolist(),
                    result.spectra.intensity.detach().cpu().tolist(),
                    result.spectra.formula_tensor,
                    result.spectra.confidence.detach().cpu().tolist(),
                )
            )
            # Keep all valid peaks unless output filtering is explicitly enabled.
            if (intensity_threshold is None or intensity > intensity_threshold)
            and bool((formula_row[:-1] >= 0).all())
            and mz > 0
        ),
        key=lambda peak: peak["mz"],
    )
    if not peaks:
        raise ValueError(
            "Every candidate peak was below the model's inference thresholds."
        )
    tree = _build_tree(
        result,
        sample=0,
        tensorizer=generator.tensorizer,
        threshold=intensity_threshold,
        reactions=reactions,
    )

    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(420, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")

    source_drawer = rdMolDraw2D.MolDraw2DSVG(620, 360)
    source_drawer.DrawMolecule(source.mapped_mol)
    source_drawer.FinishDrawing()
    source_svg = source_drawer.GetDrawingText().replace("svg:", "")

    return {
        "canonicalSmiles": Chem.MolToSmiles(mol),
        "svg": svg,
        "sourceSvg": source_svg,
        "formula": str(neutral_formula),
        "exactMass": neutral_formula.exact_mass,
        "adduct": str(adduct),
        "precursorMz": precursor_mz,
        "peaks": peaks,
        "tree": tree,
        "reactions": _reaction_registry(generator, reactions),
        "reactionPatternSet": (
            generator.fragmenter.fragment_ion_tree_builder.reaction_pattern_set.to_dict()
        ),
    }


COMMANDS = {"predict": predict, "listAdducts": list_adducts}


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        command = str(request.get("command", "predict"))
        if command not in COMMANDS:
            raise ValueError(f"Unknown command: {command}")
        # load_generator prints progress lines; redirect them to stderr so
        # stdout stays a single JSON line.
        with contextlib.redirect_stdout(sys.stderr):
            result = COMMANDS[command](request.get("payload", {}))
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
        )
