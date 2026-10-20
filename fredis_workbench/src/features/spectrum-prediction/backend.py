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
        raise ValueError("Predicted tree transition references an unknown reaction.") from exc


def _reaction_registry(generator, reactions) -> list[dict[str, Any]]:
    """Describe the same primitive Source reactions used during prediction."""
    patterns = {
        pattern.pattern_id: pattern
        for pattern in generator.fragmenter.fragment_ion_tree_builder.reaction_pattern_set
    }
    registry = []
    for number, reaction in enumerate(reactions):
        pattern = patterns.get(reaction.reaction_pattern_id)
        rule = next(
            (
                item
                for item in pattern.reaction_rules
                if item.id == reaction.reaction_rule_id
            ),
            None,
        ) if pattern else None
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


def _build_tree(result, sample: int, tensorizer, threshold: float, reactions=()) -> dict[str, Any]:
    """Fragment-tree nodes/edges explored for this prediction, annotated with
    per-node neutral formula/exact mass plus any predicted-peak ion states."""
    fragments = result.fragments
    node_offsets = [i for i, s in enumerate(fragments.node_sample_index.tolist()) if s == sample]
    local_index = {global_index: local for local, global_index in enumerate(node_offsets)}
    num_nodes = len(node_offsets)

    tree = fragments.trees[sample]
    reaction_sets_by_node: dict[int, list[list[int]]] = {0: [[]]}
    edges = []
    for index in range(tree.num_edges):
        edge = tree.get_edge(index)
        transitions = []
        for transition in edge.transitions:
            parent_reactions = (
                [_reaction_id(reactions, reaction) for reaction in transition.parent_reaction_sequence.reactions]
                if transition.parent_reaction_sequence else []
            )
            target_reactions = [
                _reaction_id(reactions, reaction) for reaction in transition.reaction_sequence.reactions
            ]
            transitions.append(
                {
                    "addedReaction": _reaction_id(reactions, transition.added_reaction),
                    "parentReactions": parent_reactions,
                    "targetReactions": target_reactions,
                    "seed": bool(transition.is_seed),
                }
            )
            histories = reaction_sets_by_node.setdefault(edge.target_index, [])
            if target_reactions not in histories:
                histories.append(target_reactions)
        edges.append(
            {
                "id": edge.index,
                "source": edge.source_index,
                "target": edge.target_index,
                "transitions": transitions,
            }
        )

    indegree = [0] * num_nodes
    for edge in edges:
        indegree[edge["target"]] += 1
    depth = [0 if indegree[i] == 0 else -1 for i in range(num_nodes)]
    for _ in range(num_nodes):
        changed = False
        for edge in edges:
            if depth[edge["source"]] < 0:
                continue
            candidate = depth[edge["source"]] + 1
            if depth[edge["target"]] < 0 or candidate < depth[edge["target"]]:
                depth[edge["target"]] = candidate
                changed = True
        if not changed:
            break

    spectrum_batch = result.spectrum_batch
    spectra = result.spectra
    candidate_node_index = spectrum_batch.physical_candidate_node_index.detach().cpu().tolist()
    ion_probability = spectra.ion_probability.detach().cpu().tolist()
    candidate_adduct = spectrum_batch.physical_candidate_adduct
    candidate_sample_index = spectrum_batch.physical_candidate_sample_index.detach().cpu().tolist()
    candidate_mz = spectrum_batch.physical_candidate_mz.detach().cpu().tolist()
    candidate_formula_tensor = spectrum_batch.physical_candidate_formula_tensor
    intensity = spectra.intensity.detach().cpu().tolist()

    # Candidate ions below the trained confidence threshold are the same
    # unrelated adduct/hydrogen-shift guesses excluded from the peak list
    # below; keep the node inspector consistent with what the spectrum shows.
    annotations_by_node: dict[int, list[dict[str, Any]]] = {}
    # Each physical candidate row is its own formula.
    for entry, node in enumerate(candidate_node_index):
        if node not in local_index or candidate_sample_index[entry] != sample:
            continue
        probability = float(ion_probability[entry])
        if probability < threshold or not bool((candidate_formula_tensor[entry, :-1] >= 0).all()) or candidate_mz[entry] <= 0:
            continue
        annotations_by_node.setdefault(local_index[node], []).append(
            {
                "peakMz": float(candidate_mz[entry]),
                "peakIntensity": float(intensity[entry]),
                "adduct": candidate_adduct[entry] if entry < len(candidate_adduct) else "",
                "ionFormula": str(tensorizer.tensor_to_formula(candidate_formula_tensor[entry])),
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
                "precursor": depth[local] == 0,
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
            [source], [adduct], [ce_ev],
            precomputed={source.smiles: {
                "reactions": reactions,
                "relations": ReactionRelations.from_reactions(reactions),
            }},
        )

    if not result.spectra.mz.numel():
        raise ValueError("The model did not produce any peaks for this input.")
    intensity_threshold = generator.generation.peak_intensity_threshold
    node_indices = result.spectrum_batch.physical_candidate_node_index.detach().cpu().tolist()
    node_smiles = result.spectrum_batch.node_smiles
    ion_adducts = result.spectrum_batch.physical_candidate_adduct
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
                "precursor": int(node_indices[index]) == 0,
            }
            for index, (mz, intensity, formula_row, confidence) in enumerate(zip(
                result.spectra.mz.detach().cpu().tolist(),
                result.spectra.intensity.detach().cpu().tolist(),
                result.spectra.formula_tensor,
                result.spectra.confidence.detach().cpu().tolist(),
            ))
            # Below the trained ion confidence threshold: the same unrelated
            # adduct/hydrogen-shift candidate this model now learns to push
            # toward zero, kept out of the predicted spectrum shown here.
            if intensity >= intensity_threshold and bool((formula_row[:-1] >= 0).all()) and mz > 0
        ),
        key=lambda peak: peak["mz"],
    )
    if not peaks:
        raise ValueError("Every candidate peak was below the model's inference thresholds.")
    tree = _build_tree(
        result, sample=0, tensorizer=generator.tensorizer,
        threshold=threshold, reactions=reactions,
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
            generator.fragmenter.fragment_ion_tree_builder
            .reaction_pattern_set.to_dict()
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
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
