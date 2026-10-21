"""Shared readers for saved fragment-tree preparation results.

Chemistry is performed during preparation.  These helpers only restore the
saved reaction registry and the reaction combinations reaching each saved
fragment node.  Workbench viewers and dataset analyses use the same mapping.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator
import json

import torch


def load_prepared_structure(file_path: str | Path):
    payload = torch.load(Path(file_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "structure" not in payload:
        raise TypeError("This .preft.pt file has no saved structure.")
    structure = payload["structure"]
    required = (
        "teacher_node_reaction_ptr",
        "sample_branch_group_index",
        "reactions",
    )
    if not all(hasattr(structure, name) for name in required):
        raise TypeError("This is not a prepared fragment-tree training structure.")
    return structure, dict(payload.get("metadata") or {})


def find_fragmenter_params(
    file_path: str | Path, metadata: dict | None = None
) -> dict | None:
    metadata = metadata or {}
    if metadata.get("fragmenter_params"):
        return metadata["fragmenter_params"]
    for parent in Path(file_path).resolve().parents:
        config = parent / "preparation_config.json"
        if not config.exists():
            continue
        value = json.loads(config.read_text())
        return value.get("chemistry_config", {}).get("fragmenter_params")
    return None


def restore_reaction_registry(
    structure, source_smiles: str, fragmenter_params: dict
) -> list[dict]:
    """Return reaction metadata in exactly the saved tensor reaction order."""
    from fredis.domain.fragment.fragmenter import Fragmenter
    from fredis.libs.mmkit.mmkit import Compound

    fragmenter = Fragmenter.from_dict(fragmenter_params)
    source = Compound.from_smiles(source_smiles)
    originals = list(
        fragmenter.fragment_ion_tree_builder.create_reaction_instances(source)
    )
    if len(originals) != len(structure.reactions.reaction_type):
        originals = []
    patterns = {
        pattern.pattern_id: pattern
        for pattern in fragmenter.fragment_ion_tree_builder.reaction_pattern_set
    }
    atoms = list(source.mapped_mol.GetAtoms())
    if len(atoms) != structure.source_graph.num_nodes:
        atoms = [atom for atom in atoms if atom.GetAtomicNum() != 1]
    pointers = structure.reactions.reaction_source_atom_ptr.tolist()
    atom_indexes = structure.reactions.reaction_source_atom_index.tolist()
    rows = []
    for index, category in enumerate(structure.reactions.reaction_type.tolist()):
        reaction = originals[index] if originals else None
        pattern = patterns.get(category[0])
        rule = (
            next(
                (item for item in pattern.reaction_rules if item.id == category[1]),
                None,
            )
            if pattern
            else None
        )
        rows.append(
            {
                "id": index,
                "reactionPatternId": category[0],
                "reactionPatternName": pattern.name if pattern else "",
                "reactantSmarts": pattern.reactant_smarts if pattern else "",
                "reactionRuleId": category[1],
                "reactionRuleName": rule.source_rule.name if rule else "",
                "reactionRuleProductSmarts": rule.source_rule.smarts if rule else "",
                "productMoleculeId": category[2],
                "sourceAtomMaps": (
                    list(reaction.source_atom_maps)
                    if reaction
                    else [
                        atoms[value].GetAtomMapNum()
                        for value in atom_indexes[pointers[index] : pointers[index + 1]]
                    ]
                ),
                "retainedAtomMaps": (
                    sorted(reaction.retained_atom_maps) if reaction else []
                ),
                "discardedAtomMaps": (
                    sorted(reaction.discarded_atom_maps) if reaction else []
                ),
                "cutBondMaps": sorted(reaction.cut_bond_maps) if reaction else [],
                "changedBondMaps": (
                    sorted(reaction.changed_bond_maps) if reaction else []
                ),
            }
        )
    return rows


def state_reaction_combinations(
    structure,
) -> dict[tuple[int, int], list[tuple[int, ...]]]:
    """Map ``(branch group, fragment node)`` to every saved reaction combination."""
    pointers = structure.teacher_node_reaction_ptr.tolist()
    indexes = structure.teacher_node_reaction_index.tolist()
    groups = structure.teacher_node_branch_group_index.tolist()
    nodes = structure.state_fragment_node_index.tolist()
    result: dict[tuple[int, int], list[tuple[int, ...]]] = defaultdict(list)
    for row, (group, node) in enumerate(zip(groups, nodes)):
        if node < 0:
            continue
        combination = tuple(sorted(indexes[pointers[row] : pointers[row + 1]]))
        if combination not in result[(group, node)]:
            result[(group, node)].append(combination)
    return dict(result)


def iter_prepared_samples(
    file_path: str | Path, fragmenter_params: dict | None = None
) -> Iterator[dict]:
    """Yield compact samples with observed peaks and every explaining reaction combination."""
    structure, metadata = load_prepared_structure(file_path)
    source_smiles = str(metadata.get("smiles", ""))
    params = fragmenter_params or find_fragmenter_params(file_path, metadata)
    if not params:
        raise ValueError("Fragmenter parameters are required to restore reactions.")
    reactions = list(metadata.get("reaction_registry") or ())
    if not reactions:
        reactions = restore_reaction_registry(structure, source_smiles, params)
    combinations = state_reaction_combinations(structure)
    annotations = tuple(
        getattr(structure, "sample_annotations", ())
        or metadata.get("sample_annotations", ())
    )
    record_indexes = list(metadata.get("record_indexes", range(len(annotations))))
    groups = structure.sample_branch_group_index.tolist()
    for sample_index, annotation in enumerate(annotations):
        group = groups[sample_index]
        peaks = []
        for peak in annotation.get("peaks", ()):
            matches = []
            for match in peak.get("matches", ()):
                for node in match.get("nodeIndices", ()):
                    for combination in combinations.get((group, node), ()):
                        reaction_rows = [
                            reactions[reaction] for reaction in combination
                        ]
                        matches.append(
                            {
                                **match,
                                "nodeIndex": node,
                                "reactionCount": len(combination),
                                "reactions": reaction_rows,
                                "reactionPatterns": sorted(
                                    {
                                        row["reactionPatternName"]
                                        for row in reaction_rows
                                    }
                                ),
                            }
                        )
            peaks.append({**peak, "matches": matches})
        yield {
            **annotation,
            "sampleIndex": sample_index,
            "recordIndex": record_indexes[sample_index],
            "branchGroup": group,
            "sourceSmiles": source_smiles,
            "peaks": peaks,
        }
