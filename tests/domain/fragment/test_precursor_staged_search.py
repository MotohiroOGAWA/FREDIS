"""Precursor-aware, depth-by-depth fragment tree search.

The search finishes each reaction count before the next.  Each precursor
adduct is fixed at the first reaction count that reaches its formula
(Source for [M+H]+, one water loss for [M+H-H2O]+); deeper histories are kept
only when they contain such a precursor history.  A tree shared by several
precursor adducts keeps a history when any of them keeps it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fredis.libs.mmkit.mmkit import Adduct, Compound
from fredis.ml.data_preparation.fragment_tree.context import create_preparation_context
from tests.ml.reaction.TestReactionPipeline import chemistry

PATTERN_SET = Path("fredis/presets/fragment/reaction_pattern_sets/nonring_single_bond/nonring_single_bond.reactionset.pft")
C_BOND, O_BOND = "nonring_c_single_bond", "nonring_o_single_bond"


def _fragmenter(*names: str, max_reaction_count: int = 3):
    patterns = {row["name"]: row for row in json.loads(PATTERN_SET.read_text())["reaction_pattern_set"]["patterns"]}
    config = chemistry()
    builder = config["fragmenter_params"]["fragment_ion_tree_builder"]
    builder["max_reaction_count"] = max_reaction_count
    builder["reaction_pattern_set"] = {"name": "test", "patterns": [patterns[name] for name in names]}
    return create_preparation_context(config).fragmenter


def _sets(tree) -> set[frozenset]:
    return {frozenset(sequence.reactions) for sequence in tree.reaction_sequence_compounds}


def _precursors(fragmenter, tree, adduct) -> set:
    """Precursor histories by fragment SMILES; node numbering depends on build order."""
    return {(tree.get_node(p.node_index).smiles, p.reaction_sequence)
            for p in fragmenter.resolve_precursor_reactions(tree, adduct)}


def _expected(fragmenter, source, full, adducts) -> set[frozenset]:
    """Brute force over the unpruned tree: a history is kept by some adduct."""
    everything = _sets(full)
    kept = set()
    for adduct in adducts:
        precursors = fragmenter.resolve_precursor_reactions(full, adduct)
        if not precursors:
            return everything  # unresolved: nothing can be pruned for the shared tree
        depth = min(len(p.reaction_sequence.reactions) if p.reaction_sequence else 0 for p in precursors)
        if depth == 0:
            return everything
        histories = [frozenset(p.reaction_sequence.reactions) for p in precursors]
        kept |= {values for values in everything if len(values) <= depth or any(h <= values for h in histories)}
    return kept


@pytest.mark.parametrize("smiles", ["CCCO", "OCC(O)CO", "OCCO"])
@pytest.mark.parametrize("adducts", [["[M+H]+"], ["[M+H-H2O]+"], ["[M+H-2H2O]+"], ["[M+H-H2O]+", "[M+H-2H2O]+"],
                                     ["[M+H]+", "[M+H-H2O]+"]])
def test_staged_search_keeps_exactly_the_precursor_histories(smiles, adducts):
    fragmenter = _fragmenter(C_BOND, O_BOND)
    source = Compound.from_smiles(smiles)
    types = [Adduct.parse(value) for value in adducts]
    full = fragmenter.build_fragment_ion_tree(source)
    tree = fragmenter.build_fragment_ion_tree(source, precursor_filter=fragmenter.precursor_filter(source, types))
    assert _sets(tree) == _expected(fragmenter, source, full, types)
    # Peaks are explained exactly as in the unpruned tree.
    for adduct in types:
        assert _precursors(fragmenter, tree, adduct) == _precursors(fragmenter, full, adduct)


def test_water_loss_precursor_prunes_histories_without_it():
    fragmenter = _fragmenter(C_BOND, O_BOND)
    source = Compound.from_smiles("CCCO")
    adduct = Adduct.parse("[M+H-H2O]+")
    full = fragmenter.build_fragment_ion_tree(source)
    tree = fragmenter.build_fragment_ion_tree(source, precursor_filter=fragmenter.precursor_filter(source, [adduct]))
    precursor = {frozenset(p.reaction_sequence.reactions) for p in fragmenter.resolve_precursor_reactions(full, adduct)}
    assert {len(values) for values in precursor} == {1}
    deeper = {values for values in _sets(tree) if len(values) > 1}
    assert deeper and all(any(h <= values for h in precursor) for values in deeper)
    assert len(_sets(tree)) < len(_sets(full))
    assert tree.search_stats["num_precursor_pruned"] > 0


def test_only_the_fewest_reaction_precursors_count():
    fragmenter = _fragmenter(C_BOND, O_BOND)
    source = Compound.from_smiles("OCC(O)CO")
    full = fragmenter.build_fragment_ion_tree(source)
    for adduct in ("[M+H]+", "[M+H-H2O]+", "[M+H-2H2O]+"):
        depths = {len(p.reaction_sequence.reactions) if p.reaction_sequence else 0
                  for p in fragmenter.resolve_precursor_reactions(full, Adduct.parse(adduct))}
        assert len(depths) == 1


@pytest.mark.parametrize("smiles", ["CCCO", "OCC(O)CO", "OCCO", "CCOCCO"])
@pytest.mark.parametrize("adducts", [["[M+H-H2O]+"], ["[M+H-2H2O]+"], ["[M+H]+"], ["[M+H-H2O]+", "[M+H-2H2O]+"]])
@pytest.mark.parametrize("old, new", [(C_BOND, O_BOND), (O_BOND, C_BOND)])
def test_seeded_child_equals_a_full_precursor_aware_build(smiles, adducts, old, new):
    """With old = C–C bonds the child's C–O bonds create the water-loss precursor,
    so parent-only histories must follow the new precursor history; with old =
    C–O bonds the parent tree itself is already pruned."""
    child = _fragmenter(C_BOND, O_BOND)
    source = Compound.from_smiles(smiles)
    types = [Adduct.parse(value) for value in adducts]
    ids = {pattern.name: pattern.pattern_id for pattern in child.reaction_pattern_set}
    parent_reactions = child.fragment_ion_tree_builder.create_reaction_instances(
        source, pattern_ids=frozenset({ids[old]}))
    parent = child.build_fragment_ion_tree(source, precomputed_reactions=parent_reactions,
                                           precursor_filter=child.precursor_filter(source, types))
    added = frozenset({ids[new]})
    seeded = child.build_fragment_ion_tree(
        source,
        seed_reaction_sequences=tuple(parent.reaction_sequence_compounds),
        seed_compounds=dict(parent.reaction_sequence_compounds),
        addable_pattern_ids=added,
        precomputed_reactions=child.fragment_ion_tree_builder.create_reaction_instances(source, pattern_ids=added),
        precursor_filter=child.precursor_filter(source, types),
    )
    full = child.build_fragment_ion_tree(source, precursor_filter=child.precursor_filter(source, types))
    assert _sets(seeded) == _sets(full)
    for adduct in types:
        assert _precursors(child, seeded, adduct) == _precursors(child, full, adduct)
