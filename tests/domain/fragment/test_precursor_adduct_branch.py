"""Precursor losses select structural ancestors, never blanket fragment offsets."""

from dataclasses import replace
from fredis.domain.fragment.fragmenter import Fragmenter
from fredis.domain.fragment.ion_tree import FragmentIonAdductRuleSet
from fredis.libs.mmkit.mmkit import Compound, Adduct


def loss_fragmenter():
    fragmenter = Fragmenter.from_json(
        "tests/domain/fragment/test_data/fragmenter_single_bond_pos.json"
    )
    rules = FragmentIonAdductRuleSet.from_dict(
        {
            "name": "precursor-loss",
            "hydrogen_shift_min": -5,
            "hydrogen_shift_max": 0,
            "adduct_rules": [
                {
                    "name": "H",
                    "adduct_type": "[M+H]+",
                    "fragment_ion_adducts": ["[M+H]+"],
                },
                {
                    "name": "H_water_loss",
                    "adduct_type": "[M+H-H2O]+",
                    "fragment_ion_adducts": ["[M+H]+"],
                },
            ],
        }
    )
    return replace(
        fragmenter,
        fragment_ion_tree_builder=replace(
            fragmenter.fragment_ion_tree_builder, fragment_ion_adduct_rule_set=rules
        ),
    )


def test_loss_selects_transformed_structure_and_simple_fragment_ions():
    fragmenter = loss_fragmenter()
    source = Compound.from_smiles("CCCO")
    tree = fragmenter.build_fragment_ion_tree(source)
    adduct = Adduct.parse("[M+H-H2O]+")
    precursors = fragmenter.resolve_precursor_reactions(tree, adduct)
    assert precursors
    assert all(
        p.node_index != 0 and p.reaction_sequence is not None for p in precursors
    )
    assert all(
        tree.fragment_compounds[p.node_index].formula.elements.get("O", 0) == 0
        for p in precursors
    )
    assert fragmenter._resolve_main_adduct_type(adduct) == adduct
    group = tree.get_formula_candidate_group(adduct)
    annotations = fragmenter.annotate_peaks(
        tree, adduct, [f.exact_mass for f in group.formulas]
    )
    assert any(annotations)
    for matches in annotations:
        for annotation in matches:
            assert str(annotation.fragment_ion_adduct) == "[M+H]+"
            assert annotation.precursor in precursors
            assert fragmenter._has_precursor_ancestor(
                tree,
                annotation.reaction_sequence,
                annotation.precursor.reaction_sequence,
            )
            assert annotation.node_index != 0
            assert annotation.formula.elements.get("O", 0) == 0


def test_unreachable_loss_is_not_faked_on_source():
    fragmenter = loss_fragmenter()
    tree = fragmenter.build_fragment_ion_tree(Compound.from_smiles("CCC"))
    adduct = Adduct.parse("[M+H-H2O]+")
    assert not fragmenter.resolve_precursor_reactions(tree, adduct)
    assert fragmenter.annotate_peaks(tree, adduct, [43.054]) == ((),)
