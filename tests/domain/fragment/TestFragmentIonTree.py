from __future__ import annotations

import unittest

import numpy as np

from fredis.libs.mmkit.mmkit import Adduct, Compound, Formula

from fredis.domain.fragment.tree.FragmentNode import FragmentNode
from fredis.domain.fragment.tree.FragmentTree import FragmentTree
from fredis.domain.fragment.ion_tree.FragmentIonTree import FragmentIonTree
from fredis.domain.fragment.ion_tree.FragmentIonAdductRule import (
    FragmentIonAdductRule,
)
from fredis.domain.fragment.ion_tree.FragmentIonAdductRuleSet import (
    FragmentIonAdductRuleSet,
)
from fredis.domain.fragment.ion_tree._private._FragmentIonFormulaCandidateGroup import (
    _FragmentIonFormulaCandidateGroup,
)


POSITIVE_RULE_SET = {
    "name": "positive",
    "hydrogen_shift_min": -5,
    "hydrogen_shift_max": 0,
    "adduct_rules": [
        {
            "name": "pos_H",
            "adduct_type": "[M+H]+",
            "fragment_ion_adducts": ["[M-H]+", "[M+H]+"],
        },
        {
            "name": "pos_Na",
            "adduct_type": "[M+Na]+",
            "fragment_ion_adducts": ["[M+Na]+", "[M-H]+", "[M+H]+"],
        },
        {
            "name": "pos_NH4",
            "adduct_type": "[M+NH4]+",
            "fragment_ion_adducts": ["[M+NH4]+", "[M+H]+", "[M-H]+"],
        },
        {
            "name": "pos_2M_H_to_M",
            "adduct_type": "[2M+H]+",
            "fragment_ion_adducts": ["[M+H]+", "[M-H]+"],
        },
    ],
}

NODE_SMILES = ("CCO", "CC", "[NH4+]", "Cl")
NEUTRAL_NODE_INDICES = (0, 1, 3)
CHARGED_NODE_INDEX = 2


def _strs(values) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _make_fragment_tree(node_smiles: tuple[str, ...]) -> FragmentTree:
    return FragmentTree.from_nodes_and_edges(
        smiles=node_smiles[0],
        nodes=tuple(
            FragmentNode(index=index, smiles=smiles)
            for index, smiles in enumerate(node_smiles)
        ),
        edges=(),
    )


def _make_compounds(node_smiles: tuple[str, ...]) -> tuple[Compound, ...]:
    return tuple(Compound.from_smiles(smiles) for smiles in node_smiles)


class TestFragmentIonAdductRuleSet(unittest.TestCase):
    def setUp(self) -> None:
        self.rule_set = FragmentIonAdductRuleSet.from_dict(POSITIVE_RULE_SET)

    def test_round_trips_through_dict(self) -> None:
        encoded = self.rule_set.to_dict()

        self.assertEqual(
            FragmentIonAdductRuleSet.from_dict(encoded).to_dict(),
            encoded,
        )
        self.assertEqual(
            set(encoded["adduct_rules"][0]),
            {"name", "adduct_type", "fragment_ion_adducts"},
        )
        self.assertEqual(
            FragmentIonAdductRuleSet.from_dict(
                {"fragment_ion_adduct_rule_set": POSITIVE_RULE_SET}
            ),
            self.rule_set,
        )
        self.assertEqual(self.rule_set.copy(), self.rule_set)

    def test_hydrogen_shifts_cover_inclusive_range(self) -> None:
        self.assertEqual(
            self.rule_set.hydrogen_shifts,
            (-5, -4, -3, -2, -1, 0),
        )

    def test_requires_hydrogen_shift_range(self) -> None:
        for key in ("hydrogen_shift_min", "hydrogen_shift_max"):
            with self.subTest(key=key):
                data = dict(POSITIVE_RULE_SET)
                del data[key]

                with self.assertRaises(KeyError):
                    FragmentIonAdductRuleSet.from_dict(data)

    def test_rejects_inverted_hydrogen_shift_range(self) -> None:
        data = dict(POSITIVE_RULE_SET, hydrogen_shift_min=1, hydrogen_shift_max=0)

        with self.assertRaises(ValueError):
            FragmentIonAdductRuleSet.from_dict(data)

    def test_rejects_duplicated_adduct_types(self) -> None:
        rule = POSITIVE_RULE_SET["adduct_rules"][0]
        data = dict(POSITIVE_RULE_SET, adduct_rules=[rule, dict(rule, name="dup")])

        with self.assertRaises(ValueError):
            FragmentIonAdductRuleSet.from_dict(data)

    def test_adduct_rule_requires_fragment_ion_adducts(self) -> None:
        data = {
            "hydrogen_shift_min": 0,
            "hydrogen_shift_max": 0,
            "adduct_rules": [{"name": "pos_H", "adduct_type": "[M+H]+"}],
        }

        with self.assertRaises(KeyError):
            FragmentIonAdductRuleSet.from_dict(data)

    def test_adduct_rule_rejects_empty_fragment_ion_adducts(self) -> None:
        data = {
            "hydrogen_shift_min": 0,
            "hydrogen_shift_max": 0,
            "adduct_rules": [
                {
                    "name": "pos_H",
                    "adduct_type": "[M+H]+",
                    "fragment_ion_adducts": [],
                }
            ],
        }

        with self.assertRaises(ValueError):
            FragmentIonAdductRuleSet.from_dict(data)

    def test_adduct_rule_rejects_duplicated_fragment_ion_adducts(self) -> None:
        with self.assertRaises(ValueError):
            FragmentIonAdductRule(
                name="pos_H",
                adduct_type=Adduct.parse("[M+H]+"),
                fragment_ion_adducts=(
                    Adduct.parse("[M+H]+"),
                    Adduct.parse("[M+H]+"),
                ),
            )


class TestFragmentIonTree(unittest.TestCase):
    def setUp(self) -> None:
        self.fragment_tree = _make_fragment_tree(NODE_SMILES)
        self.rule_set = FragmentIonAdductRuleSet.from_dict(POSITIVE_RULE_SET)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_from_fragment_tree_keeps_original_topology(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        self.assertEqual(fragment_ion_tree.smiles, self.fragment_tree.smiles)
        self.assertEqual(fragment_ion_tree.num_nodes, self.fragment_tree.num_nodes)
        self.assertEqual(fragment_ion_tree.num_edges, self.fragment_tree.num_edges)

        np.testing.assert_array_equal(
            fragment_ion_tree.node_smiles,
            self.fragment_tree.node_smiles,
        )

        for node_index, smiles in enumerate(NODE_SMILES):
            self.assertEqual(fragment_ion_tree.get_node(node_index).smiles, smiles)

    def test_from_fragment_tree_stores_fields(self) -> None:
        compounds = _make_compounds(NODE_SMILES)
        search_stats = {"num_generated_fragments": 3}
        sequence_compounds = {"seq": compounds[1]}

        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
            fragment_compounds=list(compounds),
            search_stats=search_stats,
            reaction_sequence_compounds=sequence_compounds,
        )

        self.assertIs(fragment_ion_tree.fragment_ion_adduct_rule_set, self.rule_set)
        self.assertIsInstance(fragment_ion_tree.fragment_compounds, tuple)
        self.assertEqual(len(fragment_ion_tree.fragment_compounds), len(NODE_SMILES))
        for stored, original in zip(fragment_ion_tree.fragment_compounds, compounds):
            self.assertIs(stored, original)

        self.assertEqual(fragment_ion_tree.search_stats, search_stats)
        self.assertEqual(fragment_ion_tree.reaction_sequence_compounds, sequence_compounds)
        self.assertIsNot(
            fragment_ion_tree.reaction_sequence_compounds,
            sequence_compounds,
        )
        self.assertEqual(fragment_ion_tree.formula_candidate_groups, {})

    def test_from_fragment_tree_defaults(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        self.assertIsNone(fragment_ion_tree.search_stats)
        self.assertEqual(fragment_ion_tree.reaction_sequence_compounds, {})

    def test_fragment_compounds_length_must_match_nodes(self) -> None:
        compounds = _make_compounds(NODE_SMILES)

        for bad in (compounds[:-1], compounds + (compounds[0],), ()):
            with self.subTest(length=len(bad)):
                with self.assertRaises(ValueError):
                    FragmentIonTree.from_fragment_tree(
                        self.fragment_tree,
                        fragment_ion_adduct_rule_set=self.rule_set,
                        fragment_compounds=bad,
                    )

    def test_fragment_compounds_must_be_compounds(self) -> None:
        bad = _make_compounds(NODE_SMILES)[:-1] + ("Cl",)

        with self.assertRaises(TypeError):
            FragmentIonTree.from_fragment_tree(
                self.fragment_tree,
                fragment_ion_adduct_rule_set=self.rule_set,
                fragment_compounds=bad,
            )

    def test_invalid_fragment_tree_argument_raises(self) -> None:
        with self.assertRaises(TypeError):
            FragmentIonTree.from_fragment_tree(
                object(),
                fragment_ion_adduct_rule_set=self.rule_set,
                fragment_compounds=_make_compounds(NODE_SMILES),
            )

    def test_invalid_rule_set_argument_raises(self) -> None:
        with self.assertRaises(TypeError):
            FragmentIonTree.from_fragment_tree(
                self.fragment_tree,
                fragment_ion_adduct_rule_set=object(),
                fragment_compounds=_make_compounds(NODE_SMILES),
            )

    # ------------------------------------------------------------------
    # Rule access
    # ------------------------------------------------------------------

    def test_hydrogen_shift_range_is_shared_by_adduct_types(self) -> None:
        rule_set = FragmentIonAdductRuleSet.from_dict(
            {
                "hydrogen_shift_min": -2,
                "hydrogen_shift_max": 1,
                "adduct_rules": [
                    {
                        "name": "pos_H",
                        "adduct_type": "[M+H]+",
                        "fragment_ion_adducts": ["[M+H]+"],
                    },
                    {
                        "name": "pos_Na",
                        "adduct_type": "[M+Na]+",
                        "fragment_ion_adducts": ["[M+Na]+"],
                    },
                ],
            },
        )

        fragment_ion_tree = self._make_fragment_ion_tree(rule_set=rule_set)

        self.assertEqual(fragment_ion_tree.hydrogen_shifts, (-2, -1, 0, 1))
        self.assertEqual(
            fragment_ion_tree.hydrogen_shift_adducts,
            tuple(Adduct.from_dict({"H": shift}) for shift in (-2, -1, 0, 1)),
        )
        for adduct in fragment_ion_tree.hydrogen_shift_adducts:
            self.assertEqual(adduct.charge, 0)

        for adduct_type in ("[M+H]+", "[M+Na]+"):
            with self.subTest(adduct_type=adduct_type):
                group = fragment_ion_tree.get_formula_candidate_group(adduct_type)
                self.assertEqual(
                    set(group.hydrogen_shifts.tolist()),
                    {-2, -1, 0, 1},
                )

    def test_node_is_neutral(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        np.testing.assert_array_equal(
            fragment_ion_tree.node_is_neutral,
            np.asarray([True, True, False, True]),
        )
        self.assertEqual(fragment_ion_tree.node_is_neutral.dtype, np.bool_)

    def test_negative_charged_fragment_is_not_neutral(self) -> None:
        node_smiles = ("[O-]", "O")
        fragment_ion_tree = self._make_fragment_ion_tree(node_smiles=node_smiles)

        np.testing.assert_array_equal(
            fragment_ion_tree.node_is_neutral,
            np.asarray([False, True]),
        )
        self.assertEqual(fragment_ion_tree.get_fragment_ion_adducts(0, "[M+H]+"), ())
        self.assertEqual(
            _strs(fragment_ion_tree.get_fragment_ion_adducts(1, "[M+H]+")),
            ("[M-H]+", "[M+H]+"),
        )

    def test_get_adduct_rule(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        for index, rule in enumerate(self.rule_set.adduct_rules):
            with self.subTest(adduct_type=str(rule.adduct_type)):
                self.assertEqual(
                    fragment_ion_tree.get_adduct_rule_index_by_adduct_type(
                        str(rule.adduct_type)
                    ),
                    index,
                )
                self.assertEqual(
                    fragment_ion_tree.get_adduct_rule_index_by_adduct_type(
                        rule.adduct_type
                    ),
                    index,
                )
                self.assertIs(
                    fragment_ion_tree.get_adduct_rule(str(rule.adduct_type)),
                    rule,
                )

        self.assertEqual(fragment_ion_tree.get_adduct_rule("[M+NH4]+").name, "pos_NH4")

    def test_unknown_adduct_type_raises(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        with self.assertRaises(KeyError):
            fragment_ion_tree.get_adduct_rule("[M+K]+")
        with self.assertRaises(KeyError):
            fragment_ion_tree.get_adduct_rule_index_by_adduct_type("[M+K]+")
        with self.assertRaises(KeyError):
            fragment_ion_tree.get_fragment_ion_adducts(0, "[M+K]+")
        with self.assertRaises(KeyError):
            fragment_ion_tree.get_formula_candidate_group("[M+K]+")

    def test_get_fragment_ion_adducts_for_neutral_nodes(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        expected = {
            "[M+H]+": ("[M-H]+", "[M+H]+"),
            "[M+Na]+": ("[M+Na]+", "[M-H]+", "[M+H]+"),
            "[M+NH4]+": ("[M+H4N]+", "[M+H]+", "[M-H]+"),
            "[2M+H]+": ("[M+H]+", "[M-H]+"),
        }

        for node_index in NEUTRAL_NODE_INDICES:
            for adduct_type, expected_adducts in expected.items():
                with self.subTest(node_index=node_index, adduct_type=adduct_type):
                    adducts = fragment_ion_tree.get_fragment_ion_adducts(
                        node_index,
                        adduct_type,
                    )
                    self.assertEqual(_strs(adducts), expected_adducts)
                    self.assertIs(
                        adducts,
                        fragment_ion_tree.get_adduct_rule(adduct_type).fragment_ion_adducts,
                    )

    def test_get_fragment_ion_adducts_for_charged_node_is_empty(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        for rule in self.rule_set.adduct_rules:
            with self.subTest(adduct_type=str(rule.adduct_type)):
                self.assertEqual(
                    fragment_ion_tree.get_fragment_ion_adducts(
                        CHARGED_NODE_INDEX,
                        rule.adduct_type,
                    ),
                    (),
                )

    def test_get_fragment_ion_adducts_invalid_node_index_raises(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        for node_index in (-1, fragment_ion_tree.num_nodes):
            with self.subTest(node_index=node_index):
                with self.assertRaises(IndexError):
                    fragment_ion_tree.get_fragment_ion_adducts(node_index, "[M+H]+")

    # ------------------------------------------------------------------
    # Formula candidates
    # ------------------------------------------------------------------

    def test_formula_candidate_group_is_cached_per_adduct_type(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        group = fragment_ion_tree.get_formula_candidate_group("[M+H]+")

        self.assertIsInstance(group, _FragmentIonFormulaCandidateGroup)
        self.assertIs(
            fragment_ion_tree.get_formula_candidate_group(Adduct.parse("[M+H]+")),
            group,
        )
        self.assertEqual(
            list(fragment_ion_tree.formula_candidate_groups),
            [Adduct.parse("[M+H]+")],
        )
        self.assertIsNot(
            fragment_ion_tree.get_formula_candidate_group("[M+Na]+"),
            group,
        )
        self.assertEqual(len(fragment_ion_tree.formula_candidate_groups), 2)

    def test_formula_candidate_group_holds_rule_adducts(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        for rule in self.rule_set.adduct_rules:
            with self.subTest(adduct_type=str(rule.adduct_type)):
                group = fragment_ion_tree.get_formula_candidate_group(rule.adduct_type)
                self.assertEqual(group.adduct_type, rule.adduct_type)
                self.assertEqual(group.fragment_ion_adducts, rule.fragment_ion_adducts)

    def test_formula_candidate_group_structure(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        for rule in self.rule_set.adduct_rules:
            with self.subTest(adduct_type=str(rule.adduct_type)):
                group = fragment_ion_tree.get_formula_candidate_group(rule.adduct_type)

                self.assertEqual(group.indptr[0], 0)
                self.assertEqual(len(group.indptr), group.num_formulas + 1)
                self.assertEqual(int(group.indptr[-1]), group.num_candidates)
                self.assertTrue(np.all(np.diff(group.indptr) > 0))
                for name in (
                    "node_indices",
                    "candidate_adduct_indices",
                    "fragment_ion_adduct_positions",
                    "hydrogen_shifts",
                ):
                    self.assertEqual(len(getattr(group, name)), group.num_candidates)

                masses = [formula.exact_mass for formula in group.formulas]
                self.assertEqual(masses, sorted(masses))
                self.assertEqual(
                    len(set(_strs(group.formulas))),
                    group.num_formulas,
                )
                self.assertEqual(
                    len(set(_strs(group.candidate_adducts))),
                    len(group.candidate_adducts),
                )

    def test_formula_candidates_are_consistent_with_their_parts(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        for rule in self.rule_set.adduct_rules:
            group = fragment_ion_tree.get_formula_candidate_group(rule.adduct_type)

            for formula_index in range(group.num_formulas):
                candidate_range = group.get_candidate_range(formula_index)
                formula = group.formulas[formula_index]

                for i in range(candidate_range.start, candidate_range.stop):
                    node_index = int(group.node_indices[i])
                    position = int(group.fragment_ion_adduct_positions[i])
                    shift = int(group.hydrogen_shifts[i])
                    fragment_ion_adduct = group.fragment_ion_adducts[position]
                    shift_adduct = Adduct.from_dict({"H": shift})

                    with self.subTest(
                        adduct_type=str(rule.adduct_type),
                        formula=str(formula),
                        candidate=i,
                    ):
                        self.assertIn(node_index, NEUTRAL_NODE_INDICES)
                        self.assertIn(shift, fragment_ion_tree.hydrogen_shifts)

                        base_formula = fragment_ion_tree.fragment_compounds[
                            node_index
                        ].formula.normalized
                        self.assertEqual(
                            fragment_ion_adduct.apply_to_formula(
                                shift_adduct.apply_to_formula(base_formula)
                            ).normalized,
                            formula,
                        )

                        candidate_adduct = group.candidate_adducts[
                            int(group.candidate_adduct_indices[i])
                        ]
                        self.assertEqual(
                            str(candidate_adduct),
                            str(fragment_ion_adduct.add_prefer_self(shift_adduct)),
                        )
                        self.assertEqual(
                            candidate_adduct.apply_to_formula(base_formula).normalized,
                            formula,
                        )

    def test_formula_candidates_enumerate_every_combination(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        for rule in self.rule_set.adduct_rules:
            with self.subTest(adduct_type=str(rule.adduct_type)):
                group = fragment_ion_tree.get_formula_candidate_group(rule.adduct_type)

                actual = set(
                    zip(
                        group.node_indices.tolist(),
                        group.fragment_ion_adduct_positions.tolist(),
                        group.hydrogen_shifts.tolist(),
                    )
                )
                self.assertEqual(len(actual), group.num_candidates)

                expected = set()
                for node_index in NEUTRAL_NODE_INDICES:
                    base = fragment_ion_tree.fragment_compounds[node_index].formula
                    for shift in fragment_ion_tree.hydrogen_shifts:
                        for position, adduct in enumerate(rule.fragment_ion_adducts):
                            formula = adduct.apply_to_formula(
                                Adduct.from_dict({"H": shift}).apply_to_formula(base)
                            )
                            if formula.normalized.is_nonnegative:
                                expected.add((node_index, position, shift))

                self.assertEqual(actual, expected)

    def test_get_node_formula_candidates_examples(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        cases = (
            # Protonated ethanol.
            ("[M+H]+", "C2H7O+", ((0, "[M+H]+"),)),
            # [M-H]+ at shift 0 and [M+H]+ at shift -2 combine to the same adduct.
            ("[M+H]+", "C2H5O+", ((0, "[M-H]+"), (0, "[M-H]+"))),
            # HCl with [M-H]+ / [M+H]+ and hydrogen shifts.
            ("[M+H]+", "HCl+", ((3, "[M]+"),)),
            ("[M+H]+", "C2H7+", ((1, "[M+H]+"),)),
            ("[M+Na]+", "C2H6NaO+", ((0, "[M+Na]+"),)),
            ("[M+Na]+", "C2H5NaO+", ((0, "[M+Na-H]+"),)),
            ("[M+NH4]+", "C2H10NO+", ((0, "[M+H4N]+"),)),
            ("[2M+H]+", "C2H7O+", ((0, "[M+H]+"),)),
        )

        for adduct_type, formula, expected in cases:
            with self.subTest(adduct_type=adduct_type, formula=formula):
                candidates = fragment_ion_tree.get_node_formula_candidates(
                    adduct_type=adduct_type,
                    formula=Formula.parse(formula),
                )
                self.assertEqual(
                    sorted((node_index, str(adduct)) for node_index, adduct in candidates),
                    sorted(expected),
                )

    def test_formula_exact_masses(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()
        group = fragment_ion_tree.get_formula_candidate_group("[M+H]+")

        for formula, expected_mass in (
            ("C2H7O+", 47.049689844),
            ("HCl+", 35.976677712),
            ("C2+", 24.0),
        ):
            with self.subTest(formula=formula):
                formula_index = group.get_formula_index(Formula.parse(formula))
                self.assertAlmostEqual(
                    group.formulas[formula_index].exact_mass,
                    expected_mass,
                    places=6,
                )

        self.assertEqual(str(group.formulas[0]), "C2+")
        self.assertEqual(str(group.formulas[-1]), "C2H7O+")

    def test_formula_candidate_group_accessors(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()
        group = fragment_ion_tree.get_formula_candidate_group("[M+H]+")

        formula = Formula.parse("C2H5O+")
        formula_index = group.get_formula_index(formula)

        self.assertEqual(
            group.get_candidates_by_formula_index(formula_index),
            group.get_candidates_by_formula(formula),
        )
        np.testing.assert_array_equal(
            group.get_node_indices_by_formula_index(formula_index),
            np.asarray([0, 0]),
        )
        self.assertEqual(
            _strs(group.get_candidate_adducts_by_formula_index(formula_index)),
            ("[M-H]+", "[M-H]+"),
        )

        candidate_range = group.get_candidate_range(formula_index)
        self.assertEqual(
            sorted(
                zip(
                    group.fragment_ion_adduct_positions[candidate_range].tolist(),
                    group.hydrogen_shifts[candidate_range].tolist(),
                )
            ),
            [(0, 0), (1, -2)],
        )

    def test_formula_candidate_group_errors(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()
        group = fragment_ion_tree.get_formula_candidate_group("[M+H]+")

        with self.assertRaises(KeyError):
            group.get_formula_index(Formula.parse("C9+"))
        with self.assertRaises(KeyError):
            fragment_ion_tree.get_node_formula_candidates(
                adduct_type="[M+H]+",
                formula=Formula.parse("C9+"),
            )
        for formula_index in (-1, group.num_formulas):
            with self.subTest(formula_index=formula_index):
                with self.assertRaises(IndexError):
                    group.get_candidate_range(formula_index)
        with self.assertRaises(TypeError):
            group.get_candidate_range(0.0)

    def test_charged_node_produces_no_formula_candidates(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree()

        for rule in self.rule_set.adduct_rules:
            with self.subTest(adduct_type=str(rule.adduct_type)):
                group = fragment_ion_tree.get_formula_candidate_group(rule.adduct_type)
                self.assertNotIn(CHARGED_NODE_INDEX, group.node_indices.tolist())
                self.assertEqual(
                    set(group.node_indices.tolist()),
                    set(NEUTRAL_NODE_INDICES),
                )

    def test_only_charged_nodes_give_empty_group(self) -> None:
        fragment_ion_tree = self._make_fragment_ion_tree(
            node_smiles=("[NH4+]", "[O-]"),
        )

        group = fragment_ion_tree.get_formula_candidate_group("[M+H]+")

        self.assertEqual(group.num_formulas, 0)
        self.assertEqual(group.num_candidates, 0)
        self.assertEqual(group.candidate_adducts, ())
        np.testing.assert_array_equal(group.indptr, np.zeros(1, dtype=np.int64))

    def test_empty_rule_set(self) -> None:
        empty_rule_set = FragmentIonAdductRuleSet.from_dict(
            {
                "hydrogen_shift_min": -5,
                "hydrogen_shift_max": 0,
                "adduct_rules": [],
            },
        )

        fragment_ion_tree = self._make_fragment_ion_tree(rule_set=empty_rule_set)

        self.assertEqual(fragment_ion_tree.hydrogen_shifts, (-5, -4, -3, -2, -1, 0))
        with self.assertRaises(KeyError):
            fragment_ion_tree.get_fragment_ion_adducts(0, "[M+H]+")
        with self.assertRaises(KeyError):
            fragment_ion_tree.get_formula_candidate_group("[M+H]+")

    def test_negative_rule_set(self) -> None:
        negative_rule_set = FragmentIonAdductRuleSet.from_dict(
            {
                "hydrogen_shift_min": -5,
                "hydrogen_shift_max": 0,
                "adduct_rules": [
                    {
                        "name": "neg_H",
                        "adduct_type": "[M-H]-",
                        "fragment_ion_adducts": ["[M-H]-"],
                    },
                    {
                        "name": "neg_2M_H_to_M",
                        "adduct_type": "[2M-H]-",
                        "fragment_ion_adducts": ["[M-H]-"],
                    },
                ],
            },
        )

        fragment_ion_tree = self._make_fragment_ion_tree(rule_set=negative_rule_set)

        for adduct_type in ("[M-H]-", "[2M-H]-"):
            for node_index in range(fragment_ion_tree.num_nodes):
                with self.subTest(adduct_type=adduct_type, node_index=node_index):
                    self.assertEqual(
                        _strs(
                            fragment_ion_tree.get_fragment_ion_adducts(
                                node_index,
                                adduct_type,
                            )
                        ),
                        () if node_index == CHARGED_NODE_INDEX else ("[M-H]-",),
                    )

            candidates = fragment_ion_tree.get_node_formula_candidates(
                adduct_type=adduct_type,
                formula=Formula.parse("C2H5O-"),
            )
            self.assertEqual(
                [(node_index, str(adduct)) for node_index, adduct in candidates],
                [(0, "[M-H]-")],
            )
            group = fragment_ion_tree.get_formula_candidate_group(adduct_type)
            self.assertTrue(all(formula.charge == -1 for formula in group.formulas))

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def test_copy(self) -> None:
        compounds = _make_compounds(NODE_SMILES)
        fragment_ion_tree = FragmentIonTree.from_fragment_tree(
            self.fragment_tree,
            fragment_ion_adduct_rule_set=self.rule_set,
            fragment_compounds=compounds,
            search_stats={"num_generated_fragments": 3},
            reaction_sequence_compounds={"seq": compounds[0]},
        )
        group = fragment_ion_tree.get_formula_candidate_group("[M+H]+")

        copied = fragment_ion_tree.copy()

        self.assertIsInstance(copied, FragmentIonTree)
        self.assertEqual(copied.num_nodes, fragment_ion_tree.num_nodes)
        np.testing.assert_array_equal(copied.node_smiles, fragment_ion_tree.node_smiles)
        self.assertIs(
            copied.fragment_ion_adduct_rule_set,
            fragment_ion_tree.fragment_ion_adduct_rule_set,
        )
        self.assertEqual(copied.fragment_compounds, fragment_ion_tree.fragment_compounds)

        self.assertEqual(copied.search_stats, fragment_ion_tree.search_stats)
        self.assertIsNot(copied.search_stats, fragment_ion_tree.search_stats)
        self.assertEqual(
            copied.reaction_sequence_compounds,
            fragment_ion_tree.reaction_sequence_compounds,
        )
        self.assertIsNot(
            copied.reaction_sequence_compounds,
            fragment_ion_tree.reaction_sequence_compounds,
        )

        # The formula cache is not copied; it is rebuilt on demand.
        self.assertEqual(copied.formula_candidate_groups, {})
        copied_group = copied.get_formula_candidate_group("[M+H]+")
        self.assertIsNot(copied_group, group)
        self.assertEqual(_strs(copied_group.formulas), _strs(group.formulas))
        np.testing.assert_array_equal(copied_group.node_indices, group.node_indices)
        self.assertEqual(len(fragment_ion_tree.formula_candidate_groups), 1)

    def test_copy_without_search_stats(self) -> None:
        copied = self._make_fragment_ion_tree().copy()

        self.assertIsNone(copied.search_stats)
        self.assertEqual(copied.reaction_sequence_compounds, {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_fragment_ion_tree(
        self,
        *,
        node_smiles: tuple[str, ...] | None = None,
        rule_set: FragmentIonAdductRuleSet | None = None,
    ) -> FragmentIonTree:
        if node_smiles is None:
            fragment_tree = self.fragment_tree
            node_smiles = NODE_SMILES
        else:
            fragment_tree = _make_fragment_tree(node_smiles)
        rule_set = self.rule_set if rule_set is None else rule_set

        return FragmentIonTree.from_fragment_tree(
            fragment_tree,
            fragment_ion_adduct_rule_set=rule_set,
            fragment_compounds=_make_compounds(node_smiles),
        )


if __name__ == "__main__":
    unittest.main()
