from __future__ import annotations

from typing import Any, Tuple, List, Dict, Iterable
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import tempfile
import unittest

from fredis.domain.fragment.fragmenter import Fragmenter
from fredis.domain.fragment.reaction import ReactionSequence
from fredis.domain.fragment.tree.FragmentTree import FragmentTree
from fredis.domain.fragment.ion_tree.FragmentIonTree import FragmentIonTree
from fredis.domain.fragment.ion_tree.FragmentIonTreeBuilder import FragmentIonTreeBuilder
from fredis.domain.fragment.annotation import PrecursorReaction, ReactionAnnotation, contains
from fredis.domain.mass.tolerance import (
    parse_mass_tolerance,
    format_mass_tolerance,
)
from fredis.libs.mmkit.mmkit import Compound, Adduct, Formula

from .fragment_tree_builder_cases import (
    FragmentTreeBuilderCase,
    FragmentTreeBuilderQuestion,
    make_fragment_tree_builder_questions,
)


class TestFragmenter(unittest.TestCase):
    """Tests for Fragmenter."""

    def test_fragment_all_uses_source_reaction_histories_and_limits(self) -> None:
        from fredis.domain.fragment.reaction import ReactionSequenceResult
        question = make_fragment_tree_builder_questions()[0]
        builder = self._make_fragment_ion_tree_builder(question)
        fragmenter = self._make_fragmenter(builder)
        source = Compound.from_smiles("CCO")
        results = fragmenter.fragment_all(source, max_reaction_count=1)
        self.assertTrue(results)
        self.assertTrue(all(isinstance(result, ReactionSequenceResult) for result in results))
        self.assertTrue(all(len(result.reaction_sequence.reactions) == 1 for result in results))
        seed = results[0].reaction_sequence
        seeded = fragmenter.fragment_all(source, seed_reaction_sequences=(seed,), max_reaction_count=1)
        self.assertEqual(tuple(result.reaction_sequence for result in seeded), (seed,))
        self.assertEqual(fragmenter.fragment_all(source, max_reaction_count=0), ())

    def test_fragmenter_fields_for_questions(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            with self.subTest(question=question.name):
                builder = self._make_fragment_ion_tree_builder(question)
                fragmenter = self._make_fragmenter(builder)

                self.assertIsInstance(fragmenter, Fragmenter)
                self.assertIs(fragmenter.fragment_ion_tree_builder, builder)
                self.assertIs(fragmenter.adduct_rule_set, builder.fragment_ion_adduct_rule_set)
                self.assertEqual(
                    fragmenter.adduct_types,
                    builder.fragment_ion_adduct_rule_set.adduct_types,
                )
                self.assertEqual(fragmenter.tree_max_reaction_count, builder.max_reaction_count)
                self.assertIs(
                    fragmenter.reaction_pattern_set,
                    builder.reaction_pattern_set,
                )
                self.assertEqual(fragmenter.name, builder.name)

    def test_build_fragment_tree_returns_fragment_tree_for_all_cases(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)
            fragmenter = self._make_fragmenter(builder)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)

                    fragment_tree = fragmenter.build_fragment_tree(
                        compound,
                        max_unique_fragment_smiles=-1,
                        max_reaction_combinations=-1,
                        print_info=False,
                    )

                    self.assertIsInstance(fragment_tree, FragmentTree)
                    self.assertEqual(fragment_tree.smiles, compound.smiles)
                    self.assertEqual(fragment_tree.num_nodes, case.expected_num_nodes)
                    self.assertEqual(fragment_tree.num_edges, case.expected_num_edges)
                    self.assertGreaterEqual(
                        fragment_tree.num_transitions,
                        case.expected_min_transitions,
                    )

    def test_build_fragment_ion_tree_returns_fragment_ion_tree_for_all_cases(
        self,
    ) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)
            fragmenter = self._make_fragmenter(builder)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)

                    fragment_ion_tree = fragmenter.build_fragment_ion_tree(
                        compound,
                        max_unique_fragment_smiles=-1,
                        max_reaction_combinations=-1,
                        print_info=False,
                    )

                    self.assertIsInstance(fragment_ion_tree, FragmentIonTree)
                    self.assertEqual(fragment_ion_tree.smiles, compound.smiles)
                    self.assertEqual(
                        fragment_ion_tree.num_nodes,
                        case.expected_num_nodes,
                    )
                    self.assertEqual(
                        fragment_ion_tree.num_edges,
                        case.expected_num_edges,
                    )

    def test_search_limits_skip_only_above_the_observed_value_for_all_cases(
        self,
    ) -> None:
        # A limit equal to a source's own unlimited statistic keeps the tree
        # identical; one below it stops that source with a limit error.
        limits = (
            ("max_unique_fragment_smiles", "num_unique_fragment_smiles"),
            ("max_reaction_combinations", "num_raw_combinations"),
        )
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)
            fragmenter = self._make_fragmenter(builder)

            for case in question.cases:
                compound = Compound.from_smiles(case.smiles)
                unlimited = fragmenter.build_fragment_ion_tree(compound)
                stats = unlimited.search_stats
                for limit_name, stat_name in limits:
                    observed = stats[stat_name]
                    with self.subTest(
                        question=question.name, case=case.name, limit=limit_name
                    ):
                        if observed == 0:
                            continue
                        signature = lambda tree: (
                            tree.node_smiles.tolist(),
                            tree.source_indices.tolist(),
                            tree.target_indices.tolist(),
                            tree.num_transitions,
                        )
                        exact = fragmenter.build_fragment_ion_tree(
                            compound, **{limit_name: observed}
                        )
                        self.assertEqual(signature(exact), signature(unlimited))
                        self.assertEqual(exact.search_stats, stats)
                        self.assertEqual(
                            signature(
                                fragmenter.build_fragment_tree(
                                    compound, **{limit_name: observed}
                                )
                            ),
                            signature(unlimited),
                        )
                        if observed == 1:
                            continue
                        for build in (
                            fragmenter.build_fragment_tree,
                            fragmenter.build_fragment_ion_tree,
                        ):
                            if limit_name == "max_reaction_combinations":
                                retained = build(compound, **{limit_name: observed - 1})
                                # The ion-tree view exposes the retained depth metadata.
                                if hasattr(retained, "search_stats"):
                                    depth = retained.search_stats[
                                        "completed_max_reaction_count"
                                    ]
                                    expected = build(compound, max_reaction_count=depth)
                                    self.assertEqual(
                                        signature(retained), signature(expected)
                                    )
                                self.assertLessEqual(
                                    retained.num_transitions, unlimited.num_transitions
                                )
                                continue
                            with self.assertRaisesRegex(
                                ValueError,
                                rf"^{limit_name} exceeded: limit={observed - 1}, observed>{observed - 1}$",
                            ):
                                build(compound, **{limit_name: observed - 1})


    def test_to_dict(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            with self.subTest(question=question.name):
                builder = self._make_fragment_ion_tree_builder(question)
                fragmenter = self._make_fragmenter(builder)

                data = fragmenter.to_dict()

                self.assertIn("fragment_ion_tree_builder", data)
                self.assertIn("mass_tolerance", data)
                self.assertEqual(
                    data["mass_tolerance"],
                    format_mass_tolerance(fragmenter.mass_tolerance),
                )

    def test_from_dict(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            with self.subTest(question=question.name):
                builder = self._make_fragment_ion_tree_builder(question)
                fragmenter = self._make_fragmenter(builder)

                restored = Fragmenter.from_dict(fragmenter.to_dict())

                self.assertIsInstance(restored, Fragmenter)
                self.assertIsInstance(
                    restored.fragment_ion_tree_builder,
                    FragmentIonTreeBuilder,
                )
                self.assertEqual(
                    format_mass_tolerance(restored.mass_tolerance),
                    format_mass_tolerance(fragmenter.mass_tolerance),
                )
                self.assertEqual(restored.name, fragmenter.name)
                self.assertEqual(restored.tree_max_reaction_count, fragmenter.tree_max_reaction_count)

    def test_to_json_and_from_json(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            with self.subTest(question=question.name):
                builder = self._make_fragment_ion_tree_builder(question)
                fragmenter = self._make_fragmenter(builder)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    path = Path(tmp_dir) / "fragmenter.json"

                    fragmenter.to_json(path)

                    self.assertTrue(path.exists())

                    restored = Fragmenter.from_json(path)

                    self.assertIsInstance(restored, Fragmenter)
                    self.assertEqual(
                        format_mass_tolerance(restored.mass_tolerance),
                        format_mass_tolerance(fragmenter.mass_tolerance),
                    )
                    self.assertEqual(restored.name, fragmenter.name)
                    self.assertEqual(restored.tree_max_reaction_count, fragmenter.tree_max_reaction_count)

    def test_copy(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            with self.subTest(question=question.name):
                builder = self._make_fragment_ion_tree_builder(question)
                fragmenter = self._make_fragmenter(builder)

                copied = fragmenter.copy()

                self.assertIsInstance(copied, Fragmenter)
                self.assertIsNot(copied, fragmenter)
                self.assertIsNot(
                    copied.fragment_ion_tree_builder,
                    fragmenter.fragment_ion_tree_builder,
                )
                self.assertEqual(
                    format_mass_tolerance(copied.mass_tolerance),
                    format_mass_tolerance(fragmenter.mass_tolerance),
                )
                self.assertEqual(copied.name, fragmenter.name)
                self.assertEqual(copied.tree_max_reaction_count, fragmenter.tree_max_reaction_count)

    def test_annotate_peaks_explains_every_peak_with_reaction_combinations(self) -> None:
        """Annotate each peak m/z list of one spectrum."""

        for group in self._make_annotation_test_case_groups():
            with self.subTest(group=group["name"]):
                fragmenter, compound, fragment_ion_tree = self._build_group_tree(group)

                for case in group["test_cases"]:
                    with self.subTest(group=group["name"], case=case["name"]):
                        annotations_by_peak = fragmenter.annotate_peaks(
                            fragment_ion_tree, case["precursor_type"], case["peak_mz_list"])
                        self._assert_case_annotations(fragmenter, fragment_ion_tree, compound, case,
                                                      annotations_by_peak)

    def test_annotate_peak_sets_matches_individual_annotation(self) -> None:
        """Annotate several spectra sharing one tree at once."""

        for group in self._make_annotation_test_case_groups():
            with self.subTest(group=group["name"]):
                fragmenter, compound, fragment_ion_tree = self._build_group_tree(group)
                peak_sets = tuple((case["precursor_type"], case["peak_mz_list"]) for case in group["test_cases"])
                results = fragmenter.annotate_peak_sets(fragment_ion_tree, peak_sets)
                self.assertEqual(len(results), len(group["test_cases"]))

                for case, annotations_by_peak in zip(group["test_cases"], results):
                    with self.subTest(group=group["name"], case=case["name"]):
                        self._assert_case_annotations(fragmenter, fragment_ion_tree, compound, case,
                                                      annotations_by_peak)
                        self.assertEqual(annotations_by_peak, fragmenter.annotate_peaks(
                            fragment_ion_tree, case["precursor_type"], case["peak_mz_list"]))

    def _build_group_tree(self, group: dict[str, Any]) -> Tuple[Fragmenter, Compound, FragmentIonTree]:
        fragmenter = Fragmenter.from_json(group["fragmenter_json"])
        compound = Compound.from_smiles(group.get("smiles", "CC(=O)N[C@@H](CC1=CC=CC=C1)C2=CC(=CC(=O)O2)OC"))
        fragmenter = replace(fragmenter, precursor_candidate_max_reaction_count=group.get(
            "precursor_candidate_max_reaction_count", fragmenter.precursor_candidate_max_reaction_count))
        seeds = self._water_loss_seeds(fragmenter, compound, group.get("seed_water_loss_count", 0))
        fragment_ion_tree = fragmenter.build_fragment_ion_tree(
            compound,
            seed_reaction_sequences=seeds,
            max_unique_fragment_smiles=-1,
            max_reaction_combinations=-1,
            print_info=False,
        )
        return fragmenter, compound, fragment_ion_tree

    def test_multi_reaction_seed_precursor_limit_counts_reactions_not_edges(self) -> None:
        from dataclasses import replace
        from unittest.mock import patch
        question = make_fragment_tree_builder_questions()[0]
        builder = self._make_fragment_ion_tree_builder(question)
        builder = replace(builder, max_reaction_count=2)
        source = Compound.from_smiles("CCCCO")
        seed_result = next(result for result in builder.react_all(source, max_reaction_count=2)
                           if len(result.reaction_sequence.reactions) == 2)
        fragmenter = Fragmenter(builder, parse_mass_tolerance("0.01Da"),
                                precursor_candidate_max_reaction_count=1)
        tree = fragmenter.build_fragment_ion_tree(
            source, seed_reaction_sequences=(seed_result.reaction_sequence,),
            max_reaction_count=2)
        seed_edge = tree.get_out_edges(0)[0]
        index = seed_edge.target_index
        self.assertEqual(tree.node_depths[index], 1)
        self.assertEqual(tree.get_min_reaction_counts()[index], 2)
        main_adduct = fragmenter.adduct_types[0]
        loss = (source.formula - seed_result.compound.formula).normalized
        precursor_type = Adduct.parse(str(main_adduct)[:-2] + "-" + str(loss) + "]" + str(main_adduct)[-1])
        with patch.object(FragmentIonTree, "get_nodes_by_depth", side_effect=AssertionError("graph depth used")):
            self.assertFalse(any(precursor.node_index == index
                                 for precursor in fragmenter.resolve_precursor_reactions(tree, precursor_type)))
            allowed = replace(fragmenter, precursor_candidate_max_reaction_count=2)
            self.assertIn(PrecursorReaction(index, seed_result.reaction_sequence),
                          allowed.resolve_precursor_reactions(tree, precursor_type))
        copied = tree.copy()
        self.assertEqual(copied.get_min_reaction_counts(), tree.get_min_reaction_counts())

    def test_annotation_combination_must_contain_a_precursor_combination(self) -> None:
        question = make_fragment_tree_builder_questions()[0]
        builder = replace(self._make_fragment_ion_tree_builder(question), max_reaction_count=2)
        fragmenter = Fragmenter(builder, parse_mass_tolerance("0.01Da"), precursor_candidate_max_reaction_count=1)
        source = Compound.from_smiles("CCCCC")
        result = next(result for result in builder.react_all(source, max_reaction_count=2)
                      if len(result.reaction_sequence.reactions) == 2)
        a, b = result.reaction_sequence.reactions
        both = result.reaction_sequence
        seq_a, seq_b = ReactionSequence((a,)), ReactionSequence((b,))
        precursor_a = PrecursorReaction(1, seq_a)
        self.assertTrue(contains(both, seq_a))
        self.assertTrue(contains(both, None))
        self.assertFalse(contains(seq_b, seq_a))
        self.assertFalse(contains(None, seq_a))
        # Combinations are unordered: AB contains A no matter which reaction
        # was added first while the tree was built.
        combinations = fragmenter._reaction_combinations_for_node((seq_b, both), frozenset((precursor_a,)))
        self.assertEqual(combinations, ((both, precursor_a),))
        with self.assertRaisesRegex(ValueError, "contain the precursor"):
            ReactionAnnotation(node_index=2, smiles=result.compound.smiles, reaction_sequence=seq_b,
                               precursor=precursor_a, adduct=fragmenter.adduct_types[0],
                               fragment_ion_adduct=fragmenter.adduct_types[0], hydrogen_shift=0,
                               formula=result.compound.formula)

    def test_precursor_selection_setting_roundtrip_and_copy(self) -> None:
        question = make_fragment_tree_builder_questions()[0]
        builder = self._make_fragment_ion_tree_builder(question)
        fragmenter = Fragmenter(builder, parse_mass_tolerance('0.01Da'), precursor_candidate_max_reaction_count=1)
        data = fragmenter.to_dict()
        self.assertEqual(data['precursor_candidate_max_reaction_count'], 1)
        self.assertEqual(data['fragment_ion_tree_builder']['max_reaction_count'], builder.max_reaction_count)
        self.assertEqual(Fragmenter.from_dict(data).precursor_candidate_max_reaction_count, 1)
        self.assertEqual(fragmenter.copy().precursor_candidate_max_reaction_count, 1)
        for value in (True, 1.5, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Fragmenter.from_dict({**data, "precursor_candidate_max_reaction_count": value})
        self.assertEqual(fragmenter.tree_max_reaction_count, builder.max_reaction_count)
        with self.assertRaises(ValueError):
            Fragmenter(builder, fragmenter.mass_tolerance, precursor_candidate_max_reaction_count=-1)

    def test_batched_and_individual_assignment_keep_precursor_contexts_separate(self) -> None:
        source = Compound.from_smiles('CC(=O)N[C@@H](CC1=CC=CC=C1)C2=CC(=CC(=O)O2)OC')
        fragmenter = Fragmenter.from_json(
            Path(__file__).resolve().parents[3] / 'tests/domain/fragment/test_data/fragmenter_single_bond_pos.json')
        tree = fragmenter.build_fragment_ion_tree(source)
        records = ((Adduct.parse('[M+H]+'), (125.0233, 154.0499)),
                   (Adduct.parse('[M+H-C6H6]+'), (125.0233, 154.0499)))
        batch = fragmenter.annotate_peak_sets(tree, records)
        individual = tuple(fragmenter.annotate_peaks(tree, adduct, peaks) for adduct, peaks in records)
        self.assertEqual(batch, individual)
        self.assertGreater(len(batch[0][0]), 0)
        self.assertEqual(len(batch[1][0]), 0)
        self.assertGreater(len(batch[1][1]), 0)
        # The aromatic-only product is prohibited by the hard conflict with
        # the precursor combination, rather than lost by the cache.
        precursor = next(iter(fragmenter.resolve_precursor_reactions(tree, records[1][0]))).reaction_sequence
        target_index = tree.get_node_by_smiles('COc1ccoc(=O)c1').index
        target = tree.get_in_edges(target_index)[0].transitions[0].reaction_sequence
        self.assertTrue(precursor.reactions[0].changed_bond_maps & target.reactions[0].changed_bond_maps)
        from fredis.domain.fragment.reaction import ReactionSequence
        with self.assertRaisesRegex(ValueError, 'share changed Source bonds'):
            ReactionSequence((*precursor.reactions, *target.reactions))

    def _make_annotation_test_case_groups(self) -> list[dict[str, Any]]:
        """Create common test cases for peak annotation tests."""

        return [
            {
                "name": "positive",
                "fragmenter_json": Path(__file__).resolve().parents[3] / "tests/domain/fragment/test_data/fragmenter_single_bond_pos.json",
                "test_cases": [
                    {
                        "name": "protonated",
                        "precursor_type": Adduct.parse("[M+H]+"),
                        "peak_mz_list": [
                            91.0542,
                            125.0233,
                            154.0499,
                            155.0577,
                            229.0859,
                            246.1125,
                            288.1158,
                        ],
                        "expected_precursor_formula": Formula.parse("C16H18NO4+"),
                    },
                    {
                        "name": "sodiated",
                        "precursor_type": Adduct.parse("[M+Na]+"),
                        "peak_mz_list": [
                            91.0542,
                            125.0233,
                            154.0499,
                            155.0577,
                            229.0859,
                            246.1125,
                            310.0977,
                        ],
                        "expected_precursor_formula": Formula.parse("C16H17NNaO4+"),
                    },
                    {
                        "name": "ammonium_adduct",
                        "precursor_type": Adduct.parse("[M+H4N]+"),
                        "peak_mz_list": [
                            91.0542,
                            125.0233,
                            154.0499,
                            155.0577,
                            229.0859,
                            246.1125,
                            305.1503,
                        ],
                        "expected_precursor_formula": Formula.parse("C16H21N2O4+"),
                    },
                    {
                        "name": "protonated_dimer",
                        "precursor_type": Adduct.parse("[2M+H]+"),
                        "peak_mz_list": [
                            91.0542,
                            125.0233,
                            154.0499,
                            155.0577,
                            229.0859,
                            246.1125,
                            575.2392,
                        ],
                        "expected_precursor_formula": Formula.parse("C32H35N2O8+"),
                    },
                    {
                        "name": "benzene_loss",
                        # Source-anchored changed-bond conflicts prohibit the
                        # aromatic-only product after this precursor history.
                        "expected_unassigned_peaks": (125.0233,),
                        "precursor_type": Adduct.parse("[M+H-C6H6]+"),
                        "peak_mz_list": [
                            125.0233,
                            154.0499,
                            155.0577,
                            210.0761,
                        ],
                        "expected_precursor_formula": Formula.parse("C10H12NO4+"),
                        "precursor_reaction_count": 1,
                        "precursor_main_adduct_type": Adduct.parse("[M-H]+"),
                    },
                ],
            },
            {
                "name": "negative",
                "fragmenter_json": Path(__file__).resolve().parents[3] / "tests/domain/fragment/test_data/fragmenter_single_bond_neg.json",
                "test_cases": [
                    {
                        "name": "deprotonated",
                        "precursor_type": Adduct.parse("[M-H]-"),
                        "peak_mz_list": [
                            89.0386,
                            123.0077,
                            152.0342,
                            153.0421,
                            227.0703,
                            244.0968,
                            286.1002,
                        ],
                        "expected_precursor_formula": Formula.parse("C16H16NO4-"),
                    },
                    {
                        "name": "deprotonated_dimer",
                        "precursor_type": Adduct.parse("[2M-H]-"),
                        "peak_mz_list": [
                            89.0386,
                            123.0077,
                            152.0342,
                            153.0421,
                            227.0703,
                            244.0968,
                            573.2235,
                        ],
                        "expected_precursor_formula": Formula.parse("C32H33N2O8-"),
                    },
                ],
            },
        ] + self._make_water_loss_assignment_groups()

    def _make_water_loss_assignment_groups(self) -> list[dict[str, Any]]:
        preset_dir = Path(__file__).resolve().parents[3] / "tests/domain/fragment/test_data"
        groups = []
        scenarios = (
            ("primary_alcohol", "CCCO", "positive", 1, 1, "C3H7+", ("CCC",), 1,
             ("C3H7+", "C2H5+", "CH3+"), ("C2H5O+",)),
            ("secondary_alcohol", "CC(O)C", "positive", 1, 1, "C3H7+", ("CCC",), 1,
             ("C3H7+", "C2H5+", "CH3+"), ("C2H5O+",)),
            ("equivalent_hydroxyl_sites", "OCCO", "positive", 1, 1, "C2H5O+", ("CCO",), 2,
             ("C2H5O+", "CH3+", "C2H3+"), ()),
            ("different_hydroxyl_sites", "OCC(O)CO", "positive", 1, 1, "C3H7O2+",
             ("CC(O)CO", "OCCCO"), 3, ("C3H7O2+", "C2H5O+", "CH3+"), ()),
            ("double_water_loss", "OCCO", "positive", 2, 2, "C2H3+", ("CC",), 2,
             ("C2H3+",), ("CH3+", "C2H5O+")),
            ("seeded_double_water_loss", "OCCO", "positive", 2, 2, "C2H3+", ("CC",), 1,
             ("C2H3+",), ("C2H5O+",)),
            ("seeded_double_loss_over_budget", "OCCO", "positive", 2, 1, "C2H3+", (), 0,
             (), ("C2H3+", "C2H5O+")),
            ("double_loss_over_budget", "OCCO", "positive", 2, 1, "C2H3+", (), 0,
             (), ("C2H3+", "C2H5O+")),
            ("zero_precursor_reaction_budget", "CCCO", "positive", 1, 0, "C3H7+", (), 0,
             (), ("C3H7+", "C2H5+")),
            ("ether_without_hydroxyl", "CCOCC", "positive", 1, 1, "C4H9+", (), 0,
             (), ("C4H9+", "C2H5+")),
            ("negative_water_loss", "CCCO", "negative", 1, 1, "C3H5-", ("CCC",), 1,
             ("C3H5-", "C2H3-", "CH-"), ("C2H3O-",)),
        )
        for name, smiles, polarity, loss_count, limit, formula, structures, count, assigned, unassigned in scenarios:
            suffix = "+" if polarity == "positive" else "-"
            base = "[M+H]+" if polarity == "positive" else "[M-H]-"
            loss = "H2O" if loss_count == 1 else "2H2O"
            precursor = Adduct.parse(base[:-2] + "-" + loss + "]" + suffix)
            main = Adduct.parse("[M" + ("-H" if loss_count == 1 and polarity == "positive" else "-3H") + "]" + suffix)
            peaks = [Formula.parse(text).exact_mass for text in (*assigned, *unassigned)]
            case = dict(name=name, precursor_type=precursor, peak_mz_list=peaks,
                        expected_precursor_formula=Formula.parse(formula),
                        precursor_reaction_count=loss_count,
                        precursor_main_adduct_type=main,
                        expected_precursor_count=count,
                        expected_unassigned_peaks=tuple(Formula.parse(text).exact_mass for text in unassigned),
                        expected_precursor_smiles=structures, water_loss_count=loss_count)
            cases = [case]
            # The same tree must retain oxygen-containing assignments for the
            # normal precursor while excluding them after its OH has been lost.
            if name == "primary_alcohol":
                cases.append(dict(name="normal_precursor_same_tree", precursor_type=Adduct.parse(base),
                    peak_mz_list=[Formula.parse("C3H9O+").exact_mass, Formula.parse("C2H5O+").exact_mass],
                    expected_precursor_formula=Formula.parse("C3H9O+")))
            groups.append(dict(name=name, smiles=smiles,
                fragmenter_json=preset_dir / ("fragmenter_single_bond_pos.json" if polarity == "positive" else "fragmenter_single_bond_neg.json"),
                precursor_candidate_max_reaction_count=limit, seed_water_loss_count=loss_count if name.startswith("seeded_") else 0, test_cases=cases))
        return groups

    def _water_loss_seeds(self, fragmenter: Fragmenter, source: Compound, count: int) -> Tuple[ReactionSequence, ...]:
        if not count:
            return ()
        hydroxyl_maps = sorted(atom.GetAtomMapNum() for atom in source.mapped_mol.GetAtoms()
                               if atom.GetSymbol() == "O" and atom.GetTotalNumHs() == 1)
        reactions = fragmenter.fragment_ion_tree_builder.create_reaction_instances(source)
        selected = tuple(next(reaction for reaction in reactions if reaction.discarded_atom_maps == frozenset((atom_map,)))
                         for atom_map in hydroxyl_maps[:count])
        self.assertEqual(len(selected), count)
        sequence = ReactionSequence(selected)
        self.assertEqual(len(sequence.reactions), count)
        return (sequence,)

    def _assert_case_annotations(
        self, fragmenter: Fragmenter, tree: FragmentIonTree, source: Compound,
        case: dict[str, Any], annotations_by_peak: Tuple[Tuple[ReactionAnnotation, ...], ...],
    ) -> None:
        precursors = fragmenter.resolve_precursor_reactions(tree, case["precursor_type"])
        expected_count = case.get("expected_precursor_count", 1)
        if case.get("water_loss_count") == 2 and expected_count:
            # Both hydroxyl groups are lost in one unordered combination.
            expected_count = 1
        self.assertEqual(len(precursors), expected_count)
        main_adduct = case.get("precursor_main_adduct_type", case["precursor_type"])
        for precursor in precursors:
            self.assertEqual(len(precursor.reaction_sequence.reactions) if precursor.reaction_sequence else 0,
                             case.get("precursor_reaction_count", 0))
            self.assertEqual(main_adduct.apply_to_formula(tree.fragment_compounds[precursor.node_index].formula).normalized,
                             case["expected_precursor_formula"])

        self.assertEqual(len(annotations_by_peak), len(case["peak_mz_list"]))
        for peak_mz, annotations in zip(case["peak_mz_list"], annotations_by_peak):
            with self.subTest(peak_mz=peak_mz):
                if peak_mz in case.get("expected_unassigned_peaks", ()):
                    self.assertEqual(annotations, ())
                    continue
                self.assertGreater(len(annotations), 0, msg=f"No annotation for peak m/z {peak_mz}")
                # Only the explanations with the fewest reactions are kept.
                self.assertEqual(len({annotation.reaction_count for annotation in annotations}), 1)
                for annotation in annotations:
                    self.assertLessEqual(annotation.reaction_count, fragmenter.tree_max_reaction_count)
                    self.assertIn(annotation.precursor, precursors)
                    self.assertTrue(contains(annotation.reaction_sequence, annotation.precursor.reaction_sequence))
                    self.assertTrue(fragmenter.mass_tolerance.within(
                        observed=peak_mz, theoretical=annotation.formula.exact_mass))
                    self.assertEqual(annotation.smiles, tree.get_node(annotation.node_index).smiles)

        if "water_loss_count" not in case:
            return
        expected_smiles = set(case["expected_precursor_smiles"])
        self.assertEqual({tree.get_node(precursor.node_index).smiles for precursor in precursors}, expected_smiles)
        # Independently check the ion formula against literal H2O subtraction.
        base = Adduct.parse("[M+H]+" if case["precursor_type"].charge > 0 else "[M-H]-")
        expected_formula = (base.apply_to_formula(source.formula)
                            - Formula.parse("H2O") * case["water_loss_count"]).normalized
        self.assertEqual(case["expected_precursor_formula"], expected_formula)
        hydroxyl_maps = {atom.GetAtomMapNum() for atom in source.mapped_mol.GetAtoms()
                         if atom.GetSymbol() == "O" and atom.GetTotalNumHs() == 1}
        source_maps = frozenset(atom.GetAtomMapNum() for atom in source.mapped_mol.GetAtoms())
        discarded_sites = set()
        for precursor in precursors:
            sequence = precursor.reaction_sequence
            discarded = source_maps - sequence.retained_atom_maps
            self.assertEqual(len(discarded), case["water_loss_count"])
            self.assertTrue(discarded <= hydroxyl_maps)
            expected_cuts = frozenset(tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())))
                for bond in source.mapped_mol.GetBonds()
                if bond.GetBeginAtom().GetAtomMapNum() in discarded or bond.GetEndAtom().GetAtomMapNum() in discarded)
            self.assertEqual(sequence.cut_bond_maps, expected_cuts)
            self.assertEqual(sequence.bond_updates, frozenset())
            # Every combination is executed against the same original Source.
            products = sequence.compile(source).run(source)
            self.assertEqual(len(products), 1)
            self.assertEqual(frozenset(atom.GetAtomMapNum() for atom in products[0].GetAtoms()), sequence.retained_atom_maps)
            product = Chem.Mol(products[0])
            for atom in product.GetAtoms():
                atom.SetAtomMapNum(0)
            self.assertEqual(Chem.MolToSmiles(product), tree.get_node(precursor.node_index).smiles)
            discarded_sites.add(discarded)
        self.assertEqual(len(discarded_sites), expected_count)

    def _make_fragmenter(
        self,
        builder: FragmentIonTreeBuilder,
    ) -> Fragmenter:
        return Fragmenter(
            fragment_ion_tree_builder=builder,
            mass_tolerance=parse_mass_tolerance("0.01Da"),
        )

    def _make_fragment_ion_tree_builder(
        self,
        question: FragmentTreeBuilderQuestion,
    ) -> FragmentIonTreeBuilder:
        if question.fragment_ion_adduct_rule_set is None:
            raise ValueError(
                "question.fragment_ion_adduct_rule_set must not be None."
            )

        return FragmentIonTreeBuilder(
            max_reaction_count=question.builder.max_reaction_count,
            reaction_pattern_set=question.builder.reaction_pattern_set.copy(),
            fragment_ion_adduct_rule_set=(
                question.fragment_ion_adduct_rule_set.copy()
            ),
        )

    def test_resolve_precursor_reactions_reuses_same_tree_without_rdkit(self) -> None:
        """Precursor resolution walks the already-built tree; it never reacts again."""
        fragmenter = Fragmenter.from_json("tests/domain/fragment/test_data/fragmenter_single_bond_pos.json")
        source = Compound.from_smiles("CCCO")
        tree = fragmenter.build_fragment_ion_tree(source)

        root_only = fragmenter.resolve_precursor_reactions(tree, Adduct.parse("[M+H]+"))
        self.assertEqual({(pa.node_index, pa.reaction_sequence) for pa in root_only}, {(0, None)})

        with patch.object(rdChemReactions.ChemicalReaction, "RunReactants",
                          side_effect=AssertionError("RDKit re-invoked")):
            dehydrated = fragmenter.resolve_precursor_reactions(tree, Adduct.parse("[M+H-H2O]+"))

        self.assertEqual(len(dehydrated), 1)
        precursor_reaction = next(iter(dehydrated))
        self.assertIsNotNone(precursor_reaction.reaction_sequence)
        self.assertNotEqual(precursor_reaction.node_index, 0)
        product = precursor_reaction.reaction_sequence.compile(source).run(source)[0]
        self.assertEqual(Compound(product).smiles, "CCC")


if __name__ == "__main__":
    unittest.main()