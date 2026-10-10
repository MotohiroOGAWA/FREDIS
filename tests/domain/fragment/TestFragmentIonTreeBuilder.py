from __future__ import annotations

import unittest
from fredis.domain.fragment.reaction import ReactionSequence

from fredis.libs.mmkit.mmkit import Compound

from fredis.domain.fragment.ion_tree.FragmentIonTree import (
    FragmentIonTree,
)
from fredis.domain.fragment.ion_tree.FragmentIonTreeBuilder import (
    FragmentIonTreeBuilder,
)

from .fragment_tree_builder_cases import (
    ExpectedFragmentTreeEdge,
    FragmentTreeBuilderQuestion,
    make_fragment_tree_builder_questions,
)


class TestFragmentIonTreeBuilder(unittest.TestCase):
    """Tests for FragmentIonTreeBuilder."""

    def test_build_returns_fragment_ion_tree(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    compound = Compound.from_smiles(case.smiles)

                    fragment_ion_tree = builder.build(compound)

                    self.assertIsInstance(fragment_ion_tree, FragmentIonTree)
                    self.assertEqual(fragment_ion_tree.smiles, compound.smiles)

    def test_build_keeps_expected_tree_size(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(
                        Compound.from_smiles(case.smiles)
                    )

                    self.assertEqual(
                        fragment_ion_tree.num_nodes,
                        case.expected_num_nodes,
                    )
                    self.assertEqual(
                        fragment_ion_tree.num_edges,
                        case.expected_num_edges,
                    )

    def test_build_keeps_expected_nodes_by_depth(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(
                        Compound.from_smiles(case.smiles)
                    )

                    actual_node_smiles_by_depth = (
                        self._collect_node_smiles_by_depth(fragment_ion_tree)
                    )

                    self.assertEqual(
                        actual_node_smiles_by_depth,
                        case.expected_node_smiles_by_depth,
                    )

    def test_build_keeps_expected_edges(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(
                        Compound.from_smiles(case.smiles)
                    )

                    for expected_edge in case.expected_edges:
                        edge = self._find_edge_by_smiles(
                            fragment_ion_tree,
                            expected_edge,
                        )

                        self.assertIsNotNone(edge)

                        transition_count = len(edge.transitions)
                        self.assertGreaterEqual(
                            transition_count,
                            expected_edge.min_transition_count,
                        )

    def test_build_has_expected_min_transitions(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(
                        Compound.from_smiles(case.smiles)
                    )

                    actual_transition_count = sum(
                        len(fragment_ion_tree.get_edge(edge_index).transitions)
                        for edge_index in range(fragment_ion_tree.num_edges)
                    )

                    self.assertGreaterEqual(
                        actual_transition_count,
                        case.expected_min_transitions,
                    )

    def test_build_has_expected_hydrogen_shift_candidates(self) -> None:
        """Every adduct rule's formula candidates use the shared hydrogen-shift range."""

        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)
            rule_set = question.fragment_ion_adduct_rule_set

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(Compound.from_smiles(case.smiles))
                    self.assertEqual(fragment_ion_tree.hydrogen_shifts, rule_set.hydrogen_shifts)
                    for rule in rule_set.adduct_rules:
                        group = fragment_ion_tree.get_formula_candidate_group(rule.adduct_type)
                        self.assertTrue(set(group.hydrogen_shifts.tolist()) <= set(rule_set.hydrogen_shifts))

    def test_build_makes_every_fragment_ion_adduct_applicable_to_neutral_nodes(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            for case in question.cases:
                with self.subTest(question=question.name, case=case.name):
                    fragment_ion_tree = builder.build(Compound.from_smiles(case.smiles))
                    self.assertTrue(bool(fragment_ion_tree.node_is_neutral.all()))
                    for rule in question.fragment_ion_adduct_rule_set.adduct_rules:
                        for node_index in range(fragment_ion_tree.num_nodes):
                            self.assertEqual(
                                fragment_ion_tree.get_fragment_ion_adducts(node_index, rule.adduct_type),
                                rule.fragment_ion_adducts,
                            )

    def test_build_stores_one_compound_and_search_metadata_per_tree(self) -> None:
        question = make_fragment_tree_builder_questions()[0]
        builder = self._make_fragment_ion_tree_builder(question)

        for case in question.cases:
            with self.subTest(case=case.name):
                fragment_ion_tree = builder.build(Compound.from_smiles(case.smiles))
                self.assertEqual(len(fragment_ion_tree.fragment_compounds), fragment_ion_tree.num_nodes)
                self.assertEqual(
                    [compound.smiles for compound in fragment_ion_tree.fragment_compounds],
                    fragment_ion_tree.node_smiles.tolist(),
                )
                self.assertIsNotNone(fragment_ion_tree.search_stats)
                for sequence, compound in fragment_ion_tree.reaction_sequence_compounds.items():
                    self.assertIn(compound.smiles, fragment_ion_tree.node_smiles.tolist())
                    self.assertGreater(len(sequence.reactions), 0)

    def test_copy_returns_fragment_ion_tree_builder(self) -> None:
        for question in make_fragment_tree_builder_questions():
            if question.fragment_ion_adduct_rule_set is None:
                continue

            builder = self._make_fragment_ion_tree_builder(question)

            copied = builder.copy()

            self.assertIsInstance(copied, FragmentIonTreeBuilder)
            self.assertEqual(copied.max_reaction_count, builder.max_reaction_count)
            self.assertIsNot(
                copied.reaction_pattern_set,
                builder.reaction_pattern_set,
            )
            self.assertIsNot(
                copied.fragment_ion_adduct_rule_set,
                builder.fragment_ion_adduct_rule_set,
            )

    def test_ion_tree_copy_preserves_transitions_and_compounds(self) -> None:
        question = make_fragment_tree_builder_questions()[0]
        builder = self._make_fragment_ion_tree_builder(question)
        source = Compound.from_smiles("CCO")
        tree = builder.build(source, max_reaction_count=1)
        copied = tree.copy()
        self.assertEqual(copied.num_transitions, tree.num_transitions)
        for index in range(tree.num_edges):
            self.assertEqual(copied.get_edge(index).transitions, tree.get_edge(index).transitions)
        self.assertEqual(copied.fragment_compounds, tree.fragment_compounds)
        self.assertEqual(copied.search_stats, tree.search_stats)
        self.assertIsNot(copied.search_stats, tree.search_stats)
        self.assertIsNot(copied.formula_candidate_groups, tree.formula_candidate_groups)

    def test_reaction_limit_and_seeds_forwarded_to_tree_builder(self) -> None:
        question = make_fragment_tree_builder_questions()[0]
        builder = self._make_fragment_ion_tree_builder(question)
        source = Compound.from_smiles('CCO')
        reactions = builder.create_reaction_instances(source)
        oxygen_map = next(a.GetAtomMapNum() for a in source.mapped_mol.GetAtoms() if a.GetSymbol() == 'O')
        retained = next(a for a in reactions if a.discarded_atom_maps == frozenset((oxygen_map,)))
        seed = ReactionSequence((retained,))
        for method in (builder.build, builder.build_fragment_tree):
            with self.subTest(method=method.__name__):
                tree = method(source, seed_reaction_sequences=(seed,), max_reaction_count=2)
                self.assertEqual(tree.num_nodes, 3)
                self.assertEqual(len(tree.get_out_edges(0)), 1)
                self.assertTrue(tree.get_out_edges(0)[0].transitions[0].is_seed)
                self.assertEqual(tree.get_out_edges(0)[0].transitions[0].reaction_sequence, seed)
        data = builder.to_dict()
        self.assertEqual(data['max_reaction_count'], builder.max_reaction_count)
        self.assertEqual(FragmentIonTreeBuilder.from_dict(data).to_dict(), data)
        self.assertEqual(builder.copy().to_dict(), data)

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

    def _collect_node_smiles_by_depth(
        self,
        fragment_ion_tree: FragmentIonTree,
    ) -> dict[int, tuple[str, ...]]:
        node_smiles_by_depth: dict[int, list[str]] = {}

        for node_index in range(fragment_ion_tree.num_nodes):
            node = fragment_ion_tree.get_node(node_index)
            depth = int(fragment_ion_tree.node_depths[node_index])

            node_smiles_by_depth.setdefault(depth, []).append(node.smiles)

        return {
            depth: tuple(smiles_list)
            for depth, smiles_list in sorted(node_smiles_by_depth.items())
        }

    def _find_edge_by_smiles(
        self,
        fragment_ion_tree: FragmentIonTree,
        expected_edge: ExpectedFragmentTreeEdge,
    ):
        for edge_index in range(fragment_ion_tree.num_edges):
            edge = fragment_ion_tree.get_edge(edge_index)

            source_node = fragment_ion_tree.get_node(edge.source_index)
            target_node = fragment_ion_tree.get_node(edge.target_index)

            if (
                source_node.smiles == expected_edge.source_smiles
                and target_node.smiles == expected_edge.target_smiles
            ):
                return edge

        return None


if __name__ == "__main__":
    unittest.main()