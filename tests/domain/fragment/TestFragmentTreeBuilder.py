from __future__ import annotations

from itertools import combinations
import random
import unittest
from unittest.mock import patch

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from fredis.domain.fragment.reaction import (
    ReactionInstance, ReactionSequence, ReactionPatternSet, CompositeReaction,
)
from fredis.domain.fragment.reaction._ReactionPattern import _ReactionPattern, ProductRule
from fredis.domain.fragment.reaction import (
    ReactionSequenceSearch, FragmentTreeLimitExceeded, ReactionRelations,
)
from fredis.domain.fragment.tree import FragmentTreeBuilder
from fredis.libs.mmkit.mmkit import Compound


def pattern(reactant: str, product: str) -> _ReactionPattern:
    return _ReactionPattern.from_rules(reactant_smarts=reactant,
                                      products=(ProductRule('test', product),))


def builder(*patterns: _ReactionPattern, limit: int = 3) -> FragmentTreeBuilder:
    return FragmentTreeBuilder(limit, ReactionPatternSet.from_patterns(patterns))


def reaction(
    pattern_id: int,
    roles: tuple[int, ...],
    *,
    retained: tuple[int, ...] = (1, 2, 3, 4),
    matched: tuple[tuple[int, int], ...] = (),
    cuts: tuple[tuple[int, int], ...] = (),
    updates: tuple[tuple[int, int, Chem.BondType], ...] = (),
) -> ReactionInstance:
    return ReactionInstance(pattern_id, 0, roles, frozenset(retained),
                          frozenset((1, 2, 3, 4)) - frozenset(retained),
                          frozenset(matched), frozenset(cuts), frozenset(updates),
                          product_molecule_id=0)


class TestPrimitiveReactionInstances(unittest.TestCase):
    def test_all_matches_products_and_query_order_without_reactions(self) -> None:
        source = Compound.from_smiles('[CH3:10][CH2:20][OH:30]')
        build = builder(pattern('[C:7]-[C:2]', '[C:7].[C:2]'))
        calls = []
        original = Chem.Mol.GetSubstructMatches

        def matching(mol: Chem.Mol, query: Chem.Mol, **kwargs: object) -> tuple[tuple[int, ...], ...]:
            calls.append(query)
            return original(mol, query, **kwargs)

        with patch.object(Chem.Mol, 'GetSubstructMatches', matching), \
             patch.object(Chem.Mol, 'HasSubstructMatch', side_effect=AssertionError('per-reaction matching')), \
             patch.object(rdChemReactions.ChemicalReaction, 'RunReactants',
                          side_effect=AssertionError('primitive reaction')):
            reactions = build.create_reaction_instances(source)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(reactions), 4)
        self.assertEqual({a.source_atom_maps for a in reactions}, {(10, 20), (20, 10)})
        self.assertEqual({a.product_molecule_id for a in reactions}, {0, 1})
        self.assertEqual(reactions, tuple(sorted(reactions, key=lambda a: a.key)))
        a = next(a for a in reactions if a.source_atom_maps == (10, 20) and a.product_molecule_id == 1)
        self.assertEqual(a.retained_atom_maps, frozenset((20, 30)))
        self.assertEqual(a.discarded_atom_maps, frozenset((10,)))
        self.assertEqual(a.cut_bond_maps, frozenset(((10, 20),)))

    def test_template_inspection_once_for_all_matches(self) -> None:
        from fredis.domain.fragment.reaction import ReactionInstanceGenerator as generator
        source = Compound.from_smiles('CCCCCC')
        build = builder(pattern('[C:1]-[C:2]', '[C:1]'))
        with patch.object(generator, '_reaction_templates', wraps=generator._reaction_templates) as parsed:
            reactions = build.create_reaction_instances(source)
        self.assertEqual(parsed.call_count, 1)
        self.assertEqual(len(reactions), 10)

    def test_whole_reaction_cuts_and_product_order_update(self) -> None:
        source = Compound.from_smiles('[CH3:10][CH2:20][CH3:30]')
        build = builder(pattern('[C:7]-[C:2]-[C:9]', '[C:7]=[C:2].[C:9]'))
        reactions = build.create_reaction_instances(source)
        a = next(a for a in reactions if a.source_atom_maps == (10, 20, 30)
                 and a.product_molecule_id == 0)
        self.assertEqual(a.retained_atom_maps, frozenset((10, 20)))
        self.assertEqual(a.cut_bond_maps, frozenset(((20, 30),)))
        self.assertEqual(a.bond_updates, frozenset(((10, 20, Chem.BondType.DOUBLE),)))
        self.assertEqual(a.changed_bond_maps, frozenset(((10, 20), (20, 30))))
        self.assertEqual(Compound(ReactionSequence((a,)).compile(source).run(source)[0]).smiles, 'C=C')

    def test_multiple_anchor_components_and_creation_are_unsupported(self) -> None:
        source = Compound.from_smiles('CC')
        for rhs in ('([C:1].[C:2])', '[C:1]-[C:2]-[O:3]'):
            with self.subTest(rhs=rhs):
                self.assertEqual(builder(pattern('[C:1]-[C:2]', rhs)).create_reaction_instances(source), ())
        self.assertEqual(builder(pattern('[C:1]-[C:2]-[C:3]', '[C:1]1[C:2][C:3]1'))
                         .create_reaction_instances(Compound.from_smiles('CCC')), ())

    def test_unspecified_atom_charge_is_preserved(self) -> None:
        source = Compound.from_smiles('[CH3:1][O-:2]')
        build = builder(pattern('[O-:1]-[C:2]', '[O:1]'))
        reactions = build.create_reaction_instances(source)
        self.assertEqual(len(reactions), 1)
        self.assertEqual(build.react_all(source)[0].compound.charge, -1)
        self.assertEqual(builder(pattern('[O-:1]-[C:2]', '[O+0:1]'))
                         .create_reaction_instances(source), ())
        self.assertEqual(builder(pattern('[O-:1]-[C:2]', '[N:1]'))
                         .create_reaction_instances(source), ())

    def test_each_pattern_matches_once(self) -> None:
        build = builder(pattern('[C:1]-[O:2]', '[C:1]'), pattern('[C:1]-[N:2]', '[C:1]'))
        original = Chem.Mol.GetSubstructMatches
        calls = []

        def matching(mol: Chem.Mol, query: Chem.Mol, **kwargs: object) -> tuple[tuple[int, ...], ...]:
            calls.append(query)
            return original(mol, query, **kwargs)

        with patch.object(Chem.Mol, 'GetSubstructMatches', matching):
            build.build(Compound.from_smiles('NC(O)C'))
        self.assertEqual(len(calls), 2)


class TestReactionSequenceSearch(unittest.TestCase):
    def test_independent_three_reactions_without_permutations_or_rdkit(self) -> None:
        reactions = tuple(reaction(i, edge, matched=(edge,), cuts=(edge,))
                        for i, edge in enumerate(((1, 2), (2, 3), (3, 4))))
        search = ReactionSequenceSearch(reactions, max_reaction_count=3)
        with patch.object(Chem.Mol, 'GetSubstructMatches', side_effect=AssertionError('matching')), \
             patch.object(rdChemReactions.ChemicalReaction, 'RunReactants', side_effect=AssertionError('reaction')):
            candidates = search.enumerate()
        sequences = {c.reaction_sequence for c in candidates}
        self.assertEqual(len(sequences), 7)
        self.assertEqual(sum(len(s.reactions) == 3 for s in sequences), 1)
        self.assertEqual(ReactionSequence(reactions), ReactionSequence(reversed(reactions)))
        self.assertEqual(len(candidates), 7)

    def test_search_matches_small_exhaustive_combination_oracle(self) -> None:
        edges = tuple(combinations((1, 2, 3, 4), 2))
        rng = random.Random(0)
        for trial in range(12):
            reactions = tuple(reaction(i, (1, 2, 3, 4),
                                   matched=tuple(edge for edge in edges if edge == cut or rng.random() < 0.25),
                                   cuts=(cut,)) for i, cut in enumerate(edges))
            relations = ReactionRelations.from_reactions(reactions)
            for limit in (1, 2, 3):
                for seeded in (False, True):
                    with self.subTest(trial=trial, limit=limit, seeded=seeded):
                        expected = set()
                        for count in range(1, limit + 1):
                            for indices in combinations(range(len(reactions)), count):
                                if seeded and len(reactions) - 1 not in indices:
                                    continue
                                if relations.rejection(indices) is None:
                                    expected.add(ReactionSequence(reactions[i] for i in indices))
                        seeds = (ReactionSequence((reactions[-1],)),) if seeded else None
                        search = ReactionSequenceSearch(reactions, max_reaction_count=limit,
                                                      seed_reaction_sequences=seeds)
                        actual = {candidate.reaction_sequence for candidate in search.enumerate()}
                        self.assertEqual(actual, expected)

    def test_hard_conflict_pruned_before_compilation(self) -> None:
        a = reaction(0, (1, 2), matched=((1, 2),), cuts=((1, 2),))
        b = reaction(1, (2, 1), matched=((2, 1),), updates=((2, 1, Chem.BondType.DOUBLE),))
        search = ReactionSequenceSearch((a, b), max_reaction_count=2)
        candidates = search.enumerate()
        self.assertTrue(all(len(c.reaction_sequence.reactions) == 1 for c in candidates))
        self.assertEqual(search.stats.num_hard_conflict_pruned, 1)

    def test_matched_context_change_does_not_create_reaction_ordering(self) -> None:
        a = reaction(0, (1, 2), matched=((1, 2),), cuts=((1, 2),))
        b = reaction(1, (1, 2, 3), matched=((1, 2), (2, 3)), cuts=((2, 3),))
        relations = ReactionRelations.from_reactions((a, b))
        self.assertEqual(relations.invalidation_mask, (0, 0))
        search = ReactionSequenceSearch((a, b), max_reaction_count=2)
        self.assertIn(ReactionSequence((a, b)), {c.reaction_sequence for c in search.enumerate()})

    def test_unchanged_context_overlap_is_allowed(self) -> None:
        a = reaction(0, (1, 2, 3), matched=((1, 2), (2, 3)), cuts=((1, 2),))
        b = reaction(1, (2, 3, 4), matched=((2, 3), (3, 4)), cuts=((3, 4),))
        search = ReactionSequenceSearch((a, b), max_reaction_count=2)
        self.assertIn(ReactionSequence((a, b)), {c.reaction_sequence for c in search.enumerate()})

    def test_multiple_simultaneous_reactions_need_no_ordering(self) -> None:
        a = reaction(0, (1, 2, 3), matched=((1, 2), (2, 3)), cuts=((1, 2),))
        b = reaction(1, (1, 2, 3), matched=((1, 2), (2, 3)), cuts=((2, 3),))
        search = ReactionSequenceSearch((a, b), max_reaction_count=2)
        self.assertIn(ReactionSequence((a,b)),{c.reaction_sequence for c in search.enumerate()})
        self.assertEqual(search.stats.num_invalidated_reaction_pruned, 0)
        c = reaction(2, (1, 2, 3, 4), matched=((3, 4), (1, 2)), cuts=((3, 4),))
        b = reaction(1, (2, 3, 4), matched=((2, 3), (3, 4)), cuts=((2, 3),))
        search = ReactionSequenceSearch((a, b, c), max_reaction_count=3)
        sequences = {x.reaction_sequence for x in search.enumerate()}
        self.assertEqual(sum(len(s.reactions) == 2 for s in sequences), 3)
        self.assertTrue(any(len(s.reactions) == 3 for s in sequences))
        self.assertEqual(search.stats.num_invalidated_reaction_pruned, 0)

    def test_every_reaction_must_retain_every_other_reaction_source_match(self) -> None:
        a = reaction(0, (1, 2), retained=(1, 2, 3))
        b = reaction(1, (3, 4), matched=((3, 4),), cuts=((3, 4),))
        relations = ReactionRelations.from_reactions((a, b))
        self.assertEqual(relations.invalidation_mask, (2, 0))
        self.assertEqual(relations.rejection((0, 1)), 'invalidated_reaction')
        search = ReactionSequenceSearch((a, b), max_reaction_count=2)
        self.assertTrue(all(len(c.reaction_sequence.reactions)==1 for c in search.enumerate()))
        # Reverse containment is independently required as well.
        b = reaction(1, (2, 3), retained=(2, 3, 4))
        relations = ReactionRelations.from_reactions((a, b))
        self.assertEqual(relations.invalidation_mask, (0, 1))
        self.assertEqual(relations.rejection((0, 1)), 'invalidated_reaction')

    def test_compatible_reactions_are_preserved(self) -> None:
        deletion = reaction(2, (3, 4), retained=(1, 2, 3))
        no_effect = reaction(0, (1, 2))
        self.assertEqual(set(ReactionSequence((deletion, no_effect)).reactions), {deletion, no_effect})
        for edit in (reaction(0, (1, 2), matched=((1, 2),), cuts=((1, 2),)),
                     reaction(0, (1, 2), matched=((1, 2),), updates=((1, 2, Chem.BondType.DOUBLE),))):
            with self.subTest(edit=edit):
                self.assertEqual(len(ReactionSequence((deletion, edit)).reactions), 2)
        search = ReactionSequenceSearch((deletion, no_effect), max_reaction_count=2)
        sequences = {c.reaction_sequence for c in search.enumerate()}
        self.assertTrue(all(len(s.reactions) <= 2 for s in sequences))
        self.assertIn(ReactionSequence((deletion, no_effect)), sequences)

    def test_limit_counts_all_distinct_reactions(self) -> None:
        a = reaction(0, (1,))
        b = reaction(1, (2,), retained=(1, 2, 3))
        c = reaction(2, (1, 2), matched=((1, 2),), cuts=((1, 2),))
        search = ReactionSequenceSearch((a, b, c), max_reaction_count=2)
        sequences = {candidate.reaction_sequence for candidate in search.enumerate()}
        self.assertNotIn(ReactionSequence((a, b, c)), sequences)
        self.assertTrue(all(len(sequence.reactions) <= 2 for sequence in sequences))

    def test_discarding_another_reaction_center_rejects_combination(self) -> None:
        a = reaction(0, (1, 2), retained=(1, 2, 3))
        b = reaction(1, (3,), retained=(3, 4))
        search = ReactionSequenceSearch((a, b), max_reaction_count=2)
        self.assertEqual(search.relations.invalidation_mask, (0, 1))
        self.assertEqual(search.relations.rejection((0,1)),'invalidated_reaction')
        with self.assertRaises(ValueError):ReactionSequence((a,b))
        self.assertTrue(all(len(c.reaction_sequence.reactions)==1 for c in search.enumerate()))

    def test_effect_signature_ignores_discarded_bond_edits(self) -> None:
        a = reaction(0, (3, 4), retained=(1, 2), matched=((3, 4),), cuts=((3, 4),))
        b = reaction(1, (1, 2), retained=(1, 2))
        self.assertNotEqual(ReactionSequence((a,)).key, ReactionSequence((b,)).key)
        self.assertEqual(ReactionSequence((a,)).effect_key, ReactionSequence((b,)).effect_key)


class TestFragmentTreeBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.source = Compound.from_smiles('[CH3:1][C:2]([OH:3])([NH2:4])[SH:5]')
        self.builder = builder(*(pattern(f'[C:1]-[{symbol}:2]', '[C:1]') for symbol in ('O', 'N', 'S')))

    def test_build_anchored_to_source_and_counter_evidence(self) -> None:
        run = CompositeReaction.run
        compile_sequence = ReactionSequence.compile
        executed = []
        compiled = []

        def compiling(sequence: ReactionSequence, source: Compound | Chem.Mol) -> CompositeReaction:
            self.assertIs(source, self.source)
            compiled.append(sequence.effect_key)
            return compile_sequence(sequence, source)

        def running(reaction: CompositeReaction, source: Compound | Chem.Mol) -> tuple[Chem.Mol, ...]:
            self.assertIs(source, self.source)
            executed.append(reaction.reaction_sequence.effect_key)
            return run(reaction, source)

        with patch.object(ReactionSequence, 'compile', compiling), \
             patch.object(CompositeReaction, 'run', running):
            result = self.builder._build_result(self.source)
        tree = result['fragment_tree']
        stats = result['search_stats']
        self.assertEqual(tree.get_node(0).smiles, self.source.smiles)
        self.assertEqual(tree.num_nodes, 8)
        self.assertEqual(tree.num_transitions, 12)
        self.assertEqual(stats['num_primitive_reactions'], 3)
        self.assertEqual(stats['num_compiled_sequences'], 7)
        self.assertEqual(stats['num_rdkit_run_reactants'], 7)
        self.assertEqual(len(executed), len(set(executed)))
        self.assertEqual(compiled, executed)
        self.assertTrue(any(len(state.reaction_sequence.reactions) == 3
                            for state in result['expansion_states'] if state.reaction_sequence))

    def test_none_and_empty_seeds_start_at_source(self) -> None:
        for seeds in (None, ()):
            result = self.builder._build_result(self.source, seed_reaction_sequences=seeds)
            self.assertEqual(result['search_stats']['num_rdkit_run_reactants'], 7)
            self.assertFalse(any(t.is_seed for i in range(result['fragment_tree'].num_edges)
                                 for t in result['fragment_tree'].get_edge(i).transitions))

    def test_single_seed_and_lower_canonical_index_addition(self) -> None:
        reactions = self.builder.create_reaction_instances(self.source)
        seed = ReactionSequence((reactions[-1],))
        result = self.builder._build_result(self.source, seed_reaction_sequences=(seed,), max_reaction_count=2)
        tree = result['fragment_tree']
        self.assertEqual(len(tree.get_out_edges(0)), 1)
        transition = tree.get_out_edges(0)[0].transitions[0]
        self.assertTrue(transition.is_seed)
        self.assertEqual(transition.reaction_sequence, seed)
        sequences = [s.reaction_sequence for s in result['expansion_states'] if s.reaction_sequence]
        self.assertEqual(len(sequences), 3)
        self.assertTrue(all(len(s.reactions) <= 2 for s in sequences))
        self.assertTrue(all(reactions[-1] in s.reactions for s in sequences))

    def test_multiple_seeds_keep_histories_and_execute_effect_once(self) -> None:
        reactions = self.builder.create_reaction_instances(self.source)
        seeds = tuple(ReactionSequence((a,)) for a in reactions[:2])
        result = self.builder._build_result(self.source, seed_reaction_sequences=seeds)
        tree = result['fragment_tree']
        self.assertEqual(len(tree.get_out_edges(0)), 2)
        self.assertEqual(result['search_stats']['num_rdkit_run_reactants'], 6)
        triple = [s for s in result['expansion_states'] if s.reaction_sequence
                  and len(s.reaction_sequence.reactions) == 3]
        self.assertEqual(len(triple), 1)
        self.assertEqual(len(tree.get_in_edges(triple[0].node_index)), 3)

    def test_incremental_search_reuses_baseline_compounds_and_adds_only_selected_patterns(self) -> None:
        source = Compound.from_smiles('CC(O)N')
        build = builder(pattern('[C:1]-[O:2]', '[C:1]'),
                        pattern('[C:1]-[N:2]', '[C:1]'), limit=3)
        patterns = {item.reactant_smarts: item.pattern_id for item in build.reaction_pattern_set}
        baseline_id = patterns['[C:1]-[O:2]']
        added_id = patterns['[C:1]-[N:2]']
        baseline = build._build_result(source, addable_pattern_ids=frozenset((baseline_id,)))
        seed_compounds = {state.reaction_sequence: baseline['fragment_compound_by_index'][state.node_index]
                          for state in baseline['expansion_states'] if state.reaction_sequence
                          and len(state.reaction_sequence.reactions) < build.max_reaction_count}
        compiled = []
        original = ReactionSequence.compile

        def record(sequence, compound):
            compiled.append(sequence)
            return original(sequence, compound)

        with patch.object(ReactionSequence, 'compile', record):
            incremental = build._build_result(source, seed_reaction_sequences=tuple(seed_compounds),
                seed_compounds=seed_compounds, addable_pattern_ids=frozenset((added_id,)))
        self.assertTrue(compiled)
        self.assertTrue(all(any(reaction.reaction_pattern_id == added_id for reaction in sequence.reactions)
                            for sequence in compiled))
        sequences = {state.reaction_sequence for state in incremental['expansion_states'] if state.reaction_sequence}
        self.assertTrue(set(seed_compounds) <= sequences)
        self.assertTrue(any(len(sequence.reactions) == 1 and sequence.reactions[0].reaction_pattern_id == added_id
                            for sequence in sequences))
        seed_transitions = [transition for index in range(incremental['fragment_tree'].num_edges)
                            for transition in incremental['fragment_tree'].get_edge(index).transitions
                            if transition.reaction_sequence in seed_compounds]
        self.assertTrue(seed_transitions)
        self.assertTrue(all(not transition.is_seed and transition.added_reaction is not None
                            for transition in seed_transitions))
        self.assertTrue(all(transition.added_reaction.reaction_pattern_id == baseline_id
                            for transition in seed_transitions))

    def test_same_smiles_seeds_have_distinct_states(self) -> None:
        source = Compound.from_smiles('[CH3:1][CH:2]([OH:3])[OH:4]')
        build = builder(pattern('[C:1]-[O:2]', '[C:1]'), limit=2)
        reactions = build.create_reaction_instances(source)
        seeds = tuple(ReactionSequence((a,)) for a in reactions)
        result = build._build_result(source, seed_reaction_sequences=seeds)
        tree = result['fragment_tree']
        self.assertEqual(tree.num_nodes, 3)
        self.assertEqual(len(tree.get_out_edges(0)), 1)
        self.assertEqual(len(tree.get_out_edges(0)[0].transitions), 2)
        states = [s for s in result['expansion_states'] if s.reaction_sequence
                  and len(s.reaction_sequence.reactions) == 1]
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0].node_index, states[1].node_index)
        self.assertNotEqual(states[0].reaction_sequence, states[1].reaction_sequence)

    def test_different_histories_same_effect_use_cached_reaction(self) -> None:
        build = builder(pattern('[C:1]-[O:2]', '[C:1]'),
                        pattern('[C:1](-[O:2])-[N:3]', '[C:1]-[N:3]'), limit=1)
        result = build._build_result(Compound.from_smiles('CC(O)N'))
        self.assertEqual(result['search_stats']['num_rdkit_run_reactants'], 1)
        self.assertEqual(result['search_stats']['num_duplicate_effect_pruned'], 1)
        self.assertEqual(len(result['expansion_states']), 3)
        self.assertEqual(len(result['fragment_tree'].get_edge(0).transitions), 2)
        self.assertEqual(result['fragment_tree'].copy().get_edge(0).transitions,
                         result['fragment_tree'].get_edge(0).transitions)

    def test_react_all_returns_separate_reaction_results(self) -> None:
        results = self.builder.react_all(self.source)
        self.assertEqual(len(results), 7)
        self.assertEqual(len({r.reaction_sequence for r in results}), 7)
        self.assertTrue(all(r.smirks for r in results))

    def test_invalidated_and_conflicting_candidates_never_run_rdkit(self) -> None:
        source = Compound.from_smiles('[CH3:1][CH2:2][CH2:3][CH3:4]')
        a = reaction(0, (1, 2), retained=(1, 2, 3))
        b = reaction(1, (3, 4), matched=((3, 4),), cuts=((3, 4),))
        build = builder(limit=2)
        with patch.object(FragmentTreeBuilder, 'create_reaction_instances', return_value=(a, b)):
            result = build._build_result(source)
        stats = result['search_stats']
        self.assertEqual(stats['num_raw_combinations'], 3)
        self.assertEqual(stats['num_invalidated_reaction_pruned'], 1)
        self.assertEqual(stats['num_rdkit_run_reactants'], 2)
        a = reaction(0, (1, 2), matched=((1, 2),), cuts=((1, 2),))
        b = reaction(1, (1, 2), matched=((1, 2),), cuts=((1, 2),))
        with patch.object(FragmentTreeBuilder, 'create_reaction_instances', return_value=(a, b)):
            result = build._build_result(source)
        stats = result['search_stats']
        self.assertEqual(stats['num_hard_conflict_pruned'], 1)
        self.assertEqual(stats['num_duplicate_effect_pruned'], 1)
        self.assertEqual(stats['num_rdkit_run_reactants'], 1)

    def test_seed_multiple_reactions_count_towards_limit(self) -> None:
        reactions = self.builder.create_reaction_instances(self.source)
        seed = ReactionSequence(reactions[:2])
        result = self.builder._build_result(self.source, seed_reaction_sequences=(seed,), max_reaction_count=2)
        self.assertEqual(result['search_stats']['num_rdkit_run_reactants'], 1)
        self.assertEqual(len(result['expansion_states']), 2)
        edge = result['fragment_tree'].get_edge(0)
        self.assertTrue(edge.transitions[0].is_seed)
        self.assertEqual(edge.transitions[0].reaction_sequence, seed)
        extended = self.builder._build_result(self.source, seed_reaction_sequences=(seed,), max_reaction_count=3)
        self.assertEqual(extended['search_stats']['num_rdkit_run_reactants'], 2)

    def test_valid_combined_target_does_not_require_invalid_primitive_target(self) -> None:
        source = Compound.from_smiles('[CH3:1][C:2]([CH3:3])([CH3:4])[OH:5]')
        build = builder(pattern('[C:1]-[C:2]', '[C:1]=[C:2]'),
                        pattern('[C:1]-[O:2]', '[C:1]'), limit=2)
        result = build._build_result(source)
        tree = result['fragment_tree']
        self.assertEqual(set(tree.node_smiles), {source.smiles, 'CC(C)C', 'C=C(C)C'})
        child = tree.get_node_by_smiles('C=C(C)C')
        parent = tree.get_node_by_smiles('CC(C)C')
        self.assertTrue(any(edge.source_index == parent.index for edge in tree.get_in_edges(child.index)))
        self.assertEqual(result['search_stats']['num_rdkit_run_reactants'], 4)

    def test_unique_fragment_limit_counts_merged_fragments_once(self) -> None:
        source = Compound.from_smiles('[CH3:1][CH:2]([OH:3])[OH:4]')
        build = builder(pattern('[C:1]-[O:2]', '[C:1]'), limit=1)
        tree = build.build(source)
        self.assertEqual(tree.num_nodes, 2)
        self.assertEqual(tree.num_edges, 1)
        self.assertEqual(tree.num_transitions, 2)
        # Two generated fragments share one canonical SMILES; the source is not counted.
        result = build._build_result(source, max_unique_fragment_smiles=1)
        self.assertEqual(result['search_stats']['num_generated_fragments'], 2)
        self.assertEqual(result['search_stats']['num_unique_fragment_smiles'], 1)
        self.assertEqual(result['fragment_tree'].num_transitions, 2)
        with self.assertRaisesRegex(ValueError, r'^max_reaction_combinations exceeded: limit=1, observed>1$'):
            build.build(source, max_reaction_combinations=1)
        self.assertEqual(build.build(source, max_reaction_combinations=2).num_transitions, 2)

    def test_search_limits_stop_combination_search_and_reactions_early(self):
        baseline=self.builder._build_result(self.source)['search_stats']
        original=ReactionSequenceSearch.iter_candidates
        for limit in ({'max_unique_fragment_smiles':1},{'max_reaction_combinations':1}):
            searches=[]
            def stream(search):
                searches.append(search)
                yield from original(search)
            with self.subTest(limit=limit),patch.object(ReactionSequenceSearch,'iter_candidates',stream), \
                 patch.object(ReactionSequenceSearch,'enumerate',side_effect=AssertionError('Eager search is forbidden')):
                with self.assertRaisesRegex(ValueError,r'exceeded: limit=1, observed>1$'):
                    self.builder.build(self.source,**limit)
            stats=searches[0].stats
            self.assertLess(stats.num_raw_combinations,baseline['num_raw_combinations'])
            self.assertLess(stats.num_rdkit_run_reactants,baseline['num_rdkit_run_reactants'])

    def test_limits_seed_validation_and_removed_apis(self) -> None:
        for limit in (0, -1, True):
            with self.assertRaises(ValueError):
                FragmentTreeBuilder(limit, self.builder.reaction_pattern_set)
        for kwargs, message in (({'max_unique_fragment_smiles': 1}, 'max_unique_fragment_smiles exceeded'),
                                ({'max_reaction_combinations': 1}, 'max_reaction_combinations exceeded')):
            with self.assertRaisesRegex(ValueError, message):
                self.builder.build(self.source, **kwargs)
        for name in ('max_unique_fragment_smiles', 'max_reaction_combinations'):
            for invalid in (0, -2, True, 1.5):
                with self.subTest(name=name, invalid=invalid), \
                     self.assertRaisesRegex(ValueError, name + ' must be -1 or a positive integer'):
                    self.builder.build(self.source, **{name: invalid})
        reactions = self.builder.create_reaction_instances(self.source)
        with self.assertRaisesRegex(ValueError, 'Seed exceeds'):
            self.builder.build(self.source, seed_reaction_sequences=(ReactionSequence(reactions),),
                               max_reaction_count=2)
        with self.assertRaisesRegex(ValueError, 'universe'):
            self.builder.build(Compound.from_smiles('CC'),
                seed_reaction_sequences=(ReactionSequence((reactions[0],)),))
        self.assertEqual(self.builder.build(self.source, max_reaction_count=0).num_nodes, 1)
        data = self.builder.to_dict()
        self.assertEqual(FragmentTreeBuilder.from_dict(data).to_dict(), data)
        self.assertEqual(self.builder.copy().to_dict(), data)
        self.assertIsNot(self.builder.copy().reaction_pattern_set, self.builder.reaction_pattern_set)


    @staticmethod
    def _signature(tree):
        return (tree.node_smiles.tolist(), tree.source_indices.tolist(),
                tree.target_indices.tolist(), tree.num_transitions)

    def test_reaction_combination_limit_stops_before_rdkit_runs_any_later_combination(self) -> None:
        baseline = self.builder._build_result(self.source)
        raw = baseline['search_stats']['num_raw_combinations']
        run = CompositeReaction.run
        original = ReactionSequenceSearch.iter_candidates
        for limit in range(1, raw):
            yielded = []
            executed = []

            def stream(search):
                for candidate in original(search):
                    yielded.append(search.stats.num_raw_combinations)
                    yield candidate

            def running(reaction, source):
                executed.append(reaction)
                return run(reaction, source)

            with self.subTest(limit=limit), \
                 patch.object(ReactionSequenceSearch, 'iter_candidates', stream), \
                 patch.object(CompositeReaction, 'run', running):
                with self.assertRaises(FragmentTreeLimitExceeded) as caught:
                    self.builder.build(self.source, max_reaction_combinations=limit)
                error = caught.exception
                self.assertEqual(str(error), f'max_reaction_combinations exceeded: limit={limit}, observed>{limit}')
                self.assertEqual(error.limit_name, 'max_reaction_combinations')
                self.assertEqual(error.stats['num_raw_combinations'], limit + 1)
                # Only candidates found within the limit ever reached RDKit.
                self.assertTrue(all(count <= limit for count in yielded))
                self.assertEqual(len(executed), error.stats['num_rdkit_run_reactants'])
                self.assertLessEqual(len(executed), len(yielded))
        exact = self.builder._build_result(self.source, max_reaction_combinations=raw)
        self.assertEqual(self._signature(exact['fragment_tree']), self._signature(baseline['fragment_tree']))
        self.assertEqual(exact['search_stats'], baseline['search_stats'])

    def test_reaction_combination_limit_spans_every_size_up_to_max_reaction_count(self) -> None:
        pairs = builder(*(pattern(f'[C:1]-[{symbol}:2]', '[C:1]') for symbol in ('O', 'N', 'S')), limit=2)
        up_to_pairs = pairs._build_result(self.source)['search_stats']['num_raw_combinations']
        up_to_triples = self.builder._build_result(self.source)['search_stats']['num_raw_combinations']
        self.assertEqual(self.builder.max_reaction_count, 3)
        self.assertGreater(up_to_triples, up_to_pairs)
        # A budget sufficient for 1- and 2-reaction combinations is exceeded by the 3-reaction ones.
        pairs.build(self.source, max_reaction_combinations=up_to_pairs)
        with self.assertRaises(FragmentTreeLimitExceeded):
            self.builder.build(self.source, max_reaction_combinations=up_to_pairs)

    def test_unique_fragment_limit_stops_right_after_the_first_excess_smiles(self) -> None:
        search, candidates = self.builder._search(self.source, seed_reaction_sequences=None, max_reaction_count=None)
        runs_when_reached = {}
        for _ in self.builder._materialize_iter(self.source, search, candidates):
            runs_when_reached.setdefault(search.stats.num_unique_fragment_smiles, search.stats.num_rdkit_run_reactants)
        unique = search.stats.num_unique_fragment_smiles
        baseline = self.builder._build_result(self.source)
        # Every generated fragment is linked here, so nodes are the source plus each unique SMILES.
        self.assertEqual(unique, baseline['fragment_tree'].num_nodes - 1)
        self.assertGreater(unique, 2)
        for limit in range(1, unique):
            with self.subTest(limit=limit):
                with self.assertRaises(FragmentTreeLimitExceeded) as caught:
                    self.builder.build(self.source, max_unique_fragment_smiles=limit)
                error = caught.exception
                self.assertEqual(str(error), f'max_unique_fragment_smiles exceeded: limit={limit}, observed>{limit}')
                self.assertEqual(error.stats['num_unique_fragment_smiles'], limit + 1)
                # No further RDKit run happens after the excess SMILES appears.
                self.assertEqual(error.stats['num_rdkit_run_reactants'], runs_when_reached[limit + 1])
                self.assertLessEqual(error.stats['num_raw_combinations'], baseline['search_stats']['num_raw_combinations'])
        exact = self.builder._build_result(self.source, max_unique_fragment_smiles=unique)
        self.assertEqual(self._signature(exact['fragment_tree']), self._signature(baseline['fragment_tree']))
        self.assertEqual(exact['search_stats'], baseline['search_stats'])

    def test_limit_error_keeps_statistics_across_processes(self) -> None:
        import pickle
        with self.assertRaises(FragmentTreeLimitExceeded) as caught:
            self.builder.build(self.source, max_unique_fragment_smiles=1)
        restored = pickle.loads(pickle.dumps(caught.exception))
        self.assertEqual((restored.limit_name, restored.limit, restored.stats, str(restored)),
                         (caught.exception.limit_name, 1, caught.exception.stats, str(caught.exception)))


if __name__ == '__main__':
    unittest.main()
