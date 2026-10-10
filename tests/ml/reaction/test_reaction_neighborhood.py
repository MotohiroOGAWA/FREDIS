"""Hop-wise neighborhood context: graph algorithm, ReactionInstanceEncoder, and end-to-end checks."""
import unittest

import torch
from torch import nn

from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.ml.input.source_reaction_structure import (
    SourceReactionStructure, exact_hop_shells, hop_neighborhood_csr, csr,
)
from fredis.ml.input.reaction_batching import select_samples
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.specgen.components.reaction.reaction_instance_encoder import ReactionInstanceEncoder, csr_mean_pool
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from .TestReactionPipeline import config


def _chain_adjacency(n):
    adjacency = [set() for _ in range(n)]
    for i in range(n - 1):
        adjacency[i].add(i + 1)
        adjacency[i + 1].add(i)
    return adjacency


def _row(ptr, index, i):
    a, b = int(ptr[i]), int(ptr[i + 1])
    return index[a:b].tolist()


class TestExactHopShells(unittest.TestCase):
    def test_chain_molecule_exact_hops(self):
        # 0 - 1 - 2 - 3 - 4
        adjacency = _chain_adjacency(5)
        self.assertEqual(exact_hop_shells(adjacency, 1), [{0, 2}, {3}, {4}])
        self.assertEqual(exact_hop_shells(adjacency, 2), [{1, 3}, {0, 4}, set()])

    def test_ring_bfs_has_no_duplicate_or_wrong_distance_atoms(self):
        adjacency = [set() for _ in range(4)]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            adjacency[a].add(b)
            adjacency[b].add(a)
        self.assertEqual(exact_hop_shells(adjacency, 0), [{1, 3}, {2}, set()])


class TestHopNeighborhoodCsr(unittest.TestCase):
    def test_reactant_atoms_are_excluded_from_their_own_neighborhood(self):
        # Worked example from the spec: chain 0-1-2-3-4, reactant R = {1, 2}.
        adjacency = _chain_adjacency(5)
        atom_ptr, atom_index = csr([[1, 2]])
        (hop1_ptr, hop1_index), (hop2_ptr, hop2_index), (hop3_ptr, hop3_index) = hop_neighborhood_csr(adjacency, atom_ptr, atom_index)
        # Row 0 = atom 1: graph 1-hop neighbor 2 is internal to R and must be excluded.
        self.assertEqual(_row(hop1_ptr, hop1_index, 0), [0])
        self.assertEqual(_row(hop2_ptr, hop2_index, 0), [3])
        self.assertEqual(_row(hop3_ptr, hop3_index, 0), [4])
        # Row 1 = atom 2: graph 1-hop neighbor 1 is internal to R and must be excluded.
        self.assertEqual(_row(hop1_ptr, hop1_index, 1), [3])
        self.assertEqual(_row(hop2_ptr, hop2_index, 1), [0, 4])
        self.assertEqual(_row(hop3_ptr, hop3_index, 1), [])

    def test_distance_search_may_pass_through_reactant_atoms(self):
        # [C:1]-[O:2]-N : R = {C, O}; N is graph distance 2 from C, reached by
        # passing the BFS through O even though O itself (being internal to R)
        # never appears in C's own external neighborhood.
        adjacency = _chain_adjacency(3)  # 0=C, 1=O, 2=N
        atom_ptr, atom_index = csr([[0, 1]])
        (hop1_ptr, hop1_index), (hop2_ptr, hop2_index), _ = hop_neighborhood_csr(adjacency, atom_ptr, atom_index)
        self.assertEqual(_row(hop1_ptr, hop1_index, 0), [])  # O excluded, not merely distant
        self.assertEqual(_row(hop2_ptr, hop2_index, 0), [2])  # N reached through O

    def test_reactant_covering_whole_molecule_gives_empty_hops(self):
        adjacency = _chain_adjacency(2)
        atom_ptr, atom_index = csr([[0, 1]])
        hops = hop_neighborhood_csr(adjacency, atom_ptr, atom_index)
        for ptr, index in hops:
            self.assertEqual(index.numel(), 0)
            self.assertEqual(ptr.tolist(), [0, 0, 0])

    def test_row_count_matches_flattened_atom_index_not_reaction_count(self):
        adjacency = _chain_adjacency(5)
        atom_ptr, atom_index = csr([[0], [1, 2], [4]])  # 3 reactions, 4 tokens total
        for ptr, _ in hop_neighborhood_csr(adjacency, atom_ptr, atom_index):
            self.assertEqual(ptr.numel(), atom_index.numel() + 1)


class TestCsrMeanPool(unittest.TestCase):
    def test_empty_rows_are_exactly_zero_and_no_nan(self):
        source_atom_h = torch.randn(5, 4)
        ptr = torch.tensor([0, 0, 2])
        index = torch.tensor([1, 3])
        pooled = csr_mean_pool(source_atom_h, ptr, index)
        self.assertEqual(tuple(pooled.shape), (2, 4))
        self.assertTrue(torch.equal(pooled[0], torch.zeros(4)))
        self.assertTrue(torch.equal(pooled[1], (source_atom_h[1] + source_atom_h[3]) / 2))
        self.assertFalse(torch.isnan(pooled).any())

    def test_all_rows_empty_produces_zero_tensor_without_nan(self):
        source_atom_h = torch.randn(3, 4)
        ptr = torch.zeros(4, dtype=torch.long)
        index = torch.empty(0, dtype=torch.long)
        pooled = csr_mean_pool(source_atom_h, ptr, index)
        self.assertTrue(torch.equal(pooled, torch.zeros(3, 4)))
        self.assertFalse(torch.isnan(pooled).any())


class TestReactionInstanceEncoderNeighborhood(unittest.TestCase):
    def _encoder(self, mode='hop_pooling', seed=0, max_hop=3):
        torch.manual_seed(seed)
        return ReactionInstanceEncoder(atom_dim=4, mol_dim=6, hidden_dim=8, category_sizes=(2, 2, 2), num_heads=2, max_roles=4,
                              reaction_neighborhood_mode=mode, reaction_neighborhood_max_hop=max_hop)

    def _base_kwargs(self, source_atom_h):
        return dict(
            source_atom_h=source_atom_h, source_mol_h=torch.zeros(1, 6),
            reaction_type=torch.zeros(1, 3, dtype=torch.long), reaction_tree_index=torch.zeros(1, dtype=torch.long),
            reaction_source_atom_ptr=torch.tensor([0, 2]), reaction_source_atom_index=torch.tensor([0, 1]),
            reaction_static_features=torch.zeros(1, 6),
        )

    def _empty_hops(self, max_hop=3):
        return {'reaction_source_atom_hops': [(torch.tensor([0, 0, 0]), torch.empty(0, dtype=torch.long)) for _ in range(max_hop)]}

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            ReactionInstanceEncoder(atom_dim=4, mol_dim=4, hidden_dim=4, category_sizes=(1, 1, 1), reaction_neighborhood_mode='gnn')

    def test_invalid_max_hop_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reaction_neighborhood_max_hop'):
            ReactionInstanceEncoder(atom_dim=4, mol_dim=4, hidden_dim=4, category_sizes=(1, 1, 1), reaction_neighborhood_max_hop=0)

    def test_projection_count_matches_max_hop(self):
        for max_hop in (1, 2, 3, 5):
            encoder = self._encoder(max_hop=max_hop)
            self.assertEqual(len(encoder.neighborhood_projections), max_hop)

    def test_mismatched_hop_list_length_rejected(self):
        encoder = self._encoder(max_hop=2)
        kwargs = {**self._base_kwargs(torch.randn(5, 4)), **self._empty_hops(max_hop=3)}
        with self.assertRaisesRegex(ValueError, 'Expected 2 hop distances, got 3'):
            encoder(**kwargs)

    def test_output_shape_unaffected_by_reactant_atom_count(self):
        encoder = self._encoder()
        kwargs = {**self._base_kwargs(torch.randn(5, 4)), **self._empty_hops()}
        out = encoder(**kwargs)
        self.assertEqual(tuple(out.shape), (1, 8))

    def test_output_shape_with_a_single_hop_distance(self):
        encoder = self._encoder(max_hop=1)
        kwargs = {**self._base_kwargs(torch.randn(5, 4)), **self._empty_hops(max_hop=1)}
        out = encoder(**kwargs)
        self.assertEqual(tuple(out.shape), (1, 8))

    def test_zero_initialized_projections_make_hop_pooling_match_none_mode(self):
        encoder_hop = self._encoder('hop_pooling', seed=1)
        encoder_none = self._encoder('none', seed=1)
        kwargs = {**self._base_kwargs(torch.randn(6, 4)), **self._empty_hops()}
        # Nonempty neighbors prove 'none' truly ignores them, not that they're empty.
        kwargs['reaction_source_atom_hops'][0] = (torch.tensor([0, 1, 2]), torch.tensor([2, 3]))
        with torch.no_grad():
            out_hop = encoder_hop(**kwargs)
            out_none = encoder_none(**kwargs)
        torch.testing.assert_close(out_hop, out_none)

    def test_different_external_neighborhoods_change_the_output_once_trained(self):
        encoder = self._encoder()
        with torch.no_grad():
            nn.init.normal_(encoder.neighborhood_projections[0].weight)
        kwargs = {**self._base_kwargs(torch.randn(6, 4)), **self._empty_hops()}
        kwargs['reaction_source_atom_hops'][0] = (torch.tensor([0, 1, 1]), torch.tensor([2]))
        with torch.no_grad():
            out_a = encoder(**kwargs)
        kwargs['reaction_source_atom_hops'][0] = (torch.tensor([0, 1, 1]), torch.tensor([3]))
        with torch.no_grad():
            out_b = encoder(**kwargs)
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_gradient_flows_to_neighborhood_projections_and_source_atom_h(self):
        # A plain .sum() would pass straight through the final LayerNorm's
        # zero-mean invariant (every row's normalized output sums to ~0
        # regardless of its input), masking real gradients; weighting by a
        # fixed random vector breaks that symmetry.
        encoder = self._encoder()
        source_atom_h = torch.randn(6, 4, requires_grad=True)
        kwargs = {**self._base_kwargs(source_atom_h), **self._empty_hops()}
        kwargs['reaction_source_atom_hops'][0] = (torch.tensor([0, 1, 2]), torch.tensor([2, 3]))
        weights = torch.randn(1, 8)
        (encoder(**kwargs) * weights).sum().backward()
        self.assertIsNotNone(encoder.neighborhood_projections[0].weight.grad)
        self.assertTrue(torch.any(encoder.neighborhood_projections[0].weight.grad != 0))
        self.assertIsNotNone(source_atom_h.grad)

    def test_none_mode_neighborhood_projections_receive_no_gradient(self):
        encoder = self._encoder('none')
        kwargs = {**self._base_kwargs(torch.randn(6, 4)), **self._empty_hops()}
        kwargs['reaction_source_atom_hops'][0] = (torch.tensor([0, 1, 2]), torch.tensor([2, 3]))
        weights = torch.randn(1, 8)
        (encoder(**kwargs) * weights).sum().backward()
        self.assertIsNone(encoder.neighborhood_projections[0].weight.grad)


class TestReactionNeighborhoodModeConfig(unittest.TestCase):
    def test_reaction_neighborhood_mode_threads_through_model_config(self):
        model = config()
        model['reaction_model_params']['reaction_neighborhood_mode'] = 'none'
        generator = create_spectrum_generator(model)
        self.assertEqual(generator.feature_model.reaction_instance_encoder.reaction_neighborhood_mode, 'none')
        generator_default = create_spectrum_generator(config())
        self.assertEqual(generator_default.feature_model.reaction_instance_encoder.reaction_neighborhood_mode, 'hop_pooling')
        self.assertEqual(generator_default.feature_model.reaction_instance_encoder.reaction_neighborhood_max_hop, 3)

    def test_reaction_neighborhood_max_hop_threads_through_model_config(self):
        model = config()
        model['reaction_model_params']['reaction_neighborhood_max_hop'] = 2
        generator = create_spectrum_generator(model)
        self.assertEqual(generator.feature_model.reaction_instance_encoder.reaction_neighborhood_max_hop, 2)
        self.assertEqual(len(generator.feature_model.reaction_instance_encoder.neighborhood_projections), 2)

    def test_reaction_neighborhood_max_hop_out_of_range_rejected(self):
        from fredis.ml.input.source_reaction_structure import MAX_NEIGHBORHOOD_HOP
        model = config()
        model['reaction_model_params']['reaction_neighborhood_max_hop'] = MAX_NEIGHBORHOOD_HOP + 1
        with self.assertRaisesRegex(ValueError, 'reaction_neighborhood_max_hop'):
            create_spectrum_generator(model)
        model['reaction_model_params']['reaction_neighborhood_max_hop'] = 0
        with self.assertRaises(ValueError):
            create_spectrum_generator(model)


class TestNeighborhoodIntegration(unittest.TestCase):
    def _prepared(self):
        generator = create_spectrum_generator(config())
        source = Compound.from_smiles('CC(O)N')
        adduct = Adduct.parse('[M+H]+')
        data, _ = ReactionStructureBuilder(generator).build(source, [adduct, adduct], [10., 40.],
            [[18.033826, 44.049476], [18.033826, 44.049476]], [[.5, 1.], [1., .5]])
        return generator, data

    def test_prepared_hop_csr_rows_align_with_flattened_atom_index_and_stay_within_the_source_graph(self):
        _, data = self._prepared()
        n = data.reaction_source_atom_index.numel()
        num_atoms = data.source_graph.num_nodes
        for ptr, index in ((data.reaction_source_atom_hop1_ptr, data.reaction_source_atom_hop1_index),
                           (data.reaction_source_atom_hop2_ptr, data.reaction_source_atom_hop2_index),
                           (data.reaction_source_atom_hop3_ptr, data.reaction_source_atom_hop3_index)):
            self.assertEqual(ptr.numel(), n + 1)
            if index.numel():
                self.assertTrue(bool((index >= 0).all()))
                self.assertTrue(bool((index < num_atoms).all()))

    def test_batching_offsets_hop_neighbor_atom_indices(self):
        _, data = self._prepared()
        atoms_per_item = data.source_graph.num_nodes
        batched = SourceReactionStructure.from_structures([data, data])
        n_rows = data.reaction_source_atom_hop1_ptr.numel() - 1
        for i in range(n_rows):
            original = _row(data.reaction_source_atom_hop1_ptr, data.reaction_source_atom_hop1_index, i)
            shifted = _row(batched.reaction_source_atom_hop1_ptr, batched.reaction_source_atom_hop1_index, n_rows + i)
            self.assertEqual(shifted, [v + atoms_per_item for v in original])
        self.assertEqual(batched.reaction_source_atom_hop1_ptr.numel(), 2 * n_rows + 1)

    def test_select_samples_preserves_reaction_atom_and_hop_alignment(self):
        _, data = self._prepared()
        selected = select_samples(data, [0])
        self.assertTrue(torch.equal(selected.reaction_source_atom_index, data.reaction_source_atom_index))
        for name in ('reaction_source_atom_hop1_ptr', 'reaction_source_atom_hop1_index',
                     'reaction_source_atom_hop2_ptr', 'reaction_source_atom_hop2_index',
                     'reaction_source_atom_hop3_ptr', 'reaction_source_atom_hop3_index'):
            self.assertTrue(torch.equal(getattr(selected, name), getattr(data, name)))

    def test_forward_and_encode_static_produce_the_same_reaction_h(self):
        generator, data = self._prepared()
        generator.eval()
        with torch.no_grad():
            forward_reaction = generator.feature_model(data).reaction_h
            static_reaction, _ = generator.feature_model.encode_static(data)
        torch.testing.assert_close(forward_reaction, static_reaction)

    def test_reduced_max_hop_only_consumes_that_many_prepared_hop_fields(self):
        # Prepared data always carries all 3 hop fields; a model configured
        # with a smaller max_hop must use only the first N of them.
        model = config()
        model['reaction_model_params']['reaction_neighborhood_max_hop'] = 1
        generator = create_spectrum_generator(model)
        source = Compound.from_smiles('CC(O)N')
        adduct = Adduct.parse('[M+H]+')
        data, _ = ReactionStructureBuilder(generator).build(source, [adduct, adduct], [10., 40.],
            [[18.033826, 44.049476], [18.033826, 44.049476]], [[.5, 1.], [1., .5]])
        generator.eval()
        with torch.no_grad():
            forward_reaction = generator.feature_model(data).reaction_h
            static_reaction, _ = generator.feature_model.encode_static(data)
        torch.testing.assert_close(forward_reaction, static_reaction)


if __name__ == '__main__':
    unittest.main()
