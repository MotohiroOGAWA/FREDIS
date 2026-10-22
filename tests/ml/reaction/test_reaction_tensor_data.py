"""Prepared reaction registry collation and new-format serialization."""
import torch
from fredis.ml.input.reaction_tensor_data import ReactionTensorData
from fredis.ml.input.source_reaction_structure import SourceReactionStructure
from .test_fragment_tree_model import prepared_two_ce


def test_registry_collation_and_round_trip(tmp_path):
    _, data = prepared_two_ce()
    registry = data.reactions
    atom_count = data.source_graph.num_nodes
    path = tmp_path / 'prepared.pt'
    batch = SourceReactionStructure.from_structures([data, data])
    count = len(registry.reaction_type)
    atom_count = data.source_graph.num_nodes
    assert batch.reactions.reaction_source_compound_index[count:].eq(1).all()
    atom_indices = registry.reaction_source_atom_index
    assert torch.equal(
        batch.reactions.reaction_source_atom_index,
        torch.cat([atom_indices, atom_indices + atom_count]),
    )
    ptr = registry.reaction_source_atom_ptr
    assert torch.equal(
        batch.reactions.reaction_source_atom_ptr,
        torch.cat([ptr, ptr[1:] + atom_indices.numel()]),
    )
    # The second molecule's global indices recover its original local indices.
    assert torch.equal(
        batch.reactions.reaction_source_atom_index[atom_indices.numel():]
        - batch.source_graph.ptr[1], atom_indices,
    )
    retained = registry.reaction_retained_atom_index
    retained_ptr = registry.reaction_retained_atom_ptr
    assert torch.equal(batch.reactions.reaction_retained_atom_index,
                       torch.cat([retained, retained + atom_count]))
    assert torch.equal(batch.reactions.reaction_retained_atom_ptr,
                       torch.cat([retained_ptr, retained_ptr[1:] + retained.numel()]))
    batch.save(path)
    restored = SourceReactionStructure.load(path)
    for name in ReactionTensorData.field_names():
        assert torch.equal(getattr(restored.reactions, name), getattr(batch.reactions, name))
