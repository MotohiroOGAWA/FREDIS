"""Prepared reaction registry tensors, shared by preparation and neural models."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import torch
from torch import Tensor

def _csr(rows):
    counts = torch.tensor([len(row) for row in rows], dtype=torch.long)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)]), torch.tensor(
        [atom for row in rows for atom in row], dtype=torch.long
    )

def _coo(pairs):
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()



@dataclass(frozen=True)
class ReactionTensorData:
    """R concrete reactions on source molecules; all indices are batch-local.

    A reaction ID is a row of reaction_type, not a reaction pattern ID.
    Source atom indices address source_graph.x, not atom-map labels.
    CSR lists for reaction r occupy index[ptr[r]:ptr[r + 1]].
    """

    # [R]: source molecule index for each concrete reaction.
    reaction_tree_index: Tensor
    # [R, 3]: [reaction_pattern_id, reaction_rule_id, product_molecule_id].
    reaction_type: Tensor
    # [R + 1]: CSR boundaries for the matched source atoms of each reaction.
    reaction_source_atom_ptr: Tensor
    # [A]: concatenated matched source atom indices into source_graph.x.
    reaction_source_atom_index: Tensor
    # [2, C]: each column is [reaction_i, reaction_j]. The two reactions
    # change at least one identical Source bond (cut or bond-order update).
    # They cannot coexist in a reaction set. Example: both cut bond (3, 4).
    # This relation is symmetric; both directed pairs are stored.
    reaction_conflict_index: Tensor
    # [2, I]: each column is [invalidating_reaction, invalidated_reaction].
    # The first discards at least one Source atom required by the second's
    # concrete reactant match (source_atom_maps), even if they change no common
    # bond. Example: A discards atom 4; B needs atoms {4, 5} to match.
    # Equivalently: B.source_atom_maps is not a subset of A.retained_atom_maps.
    # Directional relation: [A, B] need not imply [B, A]. A reaction set is
    # rejected if either direction is present; reaction order cannot rescue it.
    reaction_invalidation_index: Tensor
    # [R + 1]: CSR boundaries for the atoms retained by each reaction.
    reaction_retained_atom_ptr: Tensor
    # [K]: concatenated retained atom indices into the batched source_graph.x.
    # Reaction r retains index[ptr[r]:ptr[r + 1]]; empty rows are supported.
    reaction_retained_atom_index: Tensor

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    def to(self, device: str | torch.device) -> ReactionTensorData:
        return replace(self, **{
            name: getattr(self, name).to(device) for name in self.field_names()
        })

    @classmethod
    def from_reactions(cls, reactions, maps: dict[int, int], relations) -> ReactionTensorData:
        """Pack concrete reactions using atom-map -> source_graph.x indices."""
        atom_ptr, atom_index = _csr(
            [
                tuple(maps[v] for v in reaction.source_atom_maps if v in maps)
                for reaction in reactions
            ]
        )
        retained_ptr, retained_index = _csr(
            [
                [maps[v] for v in sorted(reaction.retained_atom_maps) if v in maps]
                for reaction in reactions
            ]
        )
        return cls(
            torch.zeros(len(reactions), dtype=torch.long),
            torch.tensor(
                [
                    (a.reaction_pattern_id, a.reaction_rule_id, a.product_molecule_id)
                    for a in reactions
                ],
                dtype=torch.long,
            ).reshape(-1, 3),
            atom_ptr,
            atom_index,
            _coo(relations.conflict_pairs),
            _coo(relations.invalidation_pairs),
            retained_ptr,
            retained_index,
        )

    @classmethod
    def collate(cls, registries, atom_counts, tree_counts) -> ReactionTensorData:
        """Concatenate registries; shift global atom, molecule and reaction indices."""
        if not registries:
            raise ValueError("Cannot collate an empty reaction registry batch")
        values = {name: [] for name in cls.field_names()}
        atom_offset = tree_offset = reaction_offset = 0
        ptr_offsets = {name: 0 for name in cls.field_names() if name.endswith("_ptr")}
        for item, atom_count, tree_count in zip(registries, atom_counts, tree_counts, strict=True):
            for name in cls.field_names():
                value = getattr(item, name)
                if name.endswith('_ptr'):
                    previous = ptr_offsets[name]
                    ptr_offsets[name] = previous + value[-1]
                    value = value + previous
                    if values[name]:
                        value = value[1:]
                elif name in ('reaction_source_atom_index', 'reaction_retained_atom_index'):
                    value = value + atom_offset
                elif name == 'reaction_tree_index':
                    value = value + tree_offset
                elif name in ('reaction_conflict_index', 'reaction_invalidation_index'):
                    value = value + reaction_offset
                values[name].append(value)
            atom_offset += atom_count
            tree_offset += tree_count
            reaction_offset += item.reaction_type.shape[0]
        return cls(**{name: torch.cat(parts, dim=1 if name in (
            'reaction_conflict_index', 'reaction_invalidation_index') else 0)
            for name, parts in values.items()})

    @staticmethod
    def relation_mask(pairs: Tensor, inverse: Tensor, width: int) -> Tensor:
        """Map global reaction pairs to [groups, pool_width, pool_width] masks."""
        s = inverse.shape[0]
        device = inverse.device
        sample = torch.arange(s, device=device)[:, None]
        matrix = torch.zeros((s, width, width), dtype=torch.bool, device=device)
        left, right = inverse[:, pairs[0]], inverse[:, pairs[1]]
        okay = (left >= 0) & (right >= 0)
        rows = sample.expand_as(left)[okay]
        matrix[rows, left[okay], right[okay]] = True
        return matrix


    def retained_bitsets(self, inverse: Tensor, source_ptr: Tensor,
                         trees: Tensor, capacity: int, width: int) -> tuple[Tensor, Tensor]:
        """Build pool-local retained atom bitsets and valid Source atom bitsets.

        CSR atom indices are global; each group's Source offset is subtracted
        before packing local atom positions into 64-bit words.
        """
        s = inverse.shape[0]
        device = inverse.device
        sample = torch.arange(s, device=device)[:, None]
        # Packed Source-local retained regions avoid [expansions, atoms] memory.
        words = (capacity + 63) // 64
        starts = source_ptr[trees]
        sizes = source_ptr[trees + 1] - starts
        word_bits = (
            sizes[:, None] - 64 * torch.arange(words, device=device)[None, :]
        ).clamp(0, 64)
        atom_valid = (torch.ones_like(word_bits) << word_bits.clamp_max(63)) - 1
        atom_valid = atom_valid.masked_fill(word_bits == 64, -1)
        retained = torch.zeros((s, width, words), dtype=torch.long, device=device)
        registry = self
        ptr = registry.reaction_retained_atom_ptr
        # Expand CSR row ownership only while constructing the runtime bitsets.
        reaction_ids = torch.repeat_interleave(
            torch.arange(ptr.numel() - 1, device=device), ptr[1:] - ptr[:-1]
        )
        local = inverse[:, reaction_ids]
        atom = registry.reaction_retained_atom_index[None, :] - starts[:, None]
        okay = local >= 0
        rows = sample.expand_as(local)[okay]
        slots = local[okay]
        atom = atom[okay]
        bits = torch.ones_like(atom) << (atom % 64)
        retained.index_put_((rows, slots, atom // 64), bits, accumulate=True)
        return retained, atom_valid
