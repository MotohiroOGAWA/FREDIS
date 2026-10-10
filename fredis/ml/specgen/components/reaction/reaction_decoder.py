"""Tensor-only CE-independent branch search in cumulative log space."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .reaction_sequence_encoder import ReactionSequenceEncoder
from ...generation import GenerationSettings
from .reaction_compatibility import ReactionCompatibilityEngine, ReactionExpansion


@dataclass(frozen=True)
class ReactionPool:
    reaction_index: Tensor
    valid: Tensor
    embeddings: Tensor
    conflict: Tensor
    invalidation: Tensor
    dominance: Tensor
    retained: Tensor
    source_atom_valid: Tensor


def build_reaction_pool(reaction_h: Tensor, data: object) -> ReactionPool:
    # There is intentionally no learned absolute prefilter. Every primitive
    # reaction belonging to a branch group's source is retained; chemistry-derived
    # compatibility masks decide which continuations are valid.
    s=data.num_branch_groups;a=reaction_h.shape[0];device=reaction_h.device
    trees=data.branch_group_tree_index if data.branch_group_tree_index.numel() else data.sample_tree_index
    eligible=data.reaction_tree_index[None,:]==trees[:,None]
    width=int(eligible.sum(1).max().item()) if s else 0
    ids=torch.argsort(eligible.to(torch.int8),dim=1,descending=True,stable=True)[:,:width]
    valid=eligible.gather(1,ids)
    chosen_ids=ids.masked_fill(~valid,a)
    padded_h = torch.cat((reaction_h, reaction_h.new_zeros((1, reaction_h.shape[1]))))
    inverse = torch.full((s, a + 1), -1, dtype=torch.long, device=device)
    slots = torch.arange(width, device=device).expand(s, -1)
    inverse.scatter_(1, chosen_ids, slots)
    inverse[:, a] = -1
    sample = torch.arange(s, device=device)[:, None]

    def dense(pairs: Tensor) -> Tensor:
        matrix = torch.zeros((s, width, width), dtype=torch.bool, device=device)
        left, right = inverse[:, pairs[0]], inverse[:, pairs[1]]
        okay = (left >= 0) & (right >= 0)
        rows = sample.expand_as(left)[okay]
        matrix[rows, left[okay], right[okay]] = True
        return matrix

    # Packed Source-local retained regions avoid [expansions, atoms] memory.
    capacity=data.source_atom_capacity or data.source_graph.num_nodes
    words=(capacity+63)//64
    starts=data.source_graph.ptr[trees]
    sizes=data.source_graph.ptr[trees+1]-starts
    word_bits=(sizes[:,None]-64*torch.arange(words,device=device)[None,:]).clamp(0,64)
    atom_valid=(torch.ones_like(word_bits)<<word_bits.clamp_max(63))-1
    atom_valid=atom_valid.masked_fill(word_bits==64,-1)
    retained=torch.zeros((s,width,words),dtype=torch.long,device=device)
    pair=data.reaction_retained_index
    local=inverse[:,pair[0]]
    atom=pair[1][None,:]-starts[:,None]
    okay=local>=0
    rows=sample.expand_as(local)[okay]
    slots=local[okay]
    atom=atom[okay]
    bits=torch.ones_like(atom) << (atom%64)
    retained.index_put_((rows,slots,atom//64),bits,accumulate=True)
    return ReactionPool(chosen_ids.masked_fill(~valid, -1), valid, padded_h[chosen_ids],
                      dense(data.reaction_conflict_index),
                      dense(data.reaction_invalidation_index), dense(data.reaction_dominance_index),
                      retained, atom_valid)


def deduplicate(sample: Tensor, state: Tensor, score: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    keys = torch.cat((sample[:, None], state), dim=1)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    best = score.new_full((unique.shape[0],), -torch.inf)
    best.scatter_reduce_(0, inverse, score, reduce="amax", include_self=True)
    row = torch.arange(score.numel(), device=score.device)
    representative = torch.full((unique.shape[0],), score.numel(), dtype=torch.long, device=score.device)
    representative.scatter_reduce_(0, inverse, row.masked_fill(score != best[inverse], score.numel()), reduce="amin", include_self=True)
    return unique[:, 0], unique[:, 1:], best, representative


@dataclass(frozen=True)
class ReactionDecoderOutput:
    state_sample_index: Tensor
    state_reaction_index: Tensor
    state_log_score: Tensor
    terminal: Tensor
    parent_state_index: Tensor
    added_reaction_index: Tensor
    pool: ReactionPool
    duplicate_states_removed: Tensor | None = None


class BranchingReactionDecoder(nn.Module):
    def __init__(self, hidden_dim: int, main_adduct_dim: int, max_reaction_count: int,
                 num_heads: int = 4,
                 state_num_layers: int = 2, state_dropout: float = 0.0,
                 ) -> None:
        super().__init__()
        self.state_encoder = ReactionSequenceEncoder(hidden_dim, num_heads=num_heads,num_layers=state_num_layers,dropout=state_dropout)
        self.compatibility = ReactionCompatibilityEngine(max_reaction_count)
        self.query = nn.Linear(hidden_dim + main_adduct_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.apply_generation(GenerationSettings())
        self.scale = hidden_dim ** -0.5
        self.max_reaction_count = max_reaction_count
        self.max_depth=max_reaction_count

    def apply_generation(self, settings: GenerationSettings) -> None:
        """Use the search settings of ``settings`` for subsequent decoding."""
        self.branch_path_threshold=settings.branch_path_threshold
        self.branch_log_threshold=(float('-inf') if settings.branch_path_threshold==0
                                   else math.log(settings.branch_path_threshold))
        self.max_fragment_nodes=settings.max_fragment_nodes

    def score_states(self, pool: ReactionPool, main_adduct_h: Tensor, state: Tensor,
                     sample: Tensor) -> tuple[Tensor, ReactionExpansion]:
        h = self.state_encoder(pool.embeddings, state, sample)
        context = torch.cat((h, main_adduct_h[sample]), dim=1)
        reaction = torch.einsum("ph,pkh->pk", self.query(context), self.key(pool.embeddings[sample])) * self.scale
        expansion = self.compatibility.expand(state_reaction_index=state, state_sample_index=sample,
            pool_valid=pool.valid, pool_conflict=pool.conflict, pool_invalidation=pool.invalidation,
            pool_dominance=pool.dominance, pool_retained=pool.retained, source_atom_valid=pool.source_atom_valid)
        valid = expansion.valid.reshape(state.shape[0], pool.valid.shape[1]) & ((state >= 0).sum(dim=1) < self.max_reaction_count)[:, None]
        logits = reaction.masked_fill(~valid, -torch.inf)
        return logits, expansion

    def forward(self, pool: ReactionPool, main_adduct_h: Tensor) -> ReactionDecoderOutput:
        s, k = pool.valid.shape
        device = main_adduct_h.device
        # Every branch group starts at Source; the main adduct is supplied as
        # the branch condition rather than encoded as a CE-specific seed path.
        sample = torch.arange(s, device=device)
        state = torch.full((s, self.max_reaction_count), -1, dtype=torch.long, device=device)
        score = main_adduct_h.new_zeros(sample.shape[0])
        global_parent = torch.full((sample.shape[0],), -1, dtype=torch.long, device=device)
        added = torch.full_like(global_parent, -1)
        all_sample, all_state, all_score, all_parent, all_added, all_terminal = [], [], [], [], [], []
        offset = 0
        raw_history_score = []
        duplicate_states_removed = main_adduct_h.new_zeros(())
        # Tensor batches per generation iteration, never loops over states/reactions.
        for step in range(self.max_depth + 1):
            if sample.numel() == 0:
                break
            row = torch.arange(sample.numel(), device=device) + offset
            raw_history_score.append(score)
            logits, expansion = self.score_states(pool, main_adduct_h, state, sample)
            probabilities = F.logsigmoid(logits)
            all_sample.append(sample)
            all_state.append(state)
            all_score.append(score)
            all_parent.append(global_parent)
            all_added.append(added)
            all_terminal.append(torch.ones_like(sample, dtype=torch.bool))
            offset += sample.numel()
            if step == self.max_depth:
                break
            parent = expansion.parent_state_index
            candidate = expansion.candidate_reaction_index
            candidate_score = score[parent] + probabilities[parent, candidate]
            valid = expansion.valid & torch.isfinite(candidate_score) & (candidate_score >= self.branch_log_threshold)
            parent, candidate, candidate_score = parent[valid], candidate[valid], candidate_score[valid]
            candidate_count=candidate_score.numel()
            child_sample, child, child_score, representative = deduplicate(sample[parent], expansion.child_reaction_index[valid], candidate_score)
            duplicate_states_removed += candidate_count-child_score.numel()
            old_keys = torch.cat((torch.cat(all_sample)[:, None], torch.cat(all_state)), dim=1)
            new_keys = torch.cat((child_sample[:, None], child), dim=1)
            unique, inverse = torch.unique(torch.cat((old_keys, new_keys)), dim=0, return_inverse=True)
            old_best = child_score.new_full((unique.shape[0],), -torch.inf)
            old_best.scatter_reduce_(0, inverse[:old_keys.shape[0]], torch.cat(raw_history_score), reduce="amax", include_self=True)
            improved = child_score > old_best[inverse[old_keys.shape[0]:]]
            duplicate_states_removed += improved.numel()-int(improved.sum())
            child_sample, child, child_score, representative = child_sample[improved], child[improved], child_score[improved], representative[improved]
            parents = row[parent[representative]]
            reactions = candidate[representative]
            # max_fragment_nodes is shared by the whole branch group, not by CE
            # or generation depth. Existing accepted states consume the budget.
            used=torch.bincount(torch.cat(all_sample),minlength=s)
            remaining=(self.max_fragment_nodes-used).clamp_min(0)
            order=torch.argsort(child_score,descending=True,stable=True)
            order=order[torch.argsort(child_sample[order],stable=True)]
            counts=torch.bincount(child_sample,minlength=s);starts=counts.cumsum(0)-counts
            rank=torch.arange(order.numel(),device=device)-starts[child_sample[order]]
            selected=order[rank<remaining[child_sample[order]]]
            sample, state, score = child_sample[selected], child[selected], child_score[selected]
            global_parent, added = parents[selected], reactions[selected]
        samples, states, scores = torch.cat(all_sample), torch.cat(all_state), torch.cat(all_score)
        # Leaves stop at the threshold or a generation limit. Materialization
        # recovers all accepted ancestors from these leaves.
        terminal = torch.ones_like(samples,dtype=torch.bool)
        parent_rows=torch.cat(all_parent)
        terminal[parent_rows[parent_rows>=0]]=False
        return ReactionDecoderOutput(samples, states, scores, terminal, torch.cat(all_parent), torch.cat(all_added), pool,
                                   duplicate_states_removed)
