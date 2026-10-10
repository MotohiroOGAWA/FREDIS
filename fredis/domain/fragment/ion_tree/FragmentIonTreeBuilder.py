from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence

from ..reaction.ReactionSequence import ReactionSequence

from ....libs.mmkit.mmkit import Compound
from ..reaction.ReactionPatternSet import ReactionPatternSet
from ..tree.FragmentTree import FragmentTree
from ..tree.FragmentTreeBuilder import FragmentTreeBuilder
from .FragmentIonAdductRuleSet import FragmentIonAdductRuleSet
from .FragmentIonTree import FragmentIonTree


@dataclass(frozen=True, kw_only=True)
class FragmentIonTreeBuilder(FragmentTreeBuilder):
    """Build FragmentIonTree directly from Compound.

    This builder first builds a FragmentTree, then attaches its node
    compounds and the FragmentIonAdductRuleSet used for formula candidates.
    """

    fragment_ion_adduct_rule_set: FragmentIonAdductRuleSet

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(
            self.fragment_ion_adduct_rule_set,
            FragmentIonAdductRuleSet,
        ):
            raise TypeError(
                "fragment_ion_adduct_rule_set must be a "
                "FragmentIonAdductRuleSet."
            )

    def build(
        self,
        compound: Compound,
        *,
        max_unique_fragment_smiles: int = -1,
        max_reaction_combinations: int = -1,
        max_reaction_count: int | None = None,
        seed_reaction_sequences: Sequence[ReactionSequence] | None = None,
        seed_compounds: dict[ReactionSequence, Compound] | None = None,
        addable_pattern_ids: frozenset[int] | None = None,
        print_info: bool = False,
        precomputed_reactions: Sequence | None = None,
    ) -> FragmentIonTree:
        """Build FragmentIonTree from Compound."""

        result = self._build_result(
            compound,
            max_unique_fragment_smiles=max_unique_fragment_smiles,
            max_reaction_combinations=max_reaction_combinations,
            max_reaction_count=max_reaction_count,
            seed_reaction_sequences=seed_reaction_sequences,
            seed_compounds=seed_compounds,
            addable_pattern_ids=addable_pattern_ids,
            print_info=print_info,
            precomputed_reactions=precomputed_reactions,
        )

        fragment_tree: FragmentTree = result["fragment_tree"]
        fragment_compound_by_index: dict[int, Compound] = result[
            "fragment_compound_by_index"
        ]

        return FragmentIonTree.from_fragment_tree(
            fragment_tree,
            fragment_ion_adduct_rule_set=self.fragment_ion_adduct_rule_set,
            fragment_compounds=tuple(
                fragment_compound_by_index[node_index]
                for node_index in range(fragment_tree.num_nodes)
            ),
            search_stats=result["search_stats"],
            reaction_sequence_compounds={
                state.reaction_sequence: fragment_compound_by_index[state.node_index]
                for state in result["expansion_states"]
                if state.reaction_sequence is not None
            },
        )

    def build_fragment_tree(
        self,
        compound: Compound,
        *,
        max_unique_fragment_smiles: int = -1,
        max_reaction_combinations: int = -1,
        max_reaction_count: int | None = None,
        seed_reaction_sequences: Sequence[ReactionSequence] | None = None,
        seed_compounds: dict[ReactionSequence, Compound] | None = None,
        addable_pattern_ids: frozenset[int] | None = None,
        print_info: bool = False,
    ) -> FragmentTree:
        """Build only FragmentTree."""

        return super().build(
            compound,
            max_unique_fragment_smiles=max_unique_fragment_smiles,
            max_reaction_combinations=max_reaction_combinations,
            max_reaction_count=max_reaction_count,
            seed_reaction_sequences=seed_reaction_sequences,
            seed_compounds=seed_compounds,
            addable_pattern_ids=addable_pattern_ids,
            print_info=print_info,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "fragment_ion_adduct_rule_set": (
                self.fragment_ion_adduct_rule_set.to_dict()
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> FragmentIonTreeBuilder:
        return cls(
            max_reaction_count=data["max_reaction_count"],
            reaction_pattern_set=ReactionPatternSet.from_dict(
                data["reaction_pattern_set"]
            ),
            fragment_ion_adduct_rule_set=FragmentIonAdductRuleSet.from_dict(
                data["fragment_ion_adduct_rule_set"]
            ),
        )

    def copy(self) -> FragmentIonTreeBuilder:
        return FragmentIonTreeBuilder(
            max_reaction_count=self.max_reaction_count,
            reaction_pattern_set=self.reaction_pattern_set.copy(),
            fragment_ion_adduct_rule_set=(
                self.fragment_ion_adduct_rule_set.copy()
            ),
        )
