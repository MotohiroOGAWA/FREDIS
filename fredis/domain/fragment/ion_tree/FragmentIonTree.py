from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
import numpy as np

from ....libs.mmkit.mmkit import Adduct, Compound

from ..tree.FragmentTree import FragmentTree
from .FragmentIonAdductRule import FragmentIonAdductRule
from .FragmentIonAdductRuleSet import FragmentIonAdductRuleSet
from ._private._FragmentIonFormulaCandidateGroup import (
    _FragmentIonFormulaCandidateGroup,
)


@dataclass(frozen=True, kw_only=True)
class FragmentIonTree(FragmentTree):
    """FragmentTree with the compounds and ion rules needed for formula candidates.

    fragment_compounds:
        One Compound per node, in node-index order.

    Every fragment-ion adduct of the matching adduct rule applies to every
    neutral fragment; charged fragments get none. Hydrogen shifts come from
    the rule set's shared range.

    search_stats / reaction_sequence_compounds:
        Build metadata from the reaction combination search.
    """

    fragment_ion_adduct_rule_set: FragmentIonAdductRuleSet
    fragment_compounds: Tuple[Compound, ...]
    search_stats: Dict[str, int] | None = None
    reaction_sequence_compounds: Dict[Any, Compound] = field(default_factory=dict)
    formula_candidate_groups: Dict[Adduct, _FragmentIonFormulaCandidateGroup] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.fragment_ion_adduct_rule_set, FragmentIonAdductRuleSet):
            raise TypeError(
                "fragment_ion_adduct_rule_set must be a FragmentIonAdductRuleSet."
            )

        object.__setattr__(self, "fragment_compounds", tuple(self.fragment_compounds))
        if len(self.fragment_compounds) != self.num_nodes:
            raise ValueError("fragment_compounds must have one Compound per node.")
        if not all(
            isinstance(compound, Compound) for compound in self.fragment_compounds
        ):
            raise TypeError("fragment_compounds must contain only Compound values.")

        if not isinstance(self.formula_candidate_groups, dict):
            raise TypeError("formula_candidate_groups must be a dict.")

    @classmethod
    def from_fragment_tree(
        cls,
        fragment_tree: FragmentTree,
        *,
        fragment_ion_adduct_rule_set: FragmentIonAdductRuleSet,
        fragment_compounds: Tuple[Compound, ...],
        search_stats: Dict[str, int] | None = None,
        reaction_sequence_compounds: Dict[Any, Compound] | None = None,
    ) -> "FragmentIonTree":
        """Create FragmentIonTree from FragmentTree and its node compounds."""

        if not isinstance(fragment_tree, FragmentTree):
            raise TypeError("fragment_tree must be a FragmentTree.")

        return cls(
            smiles=fragment_tree.smiles,
            node_store=fragment_tree.node_store,
            edge_store=fragment_tree.edge_store,
            adjacency=fragment_tree.adjacency,
            depths=fragment_tree.depths,
            fragment_ion_adduct_rule_set=fragment_ion_adduct_rule_set,
            fragment_compounds=tuple(fragment_compounds),
            search_stats=search_stats,
            reaction_sequence_compounds=dict(reaction_sequence_compounds or {}),
        )

    @property
    def hydrogen_shifts(self) -> Tuple[int, ...]:
        """Hydrogen shifts shared by every adduct rule."""
        return self.fragment_ion_adduct_rule_set.hydrogen_shifts

    @property
    def hydrogen_shift_adducts(self) -> Tuple[Adduct, ...]:
        """Hydrogen shifts as neutral adducts, in ``hydrogen_shifts`` order."""
        return tuple(Adduct.from_dict({"H": shift}) for shift in self.hydrogen_shifts)

    @property
    def node_is_neutral(self) -> np.ndarray:
        """True for nodes that can take fragment-ion adducts."""
        return np.asarray(
            [compound.charge == 0 for compound in self.fragment_compounds], dtype=bool
        )

    def get_adduct_rule_index_by_adduct_type(self, adduct_type: Adduct | str) -> int:
        """Return adduct rule index by main adduct type."""
        return self.fragment_ion_adduct_rule_set.encode_adduct_type(adduct_type)

    def get_adduct_rule(self, adduct_type: Adduct | str) -> FragmentIonAdductRule:
        """Return the adduct rule of one main adduct type."""
        return self.fragment_ion_adduct_rule_set.get_rule_by_adduct_type(adduct_type)

    def get_fragment_ion_adducts(
        self, node_index: int, adduct_type: Adduct | str
    ) -> Tuple[Adduct, ...]:
        """Fragment-ion adducts applicable to one node under one main adduct type."""
        if not 0 <= node_index < self.num_nodes:
            raise IndexError(
                f"node_index must be in [0, {self.num_nodes}). Got {node_index}."
            )
        if self.fragment_compounds[node_index].charge != 0:
            return ()
        return self.get_adduct_rule(adduct_type).fragment_ion_adducts

    def get_formula_candidate_group(
        self, adduct_type: Adduct | str
    ) -> _FragmentIonFormulaCandidateGroup:
        """Return formula candidate group for one main adduct type, built lazily."""
        rule = self.get_adduct_rule(adduct_type)
        group = self.formula_candidate_groups.get(rule.adduct_type)
        if group is None:
            group = _FragmentIonFormulaCandidateGroup.from_fragment_ion_tree(
                fragment_ion_tree=self,
                adduct_rule=rule,
            )
            self.formula_candidate_groups[rule.adduct_type] = group
        return group

    def get_node_formula_candidates(
        self,
        *,
        adduct_type: Adduct | str,
        formula: object,
    ) -> tuple[tuple[int, Adduct], ...]:
        """Return (node_index, candidate_adduct) candidates by formula."""
        return self.get_formula_candidate_group(adduct_type).get_candidates_by_formula(
            formula
        )

    def copy(self) -> FragmentIonTree:
        """Return a copy of this FragmentIonTree.

        formula_candidate_groups is a lazy cache, so it is intentionally
        not copied.
        """
        return self.from_fragment_tree(
            fragment_tree=super().copy(),
            fragment_ion_adduct_rule_set=self.fragment_ion_adduct_rule_set,
            fragment_compounds=self.fragment_compounds,
            search_stats=None if self.search_stats is None else dict(self.search_stats),
            reaction_sequence_compounds=self.reaction_sequence_compounds,
        )
