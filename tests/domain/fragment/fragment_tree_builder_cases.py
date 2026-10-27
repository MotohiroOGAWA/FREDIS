from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fredis.domain.fragment.reaction._ReactionPattern import (
    ProductRule,
    _ReactionPattern,
)
from fredis.domain.fragment.reaction.ReactionPatternSet import (
    ReactionPatternSet,
)
from fredis.domain.fragment.tree.FragmentTreeBuilder import (
    FragmentTreeBuilder,
)
from fredis.domain.fragment.ion_tree.FragmentIonAdductRuleSet import (
    FragmentIonAdductRuleSet,
)
from fredis.domain.fragment.ion_tree.FragmentIonTreeBuilder import (
    FragmentIonTreeBuilder,
)


@dataclass(frozen=True)
class ExpectedFragmentTreeEdge:
    """Expected edge in a generated FragmentTree."""

    source_smiles: str
    target_smiles: str
    min_transition_count: int = 1


@dataclass(frozen=True)
class FragmentTreeBuilderCase:
    """Expected FragmentTree for one input compound."""

    name: str
    smiles: str

    expected_node_smiles_by_depth: dict[int, tuple[str, ...]]
    expected_edges: tuple[ExpectedFragmentTreeEdge, ...]

    expected_num_nodes: int
    expected_num_edges: int
    expected_min_transitions: int


@dataclass(frozen=True)
class FragmentTreeBuilderQuestion:
    """One test question for FragmentTreeBuilder.

    A question is a pair of:
    - FragmentTreeBuilder
    - FragmentTreeBuilderCase list

    This allows each question to define not only reaction patterns,
    but also builder options such as max_reaction_count.
    """

    name: str
    builder: FragmentTreeBuilder
    cases: tuple[FragmentTreeBuilderCase, ...]

    fragment_ion_adduct_rule_set: FragmentIonAdductRuleSet | None = None

def make_fragment_tree_builder_questions(
) -> tuple[FragmentTreeBuilderQuestion, ...]:
    """Return all FragmentTreeBuilder questions.

    Future questions should be added here.
    """

    return (
        make_single_bond_cleavage_question(),
    )


def make_single_bond_cleavage_question(
) -> FragmentTreeBuilderQuestion:
    builder = FragmentTreeBuilder(
        max_reaction_count=1,
        reaction_pattern_set=_make_single_bond_cleavage_pattern_set(),
    )
    fragment_ion_adduct_rule_set = _make_hydrogen_rearrangement_fragment_ion_adduct_rule_set()

    cases = (
        FragmentTreeBuilderCase(
            name="ethanol",
            smiles="CCO",
            expected_node_smiles_by_depth={
                0: ("CCO",),
                1: (
                    "C",
                    "CO",
                    "CC",
                    "O",
                ),
            },
            expected_edges=(
                ExpectedFragmentTreeEdge(
                    source_smiles="CCO",
                    target_smiles="C",
                    min_transition_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCO",
                    target_smiles="CO",
                    min_transition_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCO",
                    target_smiles="CC",
                    min_transition_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCO",
                    target_smiles="O",
                    min_transition_count=1,
                ),
            ),
            expected_num_nodes=5,
            expected_num_edges=4,
            expected_min_transitions=4,
        ),
        FragmentTreeBuilderCase(
            name="1-propanol",
            smiles="CCCO",
            expected_node_smiles_by_depth={
                0: ("CCCO",),
                1: (
                    "C",
                    "CCO",
                    "CC",
                    "CO",
                    "CCC",
                    "O",
                ),
            },
            expected_edges=(
                ExpectedFragmentTreeEdge(
                    source_smiles="CCCO",
                    target_smiles="C",
                    min_transition_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCCO",
                    target_smiles="CCO",
                    min_transition_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCCO",
                    target_smiles="CC",
                    min_transition_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCCO",
                    target_smiles="CO",
                    min_transition_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCCO",
                    target_smiles="CCC",
                    min_transition_count=1,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCCO",
                    target_smiles="O",
                    min_transition_count=1,
                ),
            ),
            expected_num_nodes=7,
            expected_num_edges=6,
            expected_min_transitions=6,
        ),
        FragmentTreeBuilderCase(
            name="diethyl_ether",
            smiles="CCOCC",
            expected_node_smiles_by_depth={
                0: ("CCOCC",),
                1: (
                    "C",
                    "CCOC",
                    "CC",
                    "CCO",
                ),
            },
            expected_edges=(
                ExpectedFragmentTreeEdge(
                    source_smiles="CCOCC",
                    target_smiles="C",
                    min_transition_count=2,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCOCC",
                    target_smiles="CCOC",
                    min_transition_count=2,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCOCC",
                    target_smiles="CC",
                    min_transition_count=2,
                ),
                ExpectedFragmentTreeEdge(
                    source_smiles="CCOCC",
                    target_smiles="CCO",
                    min_transition_count=2,
                ),
            ),
            expected_num_nodes=5,
            expected_num_edges=4,
            expected_min_transitions=8,
        ),
    )

    return FragmentTreeBuilderQuestion(
        name="non_hydrogen_single_bond_cleavage",
        builder=builder,
        cases=cases,
        fragment_ion_adduct_rule_set=fragment_ion_adduct_rule_set,
    )

def _make_single_bond_cleavage_pattern(
) -> _ReactionPattern:

    return _ReactionPattern.from_rules(
        reactant_smarts="[!#1:1]-[!#1:2]",
        products=(
            ProductRule(
                name="single_bond_cleavage",
                smarts="[!#1:1].[!#1:2]",
            ),
        ),
        name="single_bond_cleavage",
    )

def _make_single_bond_cleavage_pattern_set(
) -> ReactionPatternSet:
    pattern = _make_single_bond_cleavage_pattern()

    return ReactionPatternSet.from_patterns(
        name="single_bond_cleavage_pattern_set",
        patterns=(pattern,),
    )

def _make_hydrogen_rearrangement_fragment_ion_adduct_rule_set(
) -> FragmentIonAdductRuleSet:
    test_data_dir = Path(__file__).parent / "test_data"

    return FragmentIonAdductRuleSet.from_json(
        test_data_dir / "hydrogen_rearrangement_rule_set_pos.json",
    )