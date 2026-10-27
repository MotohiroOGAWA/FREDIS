from __future__ import annotations

import unittest

from fredis.libs.mmkit.mmkit import Compound

from fredis.domain.fragment.reaction._ReactionPattern import (
    _ReactionPattern,
    ProductRule,
)
from fredis.domain.fragment.reaction.ReactionPatternSet import (
    ReactionPattern,
    ReactionPatternSet,
    ReactionResult,
)


class TestReactionPatternSet(unittest.TestCase):
    def make_pattern(
        self,
        *,
        reactant_smarts: str,
        product_smarts: str,
        name: str,
    ) -> _ReactionPattern:
        return _ReactionPattern.from_rules(
            reactant_smarts=reactant_smarts,
            products=(
                ProductRule(
                    name=f"{name}_product",
                    smarts=product_smarts,
                ),
            ),
            name=name,
        )

    def test_from_patterns_assigns_stable_pattern_ids(self) -> None:
        pattern_b = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="c_o_cleavage",
        )

        pattern_a = self.make_pattern(
            reactant_smarts="[#6:1]-[#6:2]",
            product_smarts="[#6:1].[#6:2]",
            name="c_c_cleavage",
        )

        pattern_set = ReactionPatternSet.from_patterns(
            patterns=(pattern_b, pattern_a),
            name="test_patterns",
        )

        self.assertEqual(pattern_set.name, "test_patterns")
        self.assertEqual(len(pattern_set), 2)
        self.assertEqual(pattern_set.pattern_ids, (0, 1))

        for pattern_id, pattern in enumerate(pattern_set.patterns):
            self.assertIsInstance(pattern, ReactionPattern)
            self.assertEqual(pattern.pattern_id, pattern_id)

    def test_from_patterns_removes_duplicate_patterns(self) -> None:
        pattern_1 = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="first_name",
        )

        pattern_2 = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="second_name",
        )

        pattern_set = ReactionPatternSet.from_patterns(
            patterns=(pattern_1, pattern_2),
            name="deduplicated_patterns",
        )

        self.assertEqual(len(pattern_set), 1)
        self.assertEqual(pattern_set.pattern_ids, (0,))

    def test_from_patterns_raises_type_error_when_invalid_pattern_is_given(self) -> None:
        with self.assertRaises(TypeError):
            ReactionPatternSet.from_patterns(
                patterns=("not_a_pattern",),
                name="invalid_patterns",
            )

    def test_iter_returns_patterns_in_id_order(self) -> None:
        pattern_1 = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="c_o_cleavage",
        )

        pattern_2 = self.make_pattern(
            reactant_smarts="[#6:1]-[#7:2]",
            product_smarts="[#6:1].[#7:2]",
            name="c_n_cleavage",
        )

        pattern_set = ReactionPatternSet.from_patterns(
            patterns=(pattern_2, pattern_1),
            name="test_patterns",
        )

        iterated_patterns = tuple(pattern_set)

        self.assertEqual(iterated_patterns, pattern_set.patterns)
        self.assertEqual(
            tuple(pattern.pattern_id for pattern in iterated_patterns),
            pattern_set.pattern_ids,
        )

    def test_contains_returns_true_for_equivalent_base_pattern(self) -> None:
        pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="c_o_cleavage",
        )

        equivalent_pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="different_name",
        )

        pattern_set = ReactionPatternSet.from_patterns(
            patterns=(pattern,),
            name="test_patterns",
        )

        self.assertIn(equivalent_pattern, pattern_set)

    def test_get_pattern_returns_pattern_by_id(self) -> None:
        base_pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="c_o_cleavage",
        )

        pattern_set = ReactionPatternSet.from_patterns(
            patterns=(base_pattern,),
            name="test_patterns",
        )

        pattern = pattern_set.get_pattern(0)

        self.assertIsInstance(pattern, ReactionPattern)
        self.assertEqual(pattern.pattern_id, 0)
        self.assertEqual(pattern.key, base_pattern.key)

    def test_get_pattern_raises_key_error_when_id_is_invalid(self) -> None:
        base_pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="c_o_cleavage",
        )

        pattern_set = ReactionPatternSet.from_patterns(
            patterns=(base_pattern,),
            name="test_patterns",
        )

        with self.assertRaises(KeyError):
            pattern_set.get_pattern(999)

    def test_get_pattern_id_returns_id_for_equivalent_pattern(self) -> None:
        base_pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="c_o_cleavage",
        )

        equivalent_pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="different_name",
        )

        pattern_set = ReactionPatternSet.from_patterns(
            patterns=(base_pattern,),
            name="test_patterns",
        )

        pattern_id = pattern_set.get_pattern_id(equivalent_pattern)

        self.assertEqual(pattern_id, 0)

    def test_get_pattern_id_raises_key_error_when_pattern_is_missing(self) -> None:
        base_pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="c_o_cleavage",
        )

        missing_pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#7:2]",
            product_smarts="[#6:1].[#7:2]",
            name="c_n_cleavage",
        )

        pattern_set = ReactionPatternSet.from_patterns(
            patterns=(base_pattern,),
            name="test_patterns",
        )

        with self.assertRaises(KeyError):
            pattern_set.get_pattern_id(missing_pattern)

    def test_fragment_by_id_returns_result_with_pattern_id(self) -> None:
        base_pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="c_o_cleavage",
        )

        pattern_set = ReactionPatternSet.from_patterns(
            patterns=(base_pattern,),
            name="test_patterns",
        )

        compound = Compound.from_smiles("CCO")

        result = pattern_set.fragment_by_id(
            pattern_id=0,
            compound=compound,
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, ReactionResult)
        self.assertEqual(result.pattern_id, 0)
        self.assertEqual(result.pattern.pattern_id, 0)
        self.assertEqual(result.reactant_compound.smiles, compound.smiles)
        self.assertGreater(len(result.products), 0)

    def test_fragment_by_id_returns_none_when_pattern_does_not_match(self) -> None:
        base_pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#7:2]",
            product_smarts="[#6:1].[#7:2]",
            name="c_n_cleavage",
        )

        pattern_set = ReactionPatternSet.from_patterns(
            patterns=(base_pattern,),
            name="test_patterns",
        )

        compound = Compound.from_smiles("CCO")

        result = pattern_set.fragment_by_id(
            pattern_id=0,
            compound=compound,
        )

        self.assertIsNone(result)

    def test_fragment_all_returns_only_matching_results(self) -> None:
        matching_pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="c_o_cleavage",
        )

        non_matching_pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#7:2]",
            product_smarts="[#6:1].[#7:2]",
            name="c_n_cleavage",
        )

        pattern_set = ReactionPatternSet.from_patterns(
            patterns=(matching_pattern, non_matching_pattern),
            name="test_patterns",
        )

        compound = Compound.from_smiles("CCO")

        results = pattern_set.fragment_all(compound)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], ReactionResult)
        self.assertIn(results[0].pattern_id, pattern_set.pattern_ids)
        self.assertEqual(results[0].reactant_compound.smiles, compound.smiles)
        self.assertGreater(len(results[0].products), 0)

    def test_identity_does_not_include_name(self) -> None:
        pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="c_o_cleavage",
        )

        pattern_set_1 = ReactionPatternSet.from_patterns(
            patterns=(pattern,),
            name="first_set_name",
        )

        pattern_set_2 = ReactionPatternSet.from_patterns(
            patterns=(pattern,),
            name="second_set_name",
        )

        self.assertEqual(
            pattern_set_1.identity(),
            pattern_set_2.identity(),
        )

    def test_to_dict_returns_serializable_metadata(self) -> None:
        pattern = self.make_pattern(
            reactant_smarts="[#6:1]-[#8:2]",
            product_smarts="[#6:1].[#8:2]",
            name="c_o_cleavage",
        )

        pattern_set = ReactionPatternSet.from_patterns(
            patterns=(pattern,),
            name="test_patterns",
        )

        data = pattern_set.to_dict()

        self.assertEqual(data["name"], "test_patterns")
        self.assertEqual(len(data["patterns"]), 1)

        pattern_data = data["patterns"][0]

        self.assertEqual(pattern_data["pattern_id"], 0)
        self.assertEqual(pattern_data["name"], "c_o_cleavage")
        self.assertEqual(pattern_data["reactant_smarts"], "[#6:1]-[#8:2]")
        self.assertEqual(
            pattern_data["products"],
            [
                {
                    "name": "c_o_cleavage_product",
                    "smarts": "[#6:1].[#8:2]",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()