import unittest
from typing import Dict, Tuple

from rdkit import Chem

from fredis.domain.fragment.reaction._ReactionPattern import (
    _ReactionPattern,
    ProductRule,
)
from fredis.libs.mmkit.mmkit import Compound


class TestReactionPattern(unittest.TestCase):
    def make_pattern_from_smirks(
        self,
        smirks: str,
        *,
        name: str = "test cleavage",
        rule_name: str = "test rule",
    ) -> _ReactionPattern:
        if ">>" not in smirks:
            raise ValueError(f"Invalid SMIRKS. Missing '>>': {smirks}")

        reactant_smarts, product_smarts = smirks.split(">>", maxsplit=1)

        return _ReactionPattern.from_rules(
            reactant_smarts=reactant_smarts,
            products=(ProductRule(name=rule_name, smarts=product_smarts),),
            name=name,
        )

    def get_atom_symbols_from_smarts(self, smarts: str) -> Tuple[str, ...]:
        mol = Chem.MolFromSmarts(smarts)

        if mol is None:
            raise ValueError(f"Invalid SMARTS: {smarts}")

        return tuple(atom.GetSymbol() for atom in mol.GetAtoms())


    def get_product_atom_symbols_from_smarts(
        self,
        product_smarts: str,
    ) -> Tuple[Tuple[str, ...], ...]:
        product_symbols = []

        for part in product_smarts.split("."):
            mol = Chem.MolFromSmarts(part)

            if mol is None:
                raise ValueError(f"Invalid product SMARTS: {part}")

            product_symbols.append(
                tuple(atom.GetSymbol() for atom in mol.GetAtoms())
            )

        return tuple(product_symbols)


    def get_atom_symbols_from_smiles(self, smiles: str) -> Tuple[str, ...]:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        return tuple(atom.GetSymbol() for atom in mol.GetAtoms())

    def assert_reaction_product_atom_symbols_are_consistent(
        self,
        *,
        pattern: _ReactionPattern,
        reaction_product,
        compound: Compound,
    ) -> None:
        product_rule = next(
            product
            for product in pattern.products
            if product.name == reaction_product.rule_name
        )

        reactant_smarts_symbols = self.get_atom_symbols_from_smarts(
            pattern.reactant_smarts
        )
        reactant_smiles_symbols = self.get_atom_symbols_from_smiles(
            compound.smiles
        )

        self.assertEqual(
            len(reaction_product.reactant_indices),
            len(reactant_smarts_symbols),
        )

        for reactant_smarts_index, actual_index in enumerate(
            reaction_product.reactant_indices
        ):
            self.assertGreaterEqual(actual_index, 0)

            expected_symbol = reactant_smarts_symbols[reactant_smarts_index]
            actual_symbol = reactant_smiles_symbols[actual_index]

            self.assertEqual(
                actual_symbol,
                expected_symbol,
                msg=(
                    "Reactant atom symbol mismatch: "
                    f"reactant_smarts_index={reactant_smarts_index}, "
                    f"actual_index={actual_index}, "
                    f"expected_symbol={expected_symbol}, "
                    f"actual_symbol={actual_symbol}"
                ),
            )

        product_smarts_symbols_list = self.get_product_atom_symbols_from_smarts(
            product_rule.smarts
        )

        self.assertEqual(
            len(reaction_product.product_molecules),
            len(product_smarts_symbols_list),
        )

        for molecule, product_smarts_symbols in zip(
            reaction_product.product_molecules,
            product_smarts_symbols_list,
        ):
            product_smiles_symbols = self.get_atom_symbols_from_smiles(
                molecule.compound.smiles
            )

            self.assertEqual(
                len(molecule.product_indices),
                len(product_smarts_symbols),
            )

            for product_smarts_index, actual_index in enumerate(
                molecule.product_indices
            ):
                self.assertGreaterEqual(actual_index, 0)

                expected_symbol = product_smarts_symbols[product_smarts_index]
                actual_symbol = product_smiles_symbols[actual_index]

                self.assertEqual(
                    actual_symbol,
                    expected_symbol,
                    msg=(
                        "Product atom symbol mismatch: "
                        f"product_smarts_index={product_smarts_index}, "
                        f"actual_index={actual_index}, "
                        f"expected_symbol={expected_symbol}, "
                        f"actual_symbol={actual_symbol}, "
                        f"molecule_smiles={molecule.compound.smiles}"
                    ),
                )

    def test_fragment_various_bond_reaction_patterns(self) -> None:
        cases = (
            {
                "smirks": "[C:1]-[O:2]>>[C:1]",
                "reactant_smiles": "CCCOCC",
                "expected_first_product_smiles": "CCC",
                "expected_product_count": 2,
            },
            {
                "smirks": "[C:1]-[N:2]>>[C:1]",
                "reactant_smiles": "CCCN(CC)CC",
                "expected_first_product_smiles": "CCC",
                "expected_product_count": 3,
            },
            {
                "smirks": "[C:1]-[S:2]>>[C:1]",
                "reactant_smiles": "CCCSCC",
                "expected_first_product_smiles": "CCC",
                "expected_product_count": 2,
            },
            {
                "smirks": "[C:1]-[Cl:2]>>[C:1]",
                "reactant_smiles": "CCCCCl",
                "expected_first_product_smiles": "CCCC",
                "expected_product_count": 1,
            },
            {
                "smirks": "[C:1]-[Br:2]>>[C:1]",
                "reactant_smiles": "CCCCBr",
                "expected_first_product_smiles": "CCCC",
                "expected_product_count": 1,
            },
            {
                "smirks": "[C:1]-[C:2](=[O:3])>>[C:1]",
                "reactant_smiles": "CCCC(=O)OCC",
                "expected_first_product_smiles": "CCC",
                "expected_product_count": 1,
            },
        )

        for case in cases:
            with self.subTest(
                smirks=case["smirks"],
                reactant=case["reactant_smiles"],
            ):
                pattern = self.make_pattern_from_smirks(case["smirks"])
                compound = Compound.from_smiles(case["reactant_smiles"])

                result = pattern.fragment(compound)

                self.assertIsNotNone(result)
                self.assertEqual(result.reactant_compound.smiles, compound.smiles)
                self.assertEqual(
                    len(result.products),
                    case["expected_product_count"],
                )

                reaction_product = result.products[0]
                molecule = reaction_product.product_molecules[0]

                self.assertEqual(
                    molecule.compound.smiles,
                    case["expected_first_product_smiles"],
                )

                self.assert_reaction_product_atom_symbols_are_consistent(
                    pattern=pattern,
                    reaction_product=reaction_product,
                    compound=compound,
                )

    def test_fragment_multiple_product_molecules(self) -> None:
        pattern = _ReactionPattern.from_rules(
            reactant_smarts="[C:1]-[O:2]-[C:3]",
            products=(
                ProductRule(
                    name="split ether",
                    smarts="[C:1].[O:2]-[C:3]",
                ),
            ),
            name="multi product cleavage",
        )

        compound = Compound.from_smiles("CCCOCC")

        result = pattern.fragment(compound)

        self.assertIsNotNone(result)
        self.assertGreater(len(result.products), 0)

        reaction_product = result.products[0]

        self.assertEqual(reaction_product.rule_name, "split ether")
        self.assertEqual(len(reaction_product.product_molecules), 2)

        product_smiles = {
            molecule.compound.smiles
            for molecule in reaction_product.product_molecules
        }

        self.assertIn("CCC", product_smiles)
        self.assertIn("CCO", product_smiles)

        self.assert_reaction_product_atom_symbols_are_consistent(
            pattern=pattern,
            reaction_product=reaction_product,
            compound=compound,
        )

    def test_fragment_flavonoid_like_pattern(self) -> None:
        reactant_smarts = "[#8:1]=[#6:2]1:[#6:3]:[#6:4](-[#6:5]2:[#6:6]:[#6:7]:[#6:8](-[#8:9]):[#6:10]:[#6:11]:2):[#8:12]:[#6:13]2:[#6:14]:[#6:15](-[#8:16]):[#6:17]:[#6:18](-[#8:19]):[#6:20]:1:2"

        product_rules = (
            ProductRule(name="1,3", smarts="[#8:12]-[#6:13]1:[#6:14]:[#6:15](-[#8:16]):[#6:17]:[#6:18](-[#8:19]):[#6:20]:1.[#8:1]=[#6:2]-[#6:3]=[#6:4](-[#6:5]1:[#6:6]:[#6:7]:[#6:8](-[#8:9]):[#6:10]:[#6:11]:1)"),
            ProductRule(name="1,4", smarts="[#6:20]1:[#6:13]:[#6:14]:[#6:15](-[#8:16]):[#6:17]:[#6:18](-[#8:19]):1.[#6:4](-[#6:5]1:[#6:6]:[#6:7]:[#6:8](-[#8:9]):[#6:10]:[#6:11]:1)=[#8:12]"),
            ProductRule(name="1,2", smarts="[#6:4](-[#6:5]1:[#6:6]:[#6:7]:[#6:8](-[#8:9]):[#6:10]:[#6:11]:1)=[#6:3]"),
            ProductRule(name="0,4", smarts="[#6:5]1:[#6:6]:[#6:7]:[#6:8](-[#8:9]):[#6:10]:[#6:11]:1"),
        )

        pattern = _ReactionPattern.from_rules(
            reactant_smarts=reactant_smarts,
            products=product_rules,
            name="Example Reaction",
        )

        compound = Compound.from_smiles(
            "O=c1cc(-c2ccc(O)c(CCC)c2)oc2cc(O)cc(O)c12"
        )

        result = pattern.fragment(compound)

        self.assertIsNotNone(result)
        self.assertEqual(result.reactant_compound.smiles, compound.smiles)
        self.assertGreater(len(result.products), 0)

        rule_names = {
            reaction_product.rule_name
            for reaction_product in result.products
        }

        self.assertTrue(
            rule_names.issubset({"1,3", "1,4", "1,2", "0,4"})
        )

        multi_product_results = [
            reaction_product
            for reaction_product in result.products
            if reaction_product.rule_name in {"1,3", "1,4"}
        ]

        self.assertGreater(len(multi_product_results), 0)

        self.assertTrue(
            any(
                len(reaction_product.product_molecules) == 2
                for reaction_product in multi_product_results
            )
        )

        for reaction_product in result.products:
            self.assertGreater(len(reaction_product.reactant_indices), 0)
            self.assertTrue(
                all(index >= 0 for index in reaction_product.reactant_indices)
            )

            self.assertGreater(len(reaction_product.product_molecules), 0)

            for molecule in reaction_product.product_molecules:
                self.assertTrue(molecule.compound.smiles)
                self.assertGreater(len(molecule.product_indices), 0)
                self.assertTrue(
                    all(index >= 0 for index in molecule.product_indices)
                )

            self.assert_reaction_product_atom_symbols_are_consistent(
                pattern=pattern,
                reaction_product=reaction_product,
                compound=compound,
            )

    def test_reaction_pattern_deduplicates_products_by_smarts(self) -> None:
        pattern = _ReactionPattern.from_rules(
            reactant_smarts="[C:1]-[O:2]",
            products=(
                ProductRule(name="rule_1", smarts="[C:1]"),
                ProductRule(name="rule_2", smarts="[C:1]"),
                ProductRule(name="rule_3", smarts="[O:2]"),
            ),
            name="test",
        )

        self.assertEqual(len(pattern.products), 2)
        self.assertEqual(len(pattern.reaction_rules), 2)

        self.assertEqual(pattern.products[0].name, "rule_1")
        self.assertEqual(pattern.products[0].smarts, "[C:1]")
        self.assertEqual(pattern.products[1].name, "rule_3")
        self.assertEqual(pattern.products[1].smarts, "[O:2]")


    def test_reaction_pattern_hash_and_eq_ignore_compiled_objects_and_name(self) -> None:
        pattern_1 = _ReactionPattern.from_rules(
            reactant_smarts="[C:1]-[O:2]",
            products=(ProductRule(name="a", smarts="[C:1]"),),
            name="pattern_a",
        )

        pattern_2 = _ReactionPattern.from_rules(
            reactant_smarts="[C:1]-[O:2]",
            products=(ProductRule(name="b", smarts="[C:1]"),),
            name="pattern_b",
        )

        self.assertEqual(pattern_1, pattern_2)
        self.assertEqual(hash(pattern_1), hash(pattern_2))