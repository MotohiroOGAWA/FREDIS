import unittest
from dataclasses import replace
from rdkit import Chem
from fredis.domain.fragment.reaction import ReactionInstance, ReactionSequence
from fredis.domain.fragment.reaction._ReactionPattern import _ReactionPattern, ProductRule
from fredis.libs.mmkit.mmkit import Compound


class TestReactionInstance(unittest.TestCase):
    def reaction(self, roles=(1, 2), retained=(1, 2), pattern_id=0, matched=None, cuts=(), universe=(1, 2, 3)):
        return ReactionInstance(pattern_id, 0, roles, frozenset(retained),
                              frozenset(universe) - frozenset(retained),
                              frozenset(matched if matched is not None else [roles]), frozenset(cuts), product_molecule_id=0)

    def test_subset_order_duplicates_and_identity(self):
        a = self.reaction(retained=(1,2,3))
        b = self.reaction(retained=(1,2), pattern_id=1)
        seq = ReactionSequence((a, b, a))
        self.assertEqual(seq.reactions, (b,))
        self.assertEqual(seq, ReactionSequence((b, a)))
        self.assertEqual(hash(seq), hash(ReactionSequence((b,))))

    def test_shared_bond_rejected_before_redundancy(self):
        a = self.reaction(cuts=((1, 2),))
        b = self.reaction(roles=(2, 1), retained=(1,), pattern_id=1)
        for reactions in ((a, b), (b, a), (a, a, b)):
            with self.subTest(reactions=reactions), self.assertRaisesRegex(ValueError, 'share changed Source bonds'):
                ReactionSequence(reactions)
        self.assertEqual(ReactionSequence((a, a)).reactions, (a,))

    def test_context_bond_and_three_reactions(self):
        a = self.reaction(roles=(1, 2, 3), retained=(1, 2, 3), matched=((1, 2), (2, 3)), cuts=((1, 2),))
        b = self.reaction(roles=(2, 3), retained=(1, 2, 3), pattern_id=1, cuts=((2, 3),))
        c = self.reaction(roles=(1,), retained=(1, 2, 3), matched=(), pattern_id=2)
        self.assertTrue(ReactionSequence((c, a, b)).reactions)

    def test_shared_atom_allowed_and_single_rdkit_execution(self):
        source = Chem.MolFromSmiles('[CH3:1][CH:2]([OH:3])[NH2:4]')
        universe = (1, 2, 3, 4)
        a = self.reaction(roles=(2, 3), retained=(1, 2, 4), universe=universe)
        b = self.reaction(roles=(2, 4), retained=(1, 2, 3), universe=universe, pattern_id=1)
        seq = ReactionSequence((a, b))
        self.assertEqual(len(seq.reactions), 2)
        self.assertEqual(seq.retained_atom_maps, frozenset((1, 2)))
        compiled = seq.compile(source)
        self.assertEqual(compiled.smirks, ReactionSequence((b, a)).compile(source).smirks)
        products = compiled.run(source)
        self.assertEqual(len(products), 1)
        self.assertEqual(Chem.MolToSmiles(products[0]), '[CH3:1][CH3:2]')
        self.assertEqual(source.GetNumAtoms(), 4)
        self.assertEqual(len(compiled.rxn.RunReactants((source,))), 1)
        reordered = Chem.RenumberAtoms(source, (3, 2, 1, 0))
        self.assertEqual(Chem.MolToSmiles(compiled.run(reordered)[0]), Chem.MolToSmiles(products[0]))
        self.assertEqual(compiled.smirks, seq.compile(reordered).smirks)

    def test_surviving_cut_is_not_removed_by_subset(self):
        a = self.reaction(retained=(1, 2, 3), cuts=((1, 2),))
        b = self.reaction(roles=(2, 3), retained=(1, 2), pattern_id=1)
        seq = ReactionSequence((a, b))
        self.assertEqual(len(seq.reactions), 2)
        result = seq.compile(Chem.MolFromSmiles('[CH3:1][CH2:2][CH3:3]')).run(
            Chem.MolFromSmiles('[CH3:1][CH2:2][CH3:3]'))[0]
        self.assertEqual(result.GetNumBonds(), 0)

    def test_equal_retention_does_not_remove_all_reactions(self):
        a = self.reaction()
        b = self.reaction(roles=(2, 1), pattern_id=1)
        self.assertEqual(ReactionSequence((b, a)).reactions, (a,))

    def test_existing_rules_manual_match_with_compound(self):
        source = Compound.from_smiles('CC(O)N')
        mol = source.mapped_mol
        center = next(a for a in mol.GetAtoms() if a.GetSymbol() == 'C' and a.GetDegree() == 3)
        all_maps = frozenset(a.GetAtomMapNum() for a in mol.GetAtoms())
        reactions = []
        for pattern_id, symbol in enumerate(('O', 'N')):
            terminal = next(a for a in mol.GetAtoms() if a.GetSymbol() == symbol)
            pattern = _ReactionPattern.from_rules(reactant_smarts=f'[C:1]-[{symbol}:2]',
                                                  products=(ProductRule('retain carbon', '[C:1]'),))
            reactions.append(ReactionInstance.from_match(
                source=source, product_molecule_id=0, reaction_pattern=pattern, reaction_pattern_id=pattern_id,
                source_atom_maps=(center.GetAtomMapNum(), terminal.GetAtomMapNum()),
                retained_atom_maps=all_maps - {terminal.GetAtomMapNum()}))
        result = ReactionSequence(reactions).compile(source).run(source)[0]
        for atom in result.GetAtoms():
            atom.SetAtomMapNum(0)
        self.assertEqual(Chem.MolToSmiles(result), 'CC')

    def test_query_bonds_only_and_invalid_match(self):
        source = Chem.MolFromSmiles('[CH2:1]1[CH2:2][CH2:3]1')
        pattern = _ReactionPattern.from_rules(reactant_smarts='[C:1]-[C:2]-[C:3]',
                                              products=(ProductRule('keep', '[C:1]-[C:2]'),))
        a = ReactionInstance.from_match(source=source, product_molecule_id=0, reaction_pattern=pattern, reaction_pattern_id=0,
                                     source_atom_maps=(1, 2, 3), retained_atom_maps=(1, 2))
        self.assertEqual(a.changed_bond_maps, frozenset(((1, 3), (2, 3))))
        self.assertEqual(a.matched_bond_maps, frozenset(((1, 2), (2, 3))))
        wrong = _ReactionPattern.from_rules(reactant_smarts='[C:1]-[O:2]',
                                            products=(ProductRule('keep', '[C:1]'),))
        with self.assertRaisesRegex(ValueError, 'do not match'):
            ReactionInstance.from_match(source=source, product_molecule_id=0, reaction_pattern=wrong, reaction_pattern_id=0,
                                      source_atom_maps=(1, 2), retained_atom_maps=(1,))

    def test_product_selection_and_bond_order(self):
        source = Chem.MolFromSmiles('[CH3:10][CH2:20][CH3:30]')
        pattern = _ReactionPattern.from_rules(
            reactant_smarts='[C:7]-[C:2]-[C:9]',
            products=(ProductRule('split', '[C:7]=[C:2].[C:9]'),))
        a = ReactionInstance.from_match(source=source, reaction_pattern=pattern,
            reaction_pattern_id=0, product_molecule_id=0,
            source_atom_maps=(10, 20, 30), retained_atom_maps=(10, 20))
        self.assertEqual(a.source_atom_maps, (10, 20, 30))
        self.assertEqual(a.bond_updates, frozenset(((10, 20, Chem.BondType.DOUBLE),)))
        self.assertEqual(a.changed_bond_maps, frozenset(((10, 20), (20, 30))))
        product = ReactionSequence((a,)).compile(source).run(source)[0]
        self.assertEqual(Chem.MolToSmiles(product), '[CH2:10]=[CH2:20]')
        b = ReactionInstance.from_match(source=source, reaction_pattern=pattern,
            reaction_pattern_id=0, product_molecule_id=1,
            source_atom_maps=(10, 20, 30), retained_atom_maps=(30,))
        self.assertNotEqual(a.key, b.key)
        self.assertEqual(Chem.MolToSmiles(ReactionSequence((b,)).compile(source).run(source)[0]), '[CH4:30]')
        with self.assertRaisesRegex(ValueError, 'product_molecule_id'):
            ReactionInstance.from_match(source=source, reaction_pattern=pattern,
                reaction_pattern_id=0, product_molecule_id=2,
                source_atom_maps=(10, 20, 30), retained_atom_maps=(30,))

    def test_order_change_overlap_rejected(self):
        a = replace(self.reaction(retained=(1, 2, 3)),
                    bond_updates=frozenset(((1, 2, Chem.BondType.DOUBLE),)))
        b = self.reaction(pattern_id=1, cuts=((1, 2),))
        with self.assertRaisesRegex(ValueError, 'share changed Source bonds'):
            ReactionSequence((a, b))

    def test_all_bond_pairs_are_normalized(self):
        a = replace(self.reaction(roles=(1, 2, 3), retained=(1, 2, 3), matched=((2, 1), (3, 2)),
                                cuts=((3, 2),)),
                    bond_updates=frozenset(((2, 1, Chem.BondType.DOUBLE),)),
                    changed_bond_maps=frozenset(((3, 1),)))
        self.assertEqual(a.matched_bond_maps, frozenset(((1, 2), (2, 3))))
        self.assertEqual(a.cut_bond_maps, frozenset(((2, 3),)))
        self.assertEqual(a.bond_updates, frozenset(((1, 2, Chem.BondType.DOUBLE),)))
        self.assertEqual(a.changed_bond_maps, frozenset(((1, 2), (1, 3), (2, 3))))
        b = replace(a, matched_bond_maps=frozenset(((1, 2), (2, 3))),
                    cut_bond_maps=frozenset(((2, 3),)),
                    bond_updates=frozenset(((1, 2, Chem.BondType.DOUBLE),)))
        self.assertEqual(a, b)
        self.assertEqual(a.key, b.key)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(ReactionSequence((a, b)).reactions, (a,))
        conflict = self.reaction(roles=(3, 2), retained=(1, 2, 3),
                               pattern_id=1, cuts=((3, 2),))
        with self.assertRaisesRegex(ValueError, 'share changed Source bonds'):
            ReactionSequence((a, conflict))

    def test_validation(self):
        with self.assertRaises(ValueError):
            ReactionSequence(())
        with self.assertRaises(ValueError):
            replace(self.reaction(), cut_bond_maps=frozenset(((1, 3),)))
        with self.assertRaises(ValueError):
            ReactionSequence((self.reaction(), self.reaction(roles=(2, 3), universe=(1, 2, 3, 4))))
        with self.assertRaises(ValueError):
            ReactionSequence((self.reaction(),)).compile(Chem.MolFromSmiles('CCC'))
        compiled = ReactionSequence((self.reaction(),)).compile(Chem.MolFromSmiles('[CH3:1][CH2:2][CH3:3]'))
        with self.assertRaises(ValueError):
            compiled.run(Chem.MolFromSmiles('[CH3:1][CH2:2][OH:3]'))
