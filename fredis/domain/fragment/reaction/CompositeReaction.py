"""Compile concrete bond and atom deletion reactions into one Source-specific RDKit reaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from fredis.libs.mmkit.mmkit import Compound
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdqueries
from .ReactionInstance import mapped_source
from .ReactionSequence import ReactionSequence


@dataclass(frozen=True)
class CompositeReaction:
    reaction_sequence: ReactionSequence
    reactant_smarts: str
    product_smarts: str
    smirks: str
    rxn: rdChemReactions.ChemicalReaction = field(compare=False, repr=False)
    _source_smiles: str = field(repr=False)

    @classmethod
    def from_sequence(
        cls,
        *,
        source: Compound | Chem.Mol,
        reaction_sequence: ReactionSequence,
    ) -> CompositeReaction:
        mol = mapped_source(source)
        by_map = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
        source_bonds = {
            tuple(
                sorted(
                    (b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())
                )
            )
            for b in mol.GetBonds()
        }
        for reaction in reaction_sequence.reactions:
            if reaction.retained_atom_maps | reaction.discarded_atom_maps != frozenset(
                by_map
            ):
                raise ValueError("Reaction atom universe differs from Source")
            if (
                not (reaction.matched_bond_maps | reaction.changed_bond_maps)
                <= source_bonds
            ):
                raise ValueError("Reaction references bonds absent from Source")
        retained = reaction_sequence.retained_atom_maps
        if not retained:
            raise ValueError("Composite reaction must retain at least one Source atom")
        target = Chem.RWMol(mol)
        for u, v in sorted(reaction_sequence.cut_bond_maps):
            target.RemoveBond(by_map[u], by_map[v])
        for u, v, order in sorted(reaction_sequence.bond_updates):
            if {u, v} <= retained:
                target.GetBondBetweenAtoms(by_map[u], by_map[v]).SetBondType(order)
        for atom in reversed(list(mol.GetAtoms())):
            if atom.GetAtomMapNum() not in retained:
                target.RemoveAtom(atom.GetIdx())
        target = target.GetMol()
        removed_edges = reaction_sequence.cut_bond_maps | frozenset(
            edge for edge in source_bonds if not set(edge) <= retained
        )
        affected = (
            {m for edge in removed_edges for m in edge}
            | {m for u, v, _ in reaction_sequence.bond_updates for m in (u, v)}
        ) & retained
        # Recompute hydrogens at deletion boundaries, including bracketed mapped SMILES.
        for atom in target.GetAtoms():
            if atom.GetAtomMapNum() in affected and not atom.GetNumRadicalElectrons():
                atom.SetNumExplicitHs(0)
                atom.SetNoImplicit(False)
        Chem.SanitizeMol(target)
        # Canonical SMILES orders atoms consistently, independent of source indices.
        ordered_source = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
        ordered_target = Chem.MolFromSmiles(Chem.MolToSmiles(target, canonical=True))
        reactant = Chem.MolToSmarts(ordered_source)
        product = Chem.MolToSmiles(ordered_target, canonical=True)
        # Parentheses make disconnected retained components one product molecule.
        smirks = f"({reactant})>>({product})"
        rxn = rdChemReactions.ReactionFromSmarts(smirks)
        if rxn is None:
            raise ValueError(f"Cannot compile composite SMIRKS: {smirks}")
        # Atom maps alone do not constrain RDKit matching; bind the concrete locations.
        for atom in rxn.GetReactantTemplate(0).GetAtoms():
            atom.ExpandQuery(
                rdqueries.HasIntPropWithValueQueryAtom(
                    "molAtomMapNumber", atom.GetAtomMapNum()
                )
            )
        # Atom deletion intentionally leaves Source maps absent from the target.
        # RDKit's unmapped-product warning is expected; exceptions still propagate.
        rxn.Initialize(silent=True)
        return cls(
            reaction_sequence,
            reactant,
            product,
            smirks,
            rxn,
            Chem.MolToSmiles(mol, canonical=True),
        )

    def run(self, source: Compound | Chem.Mol) -> tuple[Chem.Mol, ...]:
        """Run once on the exact mapped Source and restore Source atom identities."""
        mol = mapped_source(source)
        if Chem.MolToSmiles(mol, canonical=True) != self._source_smiles:
            raise ValueError("Composite reaction requires its original mapped Source")
        groups = self.rxn.RunReactants((mol,))
        results = []
        for group in groups:
            product = group[0]
            for atom in product.GetAtoms():
                if atom.HasProp("react_atom_idx"):
                    atom.SetAtomMapNum(
                        mol.GetAtomWithIdx(
                            atom.GetIntProp("react_atom_idx")
                        ).GetAtomMapNum()
                    )
            Chem.SanitizeMol(product)
            results.append(product)
        return tuple(results)
