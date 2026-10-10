"""Decide which tree edges apply to which compounds.

An edge applies to a compound when its precursor SMILES matches the
``reactant_smarts`` of a pattern the child set adds.  Source reactions are only
ever matched on the precursor, so for any other compound the child condition
equals the parent condition.  ``max_reaction_count`` plays no role here.
"""
from __future__ import annotations

from typing import Any, Iterable


def compile_reactant_smarts(patterns: Iterable[dict]) -> dict[int, Any]:
    """Compile every added pattern's reactant SMARTS exactly once."""
    from rdkit import Chem

    compiled: dict[int, Any] = {}
    by_smarts: dict[str, Any] = {}
    for pattern in patterns:
        smarts = str(pattern.get("reactant_smarts", "")).strip()
        query = by_smarts.get(smarts)
        if query is None:
            query = Chem.MolFromSmarts(smarts) if smarts else None
            if query is None:
                raise ValueError(f"Invalid reactant SMARTS for {pattern.get('name')}: {smarts}")
            by_smarts[smarts] = query
        compiled[int(pattern["pattern_id"])] = query
    return compiled


def match_precursors(compound_smiles: dict[str, str], compiled: dict[int, Any]) -> dict[str, list[int]]:
    """Return ``compound_id -> matched pattern ids`` (one RDKit parse per compound).

    Callers pass one SMILES per unique compound, so samples sharing a
    compound (other CE, adduct or dataset) reuse this single result.
    """
    from rdkit import Chem

    matched: dict[str, list[int]] = {}
    for compound_id, smiles in compound_smiles.items():
        molecule = Chem.MolFromSmiles(smiles)
        matched[compound_id] = ([] if molecule is None else
                                [pattern_id for pattern_id, query in compiled.items()
                                 if molecule.HasSubstructMatch(query)])
    return matched
