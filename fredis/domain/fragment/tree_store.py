"""A shared store of complete fragment trees (every reaction history and its product).

Reaction Pattern Analysis and Training Data preparation both build trees from
Source with the same search.  A tree depends only on

* the Source SMILES string (atom order matters: histories are atom-mapped),
* the Fragmenter settings except the reaction patterns, and the search limits,
* the reaction pattern definitions, and
* the precursor adducts of the spectra that share the tree (they prune it).

Peaks, collision energies and other metadata do not change a tree, so a stored
tree serves any spectra of the same compound and precursor adducts; their
peaks are annotated again (no RDKit reaction is run).  A stored tree whose
patterns are a subset of the requested ones seeds the build, which then only
adds the remaining patterns.  A tree built without precursor pruning
(``complete``) serves any precursor adducts.

Layout::

    <root>/index.sqlite          one row per stored tree
    <root>/trees/<id>.pkl        pattern keys and (history, mapped SMILES) pairs
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import time
from typing import Iterable

STORE_SCHEMA = "fredis.fragment-tree-store"
STORE_VERSION = 1


def pattern_key(pattern) -> str:
    """Identity of a reaction pattern: its definition without the local id."""
    value = pattern.to_dict() if hasattr(pattern, "to_dict") else dict(pattern)
    value = {key: item for key, item in value.items() if key != "pattern_id"}
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def settings_key(fragmenter_params: dict, *, max_unique_fragment_smiles: int, max_reaction_combinations: int) -> str:
    """Everything except the patterns that changes a tree."""
    params = json.loads(json.dumps(fragmenter_params, default=str))
    params.get("fragment_ion_tree_builder", {}).pop("reaction_pattern_set", None)
    value = dict(params=params, max_unique_fragment_smiles=int(max_unique_fragment_smiles),
                 max_reaction_combinations=int(max_reaction_combinations))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


@dataclass(frozen=True)
class StoredTree:
    path: Path
    pattern_keys: frozenset[str]
    adducts: tuple[str, ...]
    complete: bool

    def histories(self, pattern_ids: dict[str, int]):
        """``{ReactionSequence: Compound}`` with pattern ids of the caller's library."""
        from fredis.libs.mmkit.mmkit import Compound
        from .reaction.ReactionSequence import ReactionSequence

        with self.path.open("rb") as stream:
            document = pickle.load(stream)
        mapping = {int(old): pattern_ids[key] for old, key in document["pattern_keys"].items()}
        renamed = {}

        def rename(reaction):
            if reaction not in renamed:
                renamed[reaction] = replace(reaction, reaction_pattern_id=mapping[int(reaction.reaction_pattern_id)])
            return renamed[reaction]

        return {
            ReactionSequence(tuple(rename(reaction) for reaction in sequence.reactions)): Compound.from_smiles(mapped)
            for sequence, mapped in document["histories"]
        }


class FragmentTreeStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        (self.root / "trees").mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS trees (id TEXT PRIMARY KEY, smiles TEXT, settings TEXT, "
                "patterns TEXT, pattern_keys TEXT, adducts TEXT, complete INTEGER, path TEXT, created REAL)")
            connection.execute("CREATE INDEX IF NOT EXISTS trees_lookup ON trees (smiles, settings)")
            connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            connection.execute("INSERT OR IGNORE INTO meta VALUES ('schema', ?)",
                               (json.dumps({"schema": STORE_SCHEMA, "version": STORE_VERSION}),))

    def _connect(self):
        connection = sqlite3.connect(self.root / "index.sqlite", timeout=120)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def find(self, smiles: str, settings: str, pattern_keys: Iterable[str], adducts: Iterable[str]) -> StoredTree | None:
        """The stored tree with the most patterns that are all requested, for these adducts."""
        wanted, adducts = frozenset(pattern_keys), tuple(sorted(set(adducts)))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT pattern_keys, adducts, complete, path FROM trees WHERE smiles = ? AND settings = ?",
                (smiles, settings)).fetchall()
        best = None
        for keys, stored_adducts, complete, path in rows:
            keys, stored_adducts = frozenset(json.loads(keys)), tuple(json.loads(stored_adducts))
            if not keys <= wanted or not (complete or stored_adducts == adducts):
                continue
            if not (self.root / path).is_file():
                continue
            if best is None or len(keys) > len(best.pattern_keys) or (
                    len(keys) == len(best.pattern_keys) and stored_adducts == adducts and not best.adducts == adducts):
                best = StoredTree(self.root / path, keys, stored_adducts, bool(complete))
        return best

    def save(self, smiles: str, settings: str, pattern_keys: dict[int, str], adducts: Iterable[str],
             complete: bool, histories) -> Path:
        """Store one complete (not truncated) tree; an identical entry is replaced."""
        adducts = tuple(sorted(set(adducts)))
        keys = sorted(set(pattern_keys.values()))
        identity = hashlib.sha256(json.dumps(
            [smiles, settings, _digest(keys), list(adducts), bool(complete)]).encode()).hexdigest()[:32]
        relative = f"trees/{identity}.pkl"
        target = self.root / relative
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            pickle.dump({"smiles": smiles, "pattern_keys": {int(k): v for k, v in pattern_keys.items()},
                         "adducts": adducts, "complete": bool(complete),
                         "histories": [(sequence, compound.mapped_smiles if hasattr(compound, "mapped_smiles") else compound)
                                       for sequence, compound in histories]},
                        stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(target)
        with self._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO trees VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (identity, smiles, settings, _digest(keys), json.dumps(keys), json.dumps(adducts),
                                int(bool(complete)), relative, time.time()))
        return target

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM trees").fetchone()[0])
