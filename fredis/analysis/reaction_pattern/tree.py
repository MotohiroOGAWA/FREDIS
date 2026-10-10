"""Tree Configuration: the analysis model shared by the CLI and Workbench.

A configuration keeps two independent lists::

    {
      "patternSets": [{"id": "A", "name": "A", "patterns": [<reaction pattern>, ...]}, ...],
      "nodes":       [{"id": "n1", "patternSet": "A", "parent": null},
                      {"id": "n2", "patternSet": "B", "parent": "n1"}, ...]
    }

A pattern set may be placed at any number of nodes.  A node's *condition* is
the union of the pattern sets on its path from its root, so ``A → B → C``
analyses ``A``, ``A + B`` and ``A + B + C``.  Every node with a parent is one
comparison edge (parent condition → parent condition + child set); a root has
only absolute metrics.

Patterns get stable integer ids for the whole analysis (``PatternRegistry``).
Ids appear inside saved Source reactions, so a later run that adds nodes to an
existing result must keep the ids of the patterns it already used.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterable

TREE_SCHEMA = "fredis.reaction-pattern-analysis.tree"
TREE_SCHEMA_VERSION = 1


def pattern_definition(pattern: dict) -> dict:
    """A pattern document without its analysis-local ``pattern_id``."""
    if not isinstance(pattern, dict):
        raise ValueError("A reaction pattern must be a JSON object.")
    definition = copy.deepcopy(pattern)
    definition.pop("pattern_id", None)
    if not str(definition.get("name") or "").strip():
        raise ValueError("Every reaction pattern needs a name.")
    if not str(definition.get("reactant_smarts") or "").strip():
        raise ValueError(f"Reaction pattern {definition['name']!r} has no reactant SMARTS.")
    return definition


def definition_key(definition: dict) -> str:
    return json.dumps(pattern_definition(definition), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PatternSetSpec:
    id: str
    name: str
    patterns: tuple[dict, ...]
    source: str = ""

    def to_dict(self) -> dict:
        value = {"id": self.id, "name": self.name, "patterns": [copy.deepcopy(item) for item in self.patterns]}
        if self.source:
            value["source"] = self.source
        return value


@dataclass(frozen=True)
class NodeSpec:
    id: str
    pattern_set: str
    parent: str | None = None

    def to_dict(self) -> dict:
        return {"id": self.id, "patternSet": self.pattern_set, "parent": self.parent}


@dataclass(frozen=True)
class TreeConfiguration:
    pattern_sets: tuple[PatternSetSpec, ...]
    nodes: tuple[NodeSpec, ...]

    # -- serialization --------------------------------------------------------
    @classmethod
    def from_dict(cls, value: dict | None) -> "TreeConfiguration":
        value = value or {}
        if "tree" in value and isinstance(value["tree"], dict) and "nodes" not in value:
            value = value["tree"]
        sets = []
        for row in value.get("patternSets") or ():
            if not isinstance(row, dict):
                raise ValueError("Every pattern set must be a JSON object.")
            set_id = str(row.get("id") or row.get("name") or "").strip()
            if not set_id:
                raise ValueError("Every pattern set needs an id.")
            sets.append(PatternSetSpec(set_id, str(row.get("name") or set_id),
                                       tuple(pattern_definition(item) for item in row.get("patterns") or ()),
                                       str(row.get("source") or "")))
        nodes = []
        for row in value.get("nodes") or ():
            if not isinstance(row, dict):
                raise ValueError("Every tree node must be a JSON object.")
            parent = row.get("parent")
            nodes.append(NodeSpec(str(row.get("id") or "").strip(), str(row.get("patternSet") or "").strip(),
                                  None if parent in (None, "") else str(parent)))
        tree = cls(tuple(sets), tuple(nodes))
        tree.validate()
        return tree

    def to_dict(self) -> dict:
        return {"schema": TREE_SCHEMA, "schemaVersion": TREE_SCHEMA_VERSION,
                "patternSets": [item.to_dict() for item in self.pattern_sets],
                "nodes": [item.to_dict() for item in self.nodes]}

    # -- structure ------------------------------------------------------------
    def validate(self) -> None:
        set_ids = [item.id for item in self.pattern_sets]
        duplicates = sorted({value for value in set_ids if set_ids.count(value) > 1})
        if duplicates:
            raise ValueError("Duplicate pattern set id: " + ", ".join(duplicates))
        for item in self.pattern_sets:
            names = [str(pattern["name"]) for pattern in item.patterns]
            repeated = sorted({value for value in names if names.count(value) > 1})
            if repeated:
                raise ValueError(f"Pattern set {item.name!r} repeats pattern names: {', '.join(repeated)}")
        node_ids = [item.id for item in self.nodes]
        if any(not value for value in node_ids):
            raise ValueError("Every tree node needs an id.")
        duplicates = sorted({value for value in node_ids if node_ids.count(value) > 1})
        if duplicates:
            raise ValueError("Duplicate tree node id: " + ", ".join(duplicates))
        known_sets, known_nodes = set(set_ids), set(node_ids)
        for node in self.nodes:
            if node.pattern_set not in known_sets:
                raise ValueError(f"Tree node {node.id!r} uses an unknown pattern set: {node.pattern_set!r}")
            if not self.pattern_set(node.pattern_set).patterns:
                raise ValueError(f"Pattern set {node.pattern_set!r} of node {node.id!r} has no reaction pattern.")
            if node.parent is not None and node.parent not in known_nodes:
                raise ValueError(f"Tree node {node.id!r} has an unknown parent: {node.parent!r}")
        for node in self.nodes:
            seen = {node.id}
            current = node
            while current.parent is not None:
                if current.parent in seen:
                    raise ValueError(f"The analysis tree has a cycle through node {node.id!r}.")
                seen.add(current.parent)
                current = self.node(current.parent)
        # The same pattern name must mean the same pattern everywhere, since
        # saved fragment paths and tables identify patterns by name.
        by_name: dict[str, str] = {}
        for item in self.pattern_sets:
            for pattern in item.patterns:
                key = definition_key(pattern)
                previous = by_name.setdefault(str(pattern["name"]), key)
                if previous != key:
                    raise ValueError(f"Different reaction patterns share the name {pattern['name']!r}.")

    def pattern_set(self, set_id: str) -> PatternSetSpec:
        for item in self.pattern_sets:
            if item.id == set_id:
                return item
        raise KeyError(f"Unknown pattern set: {set_id}")

    def node(self, node_id: str) -> NodeSpec:
        for item in self.nodes:
            if item.id == node_id:
                return item
        raise KeyError(f"Unknown tree node: {node_id}")

    @property
    def roots(self) -> tuple[NodeSpec, ...]:
        return tuple(item for item in self.nodes if item.parent is None)

    def children(self, node_id: str) -> tuple[NodeSpec, ...]:
        return tuple(item for item in self.nodes if item.parent == node_id)

    def path(self, node_id: str) -> tuple[NodeSpec, ...]:
        """Nodes from the root down to ``node_id``."""
        result = []
        current: NodeSpec | None = self.node(node_id)
        while current is not None:
            result.append(current)
            current = self.node(current.parent) if current.parent is not None else None
        return tuple(reversed(result))

    def depth(self, node_id: str) -> int:
        return len(self.path(node_id)) - 1

    def ordered_nodes(self) -> tuple[NodeSpec, ...]:
        """Depth-first, parents before children, keeping configuration order."""
        result: list[NodeSpec] = []

        def visit(node: NodeSpec) -> None:
            result.append(node)
            for child in self.children(node.id):
                visit(child)

        for root in self.roots:
            visit(root)
        return tuple(result)

    def used_pattern_sets(self) -> tuple[PatternSetSpec, ...]:
        used = {node.pattern_set for node in self.nodes}
        return tuple(item for item in self.pattern_sets if item.id in used)


class PatternRegistry:
    """Stable analysis-wide pattern ids, keyed by the pattern definition."""

    def __init__(self, previous: Iterable[dict] = ()):
        self._ids: dict[str, int] = {}
        self._definitions: dict[int, dict] = {}
        for row in previous:
            definition = pattern_definition(row.get("definition") or row)
            self._add(definition, int(row["pattern_id"]))

    def _add(self, definition: dict, pattern_id: int) -> int:
        key = definition_key(definition)
        if key in self._ids:
            return self._ids[key]
        if pattern_id in self._definitions:
            raise ValueError(f"Pattern id {pattern_id} is already used.")
        self._ids[key] = pattern_id
        self._definitions[pattern_id] = pattern_definition(definition)
        return pattern_id

    def id_for(self, definition: dict) -> int:
        key = definition_key(definition)
        if key not in self._ids:
            self._add(definition, max(self._definitions, default=-1) + 1)
        return self._ids[key]

    def definition(self, pattern_id: int) -> dict:
        return copy.deepcopy(self._definitions[int(pattern_id)])

    def items(self) -> list[tuple[int, dict]]:
        return [(pattern_id, copy.deepcopy(self._definitions[pattern_id])) for pattern_id in sorted(self._definitions)]

    def pattern_documents(self, pattern_ids: Iterable[int] | None = None) -> list[dict]:
        """Patterns with explicit ids, ready for ``ReactionPatternSet.from_dict``."""
        wanted = None if pattern_ids is None else {int(value) for value in pattern_ids}
        return [dict(definition, pattern_id=pattern_id) for pattern_id, definition in self.items()
                if wanted is None or pattern_id in wanted]


@dataclass
class Condition:
    """A cumulative pattern condition shared by every node that reaches it."""
    condition_id: int
    pattern_ids: frozenset[int]
    key: str
    label: str
    detail_directory: str


@dataclass
class ResolvedNode:
    node_id: str
    node_row: int
    parent_node_id: str | None
    parent_row: int
    pattern_set_id: str
    pattern_set_name: str
    depth: int
    condition_id: int
    parent_condition_id: int | None
    set_pattern_ids: tuple[int, ...]
    added_pattern_ids: tuple[int, ...]
    path_label: str
    condition_label: str


@dataclass
class ResolvedTree:
    """A validated configuration with pattern ids, conditions and node rows."""
    config: TreeConfiguration
    registry: PatternRegistry
    nodes: list[ResolvedNode]
    conditions: list[Condition]
    set_pattern_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def node(self, node_id: str) -> ResolvedNode:
        for item in self.nodes:
            if item.node_id == node_id:
                return item
        raise KeyError(node_id)

    def condition(self, condition_id: int) -> Condition:
        return self.conditions[int(condition_id)]

    @property
    def used_pattern_ids(self) -> list[int]:
        return sorted({pid for item in self.conditions for pid in item.pattern_ids})


def condition_key(registry: PatternRegistry, pattern_ids: Iterable[int]) -> str:
    """Content hash of a condition, independent of ids, names of sets and tree position."""
    documents = sorted(definition_key(registry.definition(pid)) for pid in pattern_ids)
    return hashlib.sha256("\n".join(documents).encode("utf-8")).hexdigest()[:16]


def resolve_tree(config: TreeConfiguration, registry: PatternRegistry | None = None) -> ResolvedTree:
    if not config.nodes:
        raise ValueError("Add at least one node to the analysis tree.")
    registry = registry or PatternRegistry()
    set_ids: dict[str, tuple[int, ...]] = {}
    for item in config.used_pattern_sets():
        set_ids[item.id] = tuple(dict.fromkeys(registry.id_for(pattern) for pattern in item.patterns))
    conditions: list[Condition] = []
    by_pattern_ids: dict[frozenset[int], int] = {}
    names = {item.id: item.name for item in config.pattern_sets}

    def condition_for(pattern_ids: frozenset[int], label: str) -> int:
        if pattern_ids not in by_pattern_ids:
            key = condition_key(registry, pattern_ids)
            by_pattern_ids[pattern_ids] = len(conditions)
            conditions.append(Condition(len(conditions), pattern_ids, key, label, f"details/conditions/{key}"))
        return by_pattern_ids[pattern_ids]

    ordered = config.ordered_nodes()
    rows = {node.id: index for index, node in enumerate(ordered)}
    nodes: list[ResolvedNode] = []
    accumulated: dict[str, frozenset[int]] = {}
    for node in ordered:
        own = set_ids[node.pattern_set]
        parent_ids = accumulated.get(node.parent, frozenset()) if node.parent else frozenset()
        current = parent_ids | frozenset(own)
        accumulated[node.id] = current
        path = config.path(node.id)
        set_names = list(dict.fromkeys(names[item.pattern_set] for item in path))
        label = " + ".join(set_names)
        condition_id = condition_for(current, label)
        parent_condition = condition_for(parent_ids, "") if node.parent else None
        nodes.append(ResolvedNode(
            node_id=node.id, node_row=rows[node.id], parent_node_id=node.parent,
            parent_row=rows[node.parent] if node.parent else -1, pattern_set_id=node.pattern_set,
            pattern_set_name=names[node.pattern_set], depth=len(path) - 1, condition_id=condition_id,
            parent_condition_id=parent_condition, set_pattern_ids=own,
            added_pattern_ids=tuple(pid for pid in own if pid not in parent_ids),
            path_label=" → ".join(names[item.pattern_set] for item in path), condition_label=label))
    return ResolvedTree(config, registry, nodes, conditions, set_ids)


def pattern_set_from_document(document: dict, *, source: str = "", set_id: str | None = None,
                              name: str | None = None) -> PatternSetSpec:
    """A pattern-set document, or a single pattern as a one-pattern set named after it."""
    if not isinstance(document, dict):
        raise ValueError(f"Expected a reaction pattern or pattern-set document: {source}")
    value = document.get("reaction_pattern_set") if isinstance(document.get("reaction_pattern_set"), dict) else None
    if value is not None:
        patterns = tuple(pattern_definition(item) for item in value.get("patterns") or ())
        default = str(value.get("name") or Path(source).name.split(".")[0] or "pattern_set")
    else:
        patterns = (pattern_definition(document),)
        default = str(document["name"])
    resolved = name or default
    return PatternSetSpec(set_id or resolved, resolved, patterns, source)


def load_pattern_set_file(path: str | Path, *, set_id: str | None = None, name: str | None = None) -> PatternSetSpec:
    path = Path(path)
    return pattern_set_from_document(json.loads(path.read_text()), source=str(path), set_id=set_id, name=name)


def load_tree_configuration(path: str | Path) -> TreeConfiguration:
    """A Tree Configuration file, or a run configuration that contains one."""
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict) and isinstance(value.get("config"), dict):
        value = value["config"]
    if isinstance(value, dict) and isinstance(value.get("tree"), dict):
        value = value["tree"]
    return TreeConfiguration.from_dict(value)
