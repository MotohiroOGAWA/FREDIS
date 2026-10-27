"""Inspect primitive reaction templates and match only the original Source."""

from __future__ import annotations

from dataclasses import dataclass
from rdkit import Chem

from fredis.libs.mmkit.mmkit import Compound
from .ReactionInstance import ReactionInstance, bond_key, mapped_source
from .ReactionPatternSet import ReactionPattern, ReactionPatternSet


@dataclass(frozen=True)
class _ReactionTemplate:
    reaction_pattern_id: int
    reaction_rule_id: int
    product_molecule_id: int
    query_bond_roles: tuple[tuple[int, int], ...]
    cut_bond_roles: tuple[tuple[int, int], ...]
    product_atom_roles: frozenset[int]
    product_bond_roles: tuple[tuple[int, int, Chem.BondType], ...]
    product_charges: tuple[tuple[int, int], ...]
    product_elements: tuple[tuple[int, int], ...]


def _reaction_templates(pattern: ReactionPattern) -> tuple[_ReactionTemplate, ...]:
    """Parse each rule once, before binding any Source matches.

    Unsupported templates are omitted, including created atoms/bonds and
    explicitly requested charge/radical changes. Unchanged aromatic bonds are
    permitted; aromatic bond order changes are not.
    """
    query = pattern.reactant_query
    role_by_map = {atom.GetAtomMapNum(): atom.GetIdx() for atom in query.GetAtoms()}
    query_edges = tuple(
        sorted(
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in query.GetBonds()
        )
    )
    templates: list[_ReactionTemplate] = []
    for rule in pattern.reaction_rules:
        product_data: list[
            tuple[
                frozenset[int],
                tuple[tuple[int, int, Chem.BondType], ...],
                tuple[tuple[int, int], ...],
                tuple[tuple[int, int], ...],
            ]
        ] = []
        all_edges: set[tuple[int, int]] = set()
        all_roles: set[int] = set()
        unsupported = False
        for product in rule.prod_temp:
            roles: set[int] = set()
            bonds: list[tuple[int, int, Chem.BondType]] = []
            charges: list[tuple[int, int]] = []
            elements: list[tuple[int, int]] = []
            for atom in product.GetAtoms():
                role = role_by_map.get(atom.GetAtomMapNum())
                if role is None or role in all_roles:
                    unsupported = True
                    break
                roles.add(role)
                all_roles.add(role)
                if "AtomFormalCharge" in atom.DescribeQuery():
                    charges.append((role, atom.GetFormalCharge()))
                if atom.GetAtomicNum():
                    elements.append((role, atom.GetAtomicNum()))
                if atom.GetNumRadicalElectrons():
                    unsupported = True
            if unsupported:
                break
            for bond in product.GetBonds():
                u = role_by_map[bond.GetBeginAtom().GetAtomMapNum()]
                v = role_by_map[bond.GetEndAtom().GetAtomMapNum()]
                edge = tuple(sorted((u, v)))
                if edge not in query_edges:
                    unsupported = True
                    break
                bonds.append((*edge, bond.GetBondType()))
                all_edges.add(edge)
            product_data.append(
                (
                    frozenset(roles),
                    tuple(sorted(bonds)),
                    tuple(sorted(charges)),
                    tuple(sorted(elements)),
                )
            )
        if unsupported:
            continue
        cuts = tuple(edge for edge in query_edges if edge not in all_edges)
        for product_id, (anchors, bonds, charges, elements) in enumerate(product_data):
            if anchors:
                templates.append(
                    _ReactionTemplate(
                        pattern.pattern_id,
                        rule.id,
                        product_id,
                        query_edges,
                        cuts,
                        anchors,
                        bonds,
                        charges,
                        elements,
                    )
                )
    return tuple(templates)


def _retained_component(
    anchors: frozenset[int],
    adjacency: dict[int, frozenset[int]],
    cuts: frozenset[tuple[int, int]],
) -> frozenset[int] | None:
    reached: set[int] = set()
    pending = [min(anchors)]
    while pending:
        atom = pending.pop()
        if atom in reached:
            continue
        reached.add(atom)
        pending.extend(
            other
            for other in adjacency[atom]
            if other not in reached and bond_key(atom, other) not in cuts
        )
    return frozenset(reached) if anchors <= reached else None


def create_reaction_instances(
    source_compound: Compound,
    reaction_pattern_set: ReactionPatternSet,
) -> tuple[ReactionInstance, ...]:
    """Enumerate all supported pattern × match × reaction × product reactions.

    GetSubstructMatches is called exactly once per pattern (uniquify=False
    preserves query-role assignments). No reactions or per-reaction matching run.
    Retention is the connected Source component containing all selected product
    anchors after removing cuts from the entire reaction.
    """
    mol = mapped_source(source_compound)
    maps = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
    universe = frozenset(maps)
    source_orders = {
        bond_key(
            b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
        ): b.GetBondType()
        for b in mol.GetBonds()
    }
    adjacency = {
        atom.GetAtomMapNum(): frozenset(n.GetAtomMapNum() for n in atom.GetNeighbors())
        for atom in mol.GetAtoms()
    }
    charges = {atom.GetAtomMapNum(): atom.GetFormalCharge() for atom in mol.GetAtoms()}
    elements = {atom.GetAtomMapNum(): atom.GetAtomicNum() for atom in mol.GetAtoms()}
    reactions: set[ReactionInstance] = set()
    for pattern in reaction_pattern_set:
        templates = _reaction_templates(pattern)
        matches = mol.GetSubstructMatches(
            pattern.reactant_query, uniquify=False, maxMatches=0
        )
        for match in matches:
            roles = tuple(maps[index] for index in match)
            components: dict[frozenset[tuple[int, int]], dict[int, frozenset[int]]] = {}
            for template in templates:
                cuts = frozenset(
                    bond_key(roles[u], roles[v]) for u, v in template.cut_bond_roles
                )
                anchors = frozenset(roles[role] for role in template.product_atom_roles)
                # Reuse connected components for all products in this reaction/match.
                cached = components.setdefault(cuts, {})
                retained = cached.get(min(anchors))
                if retained is None:
                    retained = _retained_component(anchors, adjacency, cuts)
                    if retained is None:
                        continue
                    for atom in retained:
                        cached[atom] = retained
                if not anchors <= retained:
                    continue
                if any(
                    charges[roles[role]] != charge
                    for role, charge in template.product_charges
                ):
                    continue
                if any(
                    elements[roles[role]] != number
                    for role, number in template.product_elements
                ):
                    continue
                updates: set[tuple[int, int, Chem.BondType]] = set()
                unsupported = False
                for u, v, order in template.product_bond_roles:
                    edge = bond_key(roles[u], roles[v])
                    if (
                        order == Chem.BondType.UNSPECIFIED
                        or source_orders[edge] == order
                    ):
                        continue
                    if order not in (
                        Chem.BondType.SINGLE,
                        Chem.BondType.DOUBLE,
                        Chem.BondType.TRIPLE,
                    ):
                        unsupported = True
                        break
                    if source_orders[edge] == Chem.BondType.AROMATIC:
                        unsupported = True
                        break
                    updates.add((*edge, order))
                if unsupported:
                    continue
                if not retained or (retained == universe and not cuts and not updates):
                    continue  # Deterministic empty/true-no-op hard pruning.
                matched = frozenset(
                    bond_key(roles[u], roles[v]) for u, v in template.query_bond_roles
                )
                changed = frozenset(
                    edge for edge in source_orders if not set(edge) <= retained
                )
                reactions.add(
                    ReactionInstance(
                        template.reaction_pattern_id,
                        template.reaction_rule_id,
                        roles,
                        retained,
                        universe - retained,
                        matched,
                        cuts,
                        frozenset(updates),
                        changed,
                        product_molecule_id=template.product_molecule_id,
                    )
                )
    return tuple(sorted(reactions, key=lambda reaction: reaction.key))
