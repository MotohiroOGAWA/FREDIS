"""Direct RDKit reaction inspection, with source-indexed matches (no tree)."""
from __future__ import annotations
from typing import Any
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolDescriptors, rdChemReactions, rdqueries
from functools import lru_cache
import json
from rdkit.Chem.Draw import rdMolDraw2D

from fredis.domain.fragment.reaction import ReactionSequence, ReactionPatternSet
from fredis.domain.fragment.reaction.ReactionInstanceGenerator import create_reaction_instances
from fredis.domain.fragment.tree import FragmentTreeBuilder
from fredis.libs.mmkit.mmkit import Compound

LIMIT = 512


def drawing(mol: Chem.Mol, width=760, height=500) -> dict[str, Any]:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    options.fixedBondLength = 45
    options.bondLineWidth = 2
    options.setAtomPalette({-1: (.84,.89,.94), 0: (.84,.89,.94), 6: (.84,.89,.94),
                            7: (.35,.7,1), 8: (1,.42,.48), 9: (.4,.85,.57),
                            16: (.95,.8,.3), 17: (.4,.85,.57)})
    drawer.DrawMolecule(mol)
    positions = [{"index": i, "x": drawer.GetDrawCoords(i).x, "y": drawer.GetDrawCoords(i).y}
                 for i in range(mol.GetNumAtoms())]
    drawer.FinishDrawing()
    return {"svg": drawer.GetDrawingText().replace("svg:", "").replace("#D6E2EF", "var(--vscode-editor-foreground,#d6e2ef)"), "positions": positions,
            "bonds": [{"index": b.GetIdx(), "begin": b.GetBeginAtomIdx(), "end": b.GetEndAtomIdx()}
                      for b in mol.GetBonds()]}


def source_molecule(payload):
    mol = Chem.MolFromSmiles(str(payload.get("smiles", "")).strip())
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Enter a valid, non-empty SMILES.")
    return mol


def _source_specific_smirks(source: Chem.Mol, products: tuple[Chem.Mol, ...]) -> str:
    """Serialize one generated result against a fully mapped Source molecule."""
    mapped_source = Chem.Mol(source)
    for atom in mapped_source.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    ordered_source = Chem.MolFromSmiles(Chem.MolToSmiles(mapped_source, canonical=True))
    reactant = Chem.MolToSmarts(ordered_source)

    # Product atoms created by a rule have no react_atom_idx. They are uncommon
    # for reaction rules, but still receive stable maps after all Source maps.
    created_maps = {}
    next_map = source.GetNumAtoms() + 1
    mapped_products = []
    for generated in products:
        product = Chem.Mol(generated)
        for atom in product.GetAtoms():
            if atom.HasProp("react_atom_idx"):
                atom.SetAtomMapNum(atom.GetIntProp("react_atom_idx") + 1)
                continue
            creation_key = atom.GetIntProp("old_mapno") if atom.HasProp("old_mapno") else (
                len(created_maps), atom.GetIdx()
            )
            if creation_key not in created_maps:
                created_maps[creation_key] = next_map
                next_map += 1
            atom.SetAtomMapNum(created_maps[creation_key])
        mapped_products.append(Chem.MolToSmiles(product, canonical=True))
    return f"({reactant})>>({'.'.join(sorted(mapped_products))})"


@lru_cache(maxsize=256)
def reactant_query(smarts):
    query = Chem.MolFromSmarts(smarts)
    if query is None or not query.GetNumAtoms():
        raise ValueError(f"Invalid reactant SMARTS: {smarts}")
    maps = [a.GetAtomMapNum() for a in query.GetAtoms()]
    if any(number <= 0 for number in maps) or len(set(maps)) != len(maps):
        raise ValueError("Reactant atoms must have unique positive atom maps.")
    return query


def matched_sites(mol, query, index=0):
    assignments = mol.GetSubstructMatches(query, uniquify=False, maxMatches=LIMIT + 1)
    sites = {}
    for assignment in assignments[:LIMIT]:
        bonds = sorted(mol.GetBondBetweenAtoms(assignment[b.GetBeginAtomIdx()], assignment[b.GetEndAtomIdx()]).GetIdx()
                       for b in query.GetBonds())
        key = (tuple(sorted(assignment)), tuple(bonds))
        if key not in sites:
            sites[key] = {"id": f"{index}:{len(sites)}", "atoms": list(key[0]), "bonds": bonds,
                          "products": [], "errors": [], "assignments": 0, "loaded": False,
                          "bindings": []}
        sites[key]["assignments"] += 1
        sites[key]["bindings"].append(assignment)
    return list(sites.values()), len(assignments) > LIMIT


def reaction_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Match reactant queries only; do not compile/run reactions or draw products."""
    mol = source_molecule(payload)
    definitions = payload.get("patterns", [])
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("Select at least one Reaction Pattern.")
    result = {"smiles": Chem.MolToSmiles(mol), "drawing": drawing(mol), "patterns": [], "truncated": False}
    for index, definition in enumerate(definitions):
        name = str(definition.get("name", "")) or f"Pattern {index + 1}"
        row = {"index": index, "name": name, "reactantSmarts": definition.get("reactant_smarts", ""),
               "productCount": len(definition.get("products", [])), "matches": [], "errors": []}
        result["patterns"].append(row)
        try:
            query = reactant_query(str(definition.get("reactant_smarts", "")))
            sites, truncated = matched_sites(mol, query, index)
            for site in sites:
                del site["bindings"]
            row["matches"] = sites
            result["truncated"] |= truncated
        except Exception as exc:
            row["errors"].append(str(exc))
    return result


@lru_cache(maxsize=128)
def compiled_pattern(definition_json):
    definition = json.loads(definition_json)
    reactant = str(definition.get("reactant_smarts", ""))
    reactant_query(reactant)
    products = {str(p.get("smarts", "")): p for p in definition.get("products", [])}
    reactions = []
    for smarts, product in sorted(products.items()):
        smirks = f"{reactant}>>{smarts}"
        rxn = rdChemReactions.ReactionFromSmarts(smirks)
        if rxn is None or rxn.GetNumReactantTemplates() != 1 or not rxn.GetNumProductTemplates():
            raise ValueError(f"Invalid single-reactant SMIRKS: {smirks}")
        maps = [atom.GetAtomMapNum() for template in rxn.GetProducts() for atom in template.GetAtoms()]
        if any(number <= 0 for number in maps) or len(set(maps)) != len(maps):
            raise ValueError("Product atoms must have unique positive atom maps.")
        reactions.append((str(product.get("name", "")) or "Product", smarts, smirks, rxn))
    return tuple(reactions)


@lru_cache(maxsize=512)
def product_drawing(smiles):
    return drawing(Chem.MolFromSmiles(smiles), 360, 220)


def reaction_preview_products(payload: dict[str, Any]) -> dict[str, Any]:
    """Run SMIRKS only at the selected physical site, including symmetric roles."""
    mol = source_molecule(payload)
    definition = payload.get("pattern", {})
    query = reactant_query(str(definition.get("reactant_smarts", "")))
    sites, truncated = matched_sites(mol, query)
    atoms, bonds = sorted(payload.get("atoms", [])), sorted(payload.get("bonds", []))
    site = next((s for s in sites if s["atoms"] == atoms and s["bonds"] == bonds), None)
    if site is None:
        raise ValueError("The selected site no longer matches the Reactant SMARTS.")
    result = {"products": [], "errors": [], "truncated": truncated}
    try:
        reactions = compiled_pattern(json.dumps(definition, sort_keys=True))
    except Exception as exc:
        result["errors"].append(str(exc))
        return result
    for atom in mol.GetAtoms():
        atom.SetIntProp("_fredis_preview_index", atom.GetIdx())
    seen = set()
    for name, smarts, _definition_smirks, template in reactions:
        for binding in site["bindings"]:
            # Bind each query role to its selected input atom. RunReactants then
            # yields this site's products, without enumerating every other site.
            rxn = rdChemReactions.ChemicalReaction(template)
            for atom in rxn.GetReactantTemplate(0).GetAtoms():
                atom.ExpandQuery(rdqueries.HasIntPropWithValueQueryAtom("_fredis_preview_index", binding[atom.GetIdx()]))
            rxn.Initialize(silent=True)
            try:
                groups = rxn.RunReactants((mol,), maxProducts=LIMIT + 1)
            except Exception as exc:
                result["errors"].append(f"{name}: {exc}")
                continue
            result["truncated"] |= len(groups) > LIMIT
            for group in groups[:LIMIT]:
                try:
                    site_smirks = _source_specific_smirks(mol, group)
                    components = []
                    for generated in group:
                        product = Chem.Mol(generated)
                        for atom in product.GetAtoms():
                            atom.SetAtomMapNum(0)
                        Chem.SanitizeMol(product)
                        for fragment in Chem.GetMolFrags(product, asMols=True):
                            components.append({"smiles": Chem.MolToSmiles(fragment),
                                               "formula": rdMolDescriptors.CalcMolFormula(fragment),
                                               "exactMass": rdMolDescriptors.CalcExactMolWt(fragment),
                                               "sourceAtoms": sorted({a.GetIntProp("react_atom_idx") for a in fragment.GetAtoms()
                                                                      if a.HasProp("react_atom_idx")})})
                    key = (smarts, tuple(sorted(c["smiles"] for c in components)))
                    if key in seen:
                        continue
                    for component in components:
                        component["drawing"] = product_drawing(component["smiles"])
                    seen.add(key)
                    result["products"].append({"name": name or "Product",
                                               "smarts": smarts, "smirks": site_smirks,
                                               "smiles": ".".join(c["smiles"] for c in components), "molecules": components})
                except Exception as exc:
                    message = f"{name or 'Product'}: {exc}"
                    if message not in result["errors"]:
                        result["errors"].append(message)
    return result


def _pattern_set(payload: dict[str, Any]) -> ReactionPatternSet:
    definitions = payload.get("patterns", [])
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("Select at least one Reaction Pattern.")
    return ReactionPatternSet.from_dict({"name": str(payload.get("name", "Preview")),
                                         "patterns": definitions})


def _molecule_summary(compound: Compound) -> dict[str, Any]:
    mol = compound.mol
    smiles = compound.smiles
    return {"smiles": smiles, "formula": rdMolDescriptors.CalcMolFormula(mol),
            "exactMass": rdMolDescriptors.CalcExactMolWt(mol),
            "drawing": product_drawing(smiles)}


def _reaction_summary(reaction, reaction_id: int, patterns: ReactionPatternSet,
                    map_to_index: dict[int, int]) -> dict[str, Any]:
    pattern = patterns.get_pattern(reaction.reaction_pattern_id)
    rule = next(r for r in pattern.reaction_rules if r.id == reaction.reaction_rule_id)
    name = rule.source_rule.name or f"Product {reaction.product_molecule_id + 1}"
    return {
        "id": reaction_id, "patternIndex": reaction.reaction_pattern_id,
        "patternName": pattern.name, "name": name,
        "productMoleculeId": reaction.product_molecule_id,
        "productMoleculeCount": len(rule.prod_temp),
        # Keep query-role order. Symmetric matches at the same physical site can
        # produce different reactions, so sorting would erase the visible distinction.
        "atoms": [map_to_index[m] for m in reaction.source_atom_maps],
        # Filled below after Source-map edges are converted to SVG bond indexes.
        "bonds": [],
        "sourceAtomMaps": list(reaction.source_atom_maps),
        "retainedAtomMaps": sorted(reaction.retained_atom_maps),
        "changedBondMaps": [list(edge) for edge in sorted(reaction.changed_bond_maps)],
        "reactionBondMaps": [list(edge) for edge in sorted(
            reaction.cut_bond_maps | frozenset((u, v) for u, v, _ in reaction.bond_updates))],
    }


def _run_sequence(source: Compound, sequence: ReactionSequence) -> Compound | None:
    try:
        products = sequence.compile(source).run(source)
        return Compound(products[0]) if len(products) == 1 else None
    except (Chem.rdchem.MolSanitizeException, ValueError):
        return None


def _sequence_summary(source: Compound, sequence: ReactionSequence) -> dict[str, Any] | None:
    try:
        composite = sequence.compile(source)
        products = composite.run(source)
        if len(products) != 1:
            return None
        result = _molecule_summary(Compound(products[0]))
        result["smirks"] = composite.smirks
        return result
    except (Chem.rdchem.MolSanitizeException, ValueError):
        return None


def reaction_explore(payload: dict[str, Any]) -> dict[str, Any]:
    """Explore Source-anchored primitive reactions and unordered reaction sets."""
    source = Compound.from_smiles(str(payload.get("smiles", "")).strip())
    patterns = _pattern_set(payload)
    raw_maximum = payload.get("maxReactionCount", 1)
    if isinstance(raw_maximum, bool) or not isinstance(raw_maximum, int):
        raise ValueError("Max reactions must be -1 or a positive integer.")
    reactions = create_reaction_instances(source, patterns)
    if raw_maximum == -1:
        # Every valid sequence is bounded by the finite primitive-reaction set.
        maximum = max(1, len(reactions))
    elif raw_maximum < 1:
        raise ValueError("Max reactions must be -1 or a positive integer.")
    else:
        maximum = raw_maximum
    mapped = source.mapped_mol
    map_to_index = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mapped.GetAtoms()}
    edge_to_bond = {tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(),
                                  bond.GetEndAtom().GetAtomMapNum()))): bond.GetIdx()
                    for bond in mapped.GetBonds()}
    summaries = []
    for reaction_id, reaction in enumerate(reactions):
        item = _reaction_summary(reaction, reaction_id, patterns, map_to_index)
        item["bonds"] = sorted(edge_to_bond[edge] for edge in reaction.matched_bond_maps)
        item["reactionBonds"] = sorted(edge_to_bond[tuple(edge)]
                                       for edge in item["reactionBondMaps"])
        summaries.append(item)

    requested_ids = tuple(int(value) for value in payload.get("selectedReactionIds", []))
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("The same reaction cannot be selected more than once.")
    chosen_ids = requested_ids
    if any(value < 0 or value >= len(reactions) for value in chosen_ids):
        raise ValueError("A selected reaction no longer exists. Run the viewer again.")
    selected = tuple(reactions[value] for value in chosen_ids)
    if len(selected) > maximum:
        raise ValueError("The selected reactions exceed Max reactions.")
    selected_sequence = ReactionSequence(selected) if selected else None
    if selected_sequence is not None and len(selected_sequence.reactions) != len(selected):
        raise ValueError("The selected reaction is redundant with the current reaction set.")

    available = []
    candidate_sequences = {}
    if len(selected) < maximum:
        for reaction_id, reaction in enumerate(reactions):
            if reaction_id in chosen_ids:
                continue
            try:
                sequence = ReactionSequence((*selected, reaction))
                if (len(sequence.reactions) == len(selected) + 1 and sequence.retained_atom_maps
                        and len(sequence.reactions) <= maximum):
                    if _run_sequence(source, sequence) is not None:
                        available.append(reaction_id)
                        candidate_sequences[reaction_id] = sequence
            except ValueError:
                pass

    selected_product = None
    if selected_sequence is not None:
        selected_product = _sequence_summary(source, selected_sequence)
        if selected_product is None:
            raise ValueError("The selected reaction set did not produce a valid fragment.")

    candidate_product = None
    inspect_id = payload.get("inspectReactionId")
    if inspect_id is not None:
        inspect_id = int(inspect_id)
        if inspect_id not in candidate_sequences:
            raise ValueError("The inspected reaction is not valid for the current reaction set.")
        candidate_product = _sequence_summary(source, candidate_sequences[inspect_id])

    result = {"smiles": source.smiles, "drawing": drawing(source.mol),
              "patterns": [{"index": pattern.pattern_id, "name": pattern.name}
                           for pattern in patterns],
              "reactions": summaries, "availableReactionIds": available,
              "selectedReactionIds": list(chosen_ids), "selectedProduct": selected_product,
              "candidateProduct": candidate_product, "inspectedReactionId": inspect_id,
              "maxReactionCount": maximum, "results": [], "truncated": False}
    if str(payload.get("mode", "stepwise")) == "exhaustive":
        builder = FragmentTreeBuilder(maximum, patterns)
        search, candidates = builder._search(source, seed_reaction_sequences=None,
                                             max_reaction_count=maximum)
        reaction_ids = {reaction: index for index, reaction in enumerate(reactions)}
        for _, item in builder._materialize_iter(source, search, candidates):
            ids = [reaction_ids[reaction] for reaction in item.reaction_sequence.reactions]
            result["results"].append({"reactionIds": ids, "reactionCount": len(ids),
                                      "smirks": item.smirks,
                                      "molecule": _molecule_summary(item.compound)})
    return result


def fragment_tree(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a Source-anchored FragmentTree from FragmentTreeBuilder settings only."""
    source = Compound.from_smiles(str(payload.get("smiles", "")).strip())
    max_reaction_count = payload.get("maxReactionCount")
    if isinstance(max_reaction_count, bool) or not isinstance(max_reaction_count, int):
        raise ValueError("Maximum reaction count must be an integer.")
    pattern_set = payload.get("reactionPatternSet")
    if not isinstance(pattern_set, dict):
        raise ValueError("A reaction pattern set is required.")
    builder = FragmentTreeBuilder.from_dict({
        "max_reaction_count": max_reaction_count,
        "reaction_pattern_set": pattern_set,
    })
    max_unique = payload.get("maxUniqueFragmentSmiles", -1)
    max_combinations = payload.get("maxReactionCombinations", -1)
    if isinstance(max_unique, bool) or not isinstance(max_unique, int):
        raise ValueError("Maximum unique fragment SMILES must be an integer.")
    if isinstance(max_combinations, bool) or not isinstance(max_combinations, int):
        raise ValueError("Maximum reaction combinations must be an integer.")

    built = builder._build_result(
        source,
        max_unique_fragment_smiles=max_unique,
        max_reaction_combinations=max_combinations,
        print_info=False,
    )
    tree = built["fragment_tree"]
    reactions = tuple(sorted(set(builder.create_reaction_instances(source)), key=lambda reaction: reaction.key))
    reaction_ids = {reaction: index for index, reaction in enumerate(reactions)}
    patterns = builder.reaction_pattern_set
    mapped = source.mapped_mol
    map_to_index = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mapped.GetAtoms()}
    edge_to_bond = {
        tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()))): bond.GetIdx()
        for bond in mapped.GetBonds()
    }

    reaction_rows = []
    for reaction_id, reaction in enumerate(reactions):
        row = _reaction_summary(reaction, reaction_id, patterns, map_to_index)
        pattern = patterns.get_pattern(reaction.reaction_pattern_id)
        rule = next(item for item in pattern.reaction_rules if item.id == reaction.reaction_rule_id)
        row.update({
            "reactantSmarts": pattern.reactant_smarts,
            "reactionRuleId": reaction.reaction_rule_id,
            "reactionRuleName": rule.source_rule.name or row["name"],
            "cutBondMaps": [list(edge) for edge in sorted(reaction.cut_bond_maps)],
            "discardedAtomMaps": sorted(reaction.discarded_atom_maps),
            "bonds": sorted(edge_to_bond[edge] for edge in reaction.matched_bond_maps),
        })
        reaction_rows.append(row)

    depths = tree.node_depths.tolist()
    minimum_reaction_counts = tree.get_min_reaction_counts()
    nodes = []
    for index in range(tree.num_nodes):
        node = tree.get_node(index)
        mol = Chem.MolFromSmiles(node.smiles)
        nodes.append({
            "id": index,
            "smiles": node.smiles,
            "depth": int(depths[index]),
            "minimumReactionCount": minimum_reaction_counts.get(index),
            "formula": rdMolDescriptors.CalcMolFormula(mol) if mol is not None else "",
            "exactMass": rdMolDescriptors.CalcExactMolWt(mol) if mol is not None else None,
            "precursor": index == 0,
        })

    def sequence_ids(sequence):
        return [] if sequence is None else [reaction_ids[reaction] for reaction in sequence.reactions]

    edges = []
    for index in range(tree.num_edges):
        edge = tree.get_edge(index)
        edges.append({
            "id": index,
            "source": edge.source_index,
            "target": edge.target_index,
            "transitions": [{
                "addedReaction": None if transition.added_reaction is None else reaction_ids[transition.added_reaction],
                "parentReactions": sequence_ids(transition.parent_reaction_sequence),
                "targetReactions": sequence_ids(transition.reaction_sequence),
                "seed": transition.is_seed,
            } for transition in edge.transitions],
        })

    stats = built["search_stats"]
    return {
        "smiles": source.smiles,
        "drawing": drawing(source.mol),
        "nodes": nodes,
        "edges": edges,
        "reactions": reaction_rows,
        "stats": stats,
        "maxReactionCount": builder.max_reaction_count,
        "patternSetName": patterns.name,
    }
