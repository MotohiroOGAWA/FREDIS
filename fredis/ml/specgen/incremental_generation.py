"""Model-guided expansion of Source reactions, without exhaustive preparation.

Saved training tensors are chemistry caches, never a teacher-selected search
space. Only the children of surviving states are materialized and scored.
"""

from __future__ import annotations

from types import SimpleNamespace
import math

import torch
from torch_geometric.data import Batch
from rdkit import Chem

from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.domain.fragment.reaction import ReactionInstance, ReactionSequence
from fredis.domain.fragment.reaction.ReactionInstance import bond_key
from fredis.domain.fragment.reaction.ReactionInstanceGenerator import (
    _reaction_templates,
)
from fredis.domain.fragment.reaction.ReactionRelations import ReactionRelations
from ..input.source_reaction_structure import (
    SpectrumPredictionData,
    SpectrumConditions,
    csr,
    rows,
    tensor,
)
from ..input.source_compound import SourceCompound
from ..input.reaction_tensor_data import ReactionTensorData
from ..input.reaction_set import ReactionSet
from ..input.branch_group import BranchGroup
from ..input.adduct_data import AdductData
from ..input.compound_batch import CompoundBatch
from ..input.compound_reaction_graph import CompoundReactionGraph
from ..input.reaction_context_data import ReactionContextData
from ..input.annotated_spectrum_data import (
    FormulaData,
    FormulaExplanationData,
    SpectrumExplanationData,
)


def _saved_registry(generator, data, source_index):
    """Restore concrete edits from saved matches/retention, without SMARTS matching.

    Template parsing supplies cut/order edits omitted from the tensor schema;
    source match order, retained atoms and pairwise relations come from disk.
    """
    builder = generator.mol_encoder.graph_builder
    source_cid = int(data.source_compounds.compound_index[source_index])
    source = data.compounds.compounds[source_cid]
    atom_maps = [a.GetAtomMapNum() for a in builder.graph_atoms(source)]
    graph = data.source_graph.to_data_list()[source_index]
    if (
        data.source_compounds.symbols != builder.symbols
        or len(atom_maps) != graph.num_nodes
    ):
        raise ValueError("Prepared source atom vocabulary differs from this model")
    offset = int(data.source_graph.ptr[source_index])
    registry = data.reactions.to("cpu")
    ids = (
        torch.nonzero(registry.reaction_source_compound_index == source_index)
        .flatten()
        .tolist()
    )
    inverse = {old: new for new, old in enumerate(ids)}
    templates = {
        (t.reaction_pattern_id, t.reaction_rule_id, t.product_molecule_id): t
        for p in generator.fragmenter.fragment_ion_tree_builder.reaction_patterns
        for t in _reaction_templates(p)
    }
    mol = source.mapped_mol
    universe = frozenset(a.GetAtomMapNum() for a in mol.GetAtoms())
    orders = {
        bond_key(
            b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
        ): b.GetBondType()
        for b in mol.GetBonds()
    }
    restored = []
    for old in ids:
        category = tuple(registry.reaction_type[old].tolist())
        template = templates.get(category)
        if template is None:
            raise ValueError(
                f"Prepared reaction {category} is absent from this model's chemistry"
            )
        a, b = registry.reaction_source_atom_ptr[old : old + 2].tolist()
        roles = tuple(
            atom_maps[i - offset]
            for i in registry.reaction_source_atom_index[a:b].tolist()
        )
        a, b = registry.reaction_retained_atom_ptr[old : old + 2].tolist()
        retained = frozenset(
            atom_maps[i - offset]
            for i in registry.reaction_retained_atom_index[a:b].tolist()
        )
        cuts = frozenset(
            bond_key(roles[u], roles[v]) for u, v in template.cut_bond_roles
        )
        matched = frozenset(
            bond_key(roles[u], roles[v]) for u, v in template.query_bond_roles
        )
        updates = frozenset(
            (*bond_key(roles[u], roles[v]), order)
            for u, v, order in template.product_bond_roles
            if order != Chem.BondType.UNSPECIFIED
            and orders[bond_key(roles[u], roles[v])] != order
        )
        changed = frozenset(edge for edge in orders if not set(edge) <= retained)
        restored.append(
            ReactionInstance(
                category[0],
                category[1],
                roles,
                retained,
                universe - retained,
                matched,
                cuts,
                updates,
                changed,
                product_molecule_id=category[2],
            )
        )

    def pairs(name):
        return tuple(
            (inverse[a], inverse[b])
            for a, b in getattr(registry, name).T.tolist()
            if a in inverse and b in inverse
        )

    conflicts = pairs("reaction_conflict_index")
    invalidations = pairs("reaction_invalidation_index")

    def masks(pairs):
        values = [0] * len(ids)
        for a, b in pairs:
            values[a] |= 1 << b
        return tuple(values)

    relations = ReactionRelations(
        masks(conflicts), masks(invalidations), conflicts, invalidations
    )
    reactions = tuple(restored)
    maps = {v: i for i, v in enumerate(atom_maps)}
    # Preserve exactly the saved row order, including the query-role order.
    packed = ReactionTensorData.from_reactions(reactions, maps, relations)
    saved_sets = data.reaction_sets.rows()
    cached = {}
    for cid, set_ids in enumerate(
        rows(
            data.compounds.generating_reaction_sets_ptr,
            data.compounds.generating_reaction_sets_index,
        )
    ):
        for sid in set_ids:
            members = saved_sets[sid]
            if members and all(r in inverse for r in members):
                seq = ReactionSequence(reactions[inverse[r]] for r in members)
                cached[seq.effect_key] = cid
    return source, graph, reactions, relations, packed, cached


class _Chemistry:
    def __init__(
        self, generator, source, *, prepared=None, source_index=0, precomputed=None
    ):
        self.generator = generator
        self.saved = prepared
        builder = generator.mol_encoder.graph_builder
        if prepared is not None:
            source, graph, reactions, relations, packed, saved_effects = (
                _saved_registry(generator, prepared, source_index)
            )
            if not prepared.adducts.same_vocabulary(
                AdductData.from_fragmenter(generator.fragmenter)
            ):
                raise ValueError("Prepared adduct vocabulary differs from this model")
            if tuple(prepared.formula_elements) != tuple(
                generator.tensorizer.element_order
            ):
                raise ValueError("Prepared formula elements differ from this model")
        else:
            reactions = (
                precomputed["reactions"]
                if precomputed is not None
                else generator.fragmenter.fragment_ion_tree_builder.create_reaction_instances(
                    source
                )
            )
            relations = (
                precomputed["relations"]
                if precomputed is not None
                else ReactionRelations.from_reactions(reactions)
            )
            graph = builder.build(source)
            maps = {
                a.GetAtomMapNum(): i for i, a in enumerate(builder.graph_atoms(source))
            }
            packed = ReactionTensorData.from_reactions(reactions, maps, relations)
            saved_effects = {}
        self.source = source
        self.source_graph = Batch.from_data_list([graph])
        self.reactions = tuple(reactions)
        self.relations = relations
        self.registry = packed
        self.compounds = [source]
        self.smiles = [
            (
                prepared.source_smiles[source_index]
                if prepared is not None
                else source.smiles
            )
        ]
        self.graphs = [graph]
        self.by_smiles = {self.smiles[0]: 0}
        self.effects = {}
        self.saved_effects = saved_effects
        self.saved_graphs = None
        self.embeddings = {}
        self.saved_formula = {}
        self.runs = 0
        if prepared is not None:
            self.saved_graphs = prepared.compounds.graph.to_data_list()
            e = prepared.formula_explanations
            for f, c, ion, shift in zip(
                e.formula_index.tolist(),
                e.compound_index.tolist(),
                e.fragment_ion_adduct_index.tolist(),
                e.hydrogen_shift_index.tolist(),
            ):
                key = (
                    prepared.compounds.smiles[c],
                    prepared.adducts.fragment_ion_adducts[ion],
                    int(prepared.adducts.hydrogen_shifts[shift]),
                )
                self.saved_formula[key] = (
                    tuple(prepared.formulas.composition[f].tolist()),
                    float(prepared.formulas.mz[f]),
                )
        device = next(generator.parameters()).device
        static_data = SimpleNamespace(source_graph=self.source_graph, reactions=packed)
        self.static = generator.feature_model.encode_static(
            static_data, max_graphs=generator.max_samples
        )
        self.slot_lookup = generator.feature_model.reaction_slot_lookup(
            self.static[3], len(self.reactions)
        )
        self.embeddings[0] = self.static[1][0]
        self.device = device

    def materialize(self, members):
        seq = ReactionSequence(self.reactions[r] for r in members)
        effect = seq.effect_key
        if effect in self.effects:
            return self.effects[effect]
        saved = self.saved_effects.get(effect)
        if saved is not None:
            compound = self.saved.compounds.compounds[saved]
            smiles = self.saved.compounds.smiles[saved]
            graph = self.saved_graphs[saved]
        else:
            try:
                self.runs += 1
                products = seq.compile(self.source).run(self.source)
                if len(products) != 1:
                    raise ValueError("Source reaction must materialize one molecule")
                compound = Compound(products[0])
            except Chem.rdchem.MolSanitizeException:
                self.effects[effect] = None
                return None
            smiles = compound.smiles
            graph = None
        cid = self.by_smiles.get(smiles)
        if cid is None:
            cid = len(self.compounds)
            self.by_smiles[smiles] = cid
            self.compounds.append(compound)
            self.smiles.append(smiles)
            self.graphs.append(
                graph
                if graph is not None
                else self.generator.mol_encoder.graph_builder.build(compound)
            )
        self.effects[effect] = cid
        return cid

    def encode(self, ids):
        missing = sorted(set(ids) - self.embeddings.keys())
        for start in range(0, len(missing), self.generator.max_samples):
            chunk = missing[start : start + self.generator.max_samples]
            h = self.generator.mol_encoder(
                Batch.from_data_list([self.graphs[i] for i in chunk]).to(self.device)
            )
            self.embeddings.update(zip(chunk, h))

    def search(self, main_adduct):
        """Prune the lowest path probabilities after each depth; keep expanding.

        The budget counts unique structures including Source, per main adduct.
        Ties retain shallower structures so parents survive with their children.
        Distinct reaction histories of a shared structure remain separate states.
        """
        model = self.generator.feature_model
        condition = model.main_adduct_projection(
            model.adduct_encoder(tensor([main_adduct], self.device))
        )[0]
        adduct_token = model.decoder_adduct_projection(condition[None])
        root_token = model.decoder_root_projection(self.static[1])
        states = {(): (0, 0.0)}
        frontier = [()]
        edges = set()
        settings = self.generator.generation
        threshold = (
            math.log(settings.branch_path_threshold)
            if settings.branch_path_threshold
            else -math.inf
        )
        for depth in range(1, self.generator.fragmenter.tree_max_reaction_count + 1):
            candidates = []
            for parent in frontier:
                parent_cid, parent_score = states[parent]
                if parent_score < threshold:
                    continue
                for r in range(len(self.reactions)):
                    if r in parent:
                        continue
                    members = tuple(sorted((*parent, r)))
                    if self.relations.rejection(members):
                        continue
                    child = self.materialize(members)
                    if child is None or child == parent_cid:
                        continue
                    candidates.append((parent, members, r, child))
            if not candidates:
                break
            parents = list(dict.fromkeys(p for p, _, _, _ in candidates))
            parent_lookup = {parent: i for i, parent in enumerate(parents)}
            hidden_chunks = []
            # Decode each unique parent once, even if it has many candidates.
            for start in range(0, len(parents), self.generator.max_samples):
                chunk_parents = parents[start:start + self.generator.max_samples]
                width = max(map(len, chunk_parents), default=0)
                states_tensor = torch.tensor(
                    [list(parent) + [-1] * (width - len(parent)) for parent in chunk_parents],
                    dtype=torch.long, device=self.device,
                )
                hidden_chunks.append(model.decode_sets(self.static[2], self.static[3], states_tensor,
                    torch.zeros(len(chunk_parents), dtype=torch.long, device=self.device),
                    adduct_token.expand(len(chunk_parents), -1),
                    root_token.expand(len(chunk_parents), -1), slot_lookup=self.slot_lookup))
            hidden = torch.cat(hidden_chunks)
            queries = model.branch_query(torch.cat((hidden, condition.expand(len(parents), -1)), 1))
            log_scores = []
            # Bound candidate projections independently of chemistry candidate count.
            for start in range(0, len(candidates), self.generator.max_samples):
                chunk = candidates[start : start + self.generator.max_samples]
                parent_rows = tensor([parent_lookup[p] for p, _, _, _ in chunk], self.device)
                keys = model.branch_key(self.static[0][tensor([r for _, _, r, _ in chunk], self.device)])
                logits = (queries[parent_rows] * keys).sum(1) * model.scale
                log_scores.extend(torch.nn.functional.logsigmoid(logits).cpu().tolist())
            new_states = set()
            for (parent, members, r, child), log_score in zip(candidates, log_scores):
                score = states[parent][1] + log_score
                if score < threshold:
                    continue
                if members not in states or score > states[members][1]:
                    states[members] = (child, score)
                new_states.add(members)
                edges.add((parent, members, r))
            states, edges = _prune(states, edges, settings.max_fragment_nodes)
            frontier = sorted(new_states & states.keys())
            if not frontier:
                break
        return states, edges


def _prune(states, edges, budget):
    best = {}
    for members, (cid, score) in states.items():
        rank = (-score, len(members), cid)
        if cid not in best or rank < best[cid]:
            best[cid] = rank
    keep = set(sorted(best, key=best.get)[:budget])
    keep.add(0)  # The root is structural, and always has probability one.
    selected = {members: value for members, value in states.items() if value[0] in keep}
    edges = {edge for edge in edges if edge[0] in selected and edge[1] in selected}
    reachable = {()}
    # Reaction count increases on every edge, so one depth-ordered pass suffices.
    for parent, child, _ in sorted(edges, key=lambda e: (len(e[0]), e)):
        if parent in reachable:
            reachable.add(child)
    return (
        {k: v for k, v in selected.items() if k in reachable},
        {e for e in edges if e[0] in reachable and e[1] in reachable},
    )


def _prediction_data(chemistry, precursor_types, energies):
    gen = chemistry.generator
    main = [str(gen.fragmenter._resolve_main_adduct_type(a)) for a in precursor_types]
    group_names = list(dict.fromkeys(main))
    group_ids = [group_names.index(a) for a in main]
    adducts = AdductData.from_fragmenter(gen.fragmenter)
    searches = [chemistry.search(adducts.main_adducts.index(a)) for a in group_names]
    ids = sorted({cid for states, _ in searches for cid, _ in states.values()})
    inverse = {old: new for new, old in enumerate(ids)}
    memberships = sorted(
        {members for states, _ in searches for members in states},
        key=lambda v: (len(v), v),
    )
    sets, set_ids = ReactionSet.from_rows(memberships)
    set_lookup = dict(zip(memberships, set_ids.tolist()))
    producing = [set() for _ in ids]
    keys = set()
    group_edges = []
    for states, edges in searches:
        for members, (cid, _) in states.items():
            producing[inverse[cid]].add(set_lookup[members])
        selected = set()
        for parent, child, r in edges:
            key = (inverse[states[parent][0]], inverse[states[child][0]], r)
            keys.add(key)
            selected.add(key)
        group_edges.append(selected)
    keys = sorted(keys)
    edge_lookup = {key: i for i, key in enumerate(keys)}
    ep, ei = csr([sorted(edge_lookup[e] for e in selected) for selected in group_edges])
    # Reuse saved formulas and ion annotations whenever a generated compound is known.
    compositions, masses, formula_lookup, explanations, explanation_lookup = (
        [],
        [],
        {},
        [],
        {},
    )
    sample_explanations = []
    for sample, group in enumerate(group_ids):
        selected = []
        main_id = adducts.main_adducts.index(group_names[group])
        a, b = adducts.fragment_ion_adduct_ptr[main_id : main_id + 2].tolist()
        for cid in sorted({cid for cid, _ in searches[group][0].values()}):
            for ion_id in adducts.fragment_ion_adduct_index[a:b].tolist():
                ion_str = adducts.fragment_ion_adducts[ion_id]
                for shift_id, shift in enumerate(adducts.hydrogen_shifts.tolist()):
                    key = (chemistry.smiles[cid], ion_str, shift)
                    cached = chemistry.saved_formula.get(key)
                    if cached is None:
                        ion = Adduct.parse(ion_str)
                        formula = ion.apply_to_formula(
                            Adduct.from_dict({"H": shift}).apply_to_formula(
                                chemistry.compounds[cid].formula
                            )
                        ).normalized
                        comp = tuple(
                            gen.tensorizer.formulas_to_tensor([formula])[0]
                            .long()
                            .tolist()
                        )
                        cached = (comp, formula.exact_mass / abs(ion.charge))
                        chemistry.saved_formula[key] = cached
                    comp, mz = cached
                    if comp not in formula_lookup:
                        formula_lookup[comp] = len(compositions)
                        compositions.append(comp)
                        masses.append(mz)
                    explanation = (formula_lookup[comp], inverse[cid], ion_id, shift_id)
                    if explanation not in explanation_lookup:
                        explanation_lookup[explanation] = len(explanations)
                        explanations.append(explanation)
                    selected.append(explanation_lookup[explanation])
        sample_explanations.append(selected)
    compounds = CompoundBatch(
        tuple(chemistry.compounds[i] for i in ids),
        Batch.from_data_list([chemistry.graphs[i] for i in ids]),
        *csr([sorted(v) for v in producing]),
    )
    compounds.__dict__["smiles"] = tuple(chemistry.smiles[i] for i in ids)
    composition = tensor(compositions).reshape(-1, gen.tensorizer.dim)
    source = SourceCompound(
        chemistry.source_graph,
        (chemistry.smiles[0],),
        gen.mol_encoder.graph_builder.symbols,
        tensor([inverse[0]]),
    )
    data = SpectrumPredictionData(
        source_compounds=source,
        reactions=chemistry.registry,
        reaction_sets=sets,
        adducts=adducts,
        branch_groups=BranchGroup(
            tensor([0] * len(group_names)),
            tensor([adducts.main_adducts.index(a) for a in group_names]),
        ),
        compounds=compounds,
        compound_graph=CompoundReactionGraph(*tensor(keys).reshape(-1, 3).unbind(1)),
        reaction_contexts=ReactionContextData(tensor([0] * len(group_names)), ep, ei),
        formulas=FormulaData(
            composition,
            torch.tensor(masses, dtype=torch.float64),
            (composition[:, :-1] >= 0).all(1),
        ),
        formula_explanations=FormulaExplanationData(
            *tensor(explanations).reshape(-1, 4).unbind(1)
        ),
        spectrum_explanations=SpectrumExplanationData(*csr(sample_explanations)),
        conditions=SpectrumConditions(
            tensor(group_ids),
            torch.tensor(energies, dtype=torch.float32),
            tuple(str(a) for a in precursor_types),
        ),
        formula_elements=tuple(gen.tensorizer.element_order),
        max_reaction_role_count=max(
            (len(r.source_atom_maps) for r in chemistry.reactions), default=0
        ),
        source_atom_capacity=chemistry.source_graph.num_nodes,
    )
    return data


def source_keys(sources, prepared=None):
    """Use saved canonical identities for saved Source objects, without RDKit."""
    known = {}
    if prepared is not None:
        known = {
            id(prepared.compounds.compounds[int(cid)]): smiles
            for cid, smiles in zip(
                prepared.source_compounds.compound_index.tolist(),
                prepared.source_smiles,
            )
        }
    result = []
    for source in sources:
        identity = id(source)
        if identity not in known:
            known[identity] = source.smiles
        result.append(known[identity])
    return result


def prepare_incremental(
    generator,
    sources,
    precursor_types,
    collision_energy,
    *,
    precomputed=None,
    prepared=None,
):
    if not sources or not len(sources) == len(precursor_types) == len(collision_energy):
        raise ValueError(
            "sources, precursor_types and collision_energy need equal nonzero lengths"
        )
    grouped = {}
    for i, key in enumerate(source_keys(sources, prepared)):
        grouped.setdefault(key, []).append(i)
    structures, unique, reactions, adducts, order = [], [], [], [], []
    for smiles, indices in grouped.items():
        source_index = (
            prepared.source_smiles.index(smiles) if prepared is not None else 0
        )
        chemistry = _Chemistry(
            generator,
            sources[indices[0]],
            prepared=prepared,
            source_index=source_index,
            precomputed=precomputed.get(smiles) if precomputed else None,
        )
        selected_adducts = [precursor_types[i] for i in indices]
        structures.append(
            _prediction_data(
                chemistry, selected_adducts, [collision_energy[i] for i in indices]
            )
        )
        unique.append(chemistry.source)
        reactions.append(chemistry.reactions)
        adducts.extend(selected_adducts)
        order.extend(indices)
    data = (
        structures[0]
        if len(structures) == 1
        else SpectrumPredictionData.from_structures(structures)
    )
    return data, tuple(unique), tuple(reactions), tuple(adducts), tuple(order)
