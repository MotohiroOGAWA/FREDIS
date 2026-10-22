# Prepared input tables

Schema and reference diagrams: [SCHEMA.md](SCHEMA.md).
Current saved format: `fredis.annotated-spectrum-training.v4`.
Prepared files from older formats must be regenerated.

## Independent tables

| Class | Structure | Contents |
|---|---|---|
| [SourceCompound](source_compound.py) | `structure.source_compounds` | `source_graph: Batch`, `source_smiles: tuple[str,...]`, `symbols: tuple[str,...]` |
| [ReactionTensorData](reaction_tensor_data.py) | `structure.reactions` | Concrete reactions referencing Source rows/atoms and other reaction rows |
| [ReactionSet](reaction_set.py) | `structure.reaction_sets` | Only `reaction_ptr: int64 [N_set+1]` and `reaction_index: int64 [N_members]` |

Each ReactionSet row is one unique ascending set of Batch-global concrete
reaction IDs. Order, CE, main adduct, topology and teacher labels do not define
another set. The empty set is one shared row; its Source must be supplied by the
referencing context. Nonempty sets may not mix reactions from different Sources.
`from_rows()` interns sets and returns the input-row -> canonical-ID mapping.

```python
reaction_sets, ids = ReactionSet.from_rows([[], [1, 0], [0, 1], [0], []])
# reaction_sets.rows() == [[], [0, 1], [0]]
# ids.tolist() == [0, 1, 1, 2, 0]

# Different groups may reference the same independent set ID.
contexts.candidate_branch_group_index = [0, 0, 1, 1]
contexts.candidate_reaction_set_index = [0, 1, 0, 1]
```

## Contexts reference sets; they do not own memberships

`structure.states` is [ReactionContextData](reaction_context_data.py). V denotes
candidate **context** rows, each unique on `(branch group, reaction-set ID)`.
Q denotes active teacher-context rows selected from V. Neither table owns a
reaction membership CSR.

| Field | Shape / referenced rows |
|---|---|
| `candidate_branch_group_index` | int64 [V], branch-group rows |
| `candidate_reaction_set_index` | int64 [V], independent ReactionSet rows |
| `candidate_fragment_index` | int64 [V], runtime occurrence rows; -1 if unmaterialized |
| `candidate_index` | int64 [Q], V context rows |
| `branch_group_index` | int64 [Q], branch groups of the selected V rows |
| `parent_index` | int64 [Q], representative Q parent; -1 for roots |
| `added_reaction_index` | int64 [Q], added registry reaction; -1 for roots |
| `fragment_node_index` | int64 [Q], runtime occurrence; -1 if unmaterialized |
| `transition_parent_index` | int64 [N_transition], Q parent rows |
| `transition_child_index` | int64 [N_transition], Q child rows |
| `transition_reaction_index` | int64 [N_transition], added registry reactions |
| `next_ptr` / `next_reaction_index` | CSR, Q context -> valid registry additions |
| `next_state_index` | Same continuation positions, V destination contexts |

`structure.candidate_reactions()` and `active_reactions()` gather memberships
through these foreign keys as temporary CSR projections. Their values reference
the global registry. `state_reaction_count` computes Q-context reaction counts.
Training translates global reaction IDs into the decoder's local pool slots only
at scoring time. There is no persisted group-local membership table.

## Compound and annotation references

[AnnotatedSpectrumData](annotated_spectrum_data.py) holds spectra, unique
Compounds, charged formulas, ions, and peak assignments. Same-structure Compounds
share one graph and identity. Within a group they share a materialized node,
even when reached through different ReactionSets.

| Field | Meaning |
|---|---|
| `compounds.occurrence_index` | Runtime occurrence -> unique Compound ID |
| `compounds.context_compound_index` | [V] materialization context -> Compound ID; -1 if unavailable |
| `context_reaction_set_index` | [V] context -> canonical ReactionSet ID |
| `compounds.reaction_set_ptr/index` | CSR, Compound -> unique generating ReactionSet IDs; pointer length N_compound+1 |
| `occurrence_group_index` | Branch group per runtime occurrence |

The inverse CSR stores distinct `(Compound, ReactionSet)` pairs, not context IDs.
Repeated groups do not duplicate a pair. The same empty set may occur under more
than one Compound because its Source context is held by the reference side.
Observation intensity and formula assignment targets are derived without copying
measurements into canonical chemical tables.

## TeacherData

Teacher labels and paths reference Q contexts, which in turn reference canonical
ReactionSets. Thus labels remain specific to group/context while membership is
shared. The fields are:

- `peak_index [H]`: observed peak rows; `peak_branch_group_index [H]`: group context.
- `observed bool [Q]`: whether a context explains an observed ion.
- `positive_reaction_ptr/index`, `negative_reaction_ptr/index`: Q-context CSR lists
  of positive and weak-negative global registry additions.
- `peak_path_ptr [H+1]`: alternative paths per supervised peak.
- `path_step_ptr [N_path+1]`: ordered steps per path.
- `path_step_state_index [N_step]`: parent Q contexts.
- `path_step_reaction_index [N_step]`: added global concrete reaction IDs.

## SourceReactionStructure fields and lifecycle

| Field | Meaning |
|---|---|
| `source_compounds` | Independent SourceCompound table |
| `reactions` | Concrete reaction registry |
| `reaction_sets` | Independent membership table |
| `states` | Group-local context/transition foreign keys |
| `teacher` | Context-specific labels and paths |
| `sample_tree_index int64 [S]` | Spectrum -> SourceCompound row |
| `collision_energy_eV float32 [S]` | Spectrum CE in eV |
| `sample_branch_group_index int64 [S]` | Spectrum -> branch group |
| `branch_group_tree_index int64 [G]` | Group -> SourceCompound row |
| `branch_group_adduct_index int64 [G]` | Group -> model main-adduct vocabulary ID |
| `annotations` | Normalized measured/chemical tables, or None |
| `observations` | Initialization-only dictionaries; normalized into annotations |
| `formula_elements` | Formula element-column order, excluding final charge |
| `spectrum_batch` | Expanded runtime topology/ion rows, or None |
| `source_atom_capacity` | Maximum Source-local atom capacity in a collated batch |
| `max_reaction_role_count` | Maximum matched Source atom count per reaction |

`source_graph` and `source_smiles` are read-only views of `source_compounds`.
Adduct is group-level int64 and CE is spectrum-level float32. `device`,
`num_samples`, and `num_branch_groups` are derived properties. `sample_annotations`
exports normalized data as viewer dictionaries.

- `from_structures()` collates Source/reaction registries, shifts global reaction
  members, interns sets, and remaps dependent context and Compound references.
- `select_samples()` shares the independent registries and slices/remaps context,
  peak, ion and topology rows. Unused independent rows can remain available.
- `without_teacher()` removes Q projections and labels, retaining V contexts,
  the ReactionSet table, materialization and observations.
- `validate()` checks canonical membership and cross-table references, including
  Source ownership and Compound's inverse set CSR.
- `.to(device)` moves tensor-bearing tables and returns a replaced structure.
- Save/load omits redundant runtime graph/formula caches and restores them from
  normalized annotations. Loading/validation does not rerun chemistry.

Dataclasses are frozen, but tensors themselves are not deeply immutable. Device
transfers and replacements must keep the independent table foreign keys aligned.
