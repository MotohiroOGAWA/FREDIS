# SourceReactionStructure field guide

Relational schema diagrams: [SCHEMA.md](SCHEMA.md).

`SourceReactionStructure` connects source molecular graphs, concrete reactions,
reaction-set states, observations, and supervision. This guide describes the
current fields, including separate integer main-adduct IDs and floating-point CE.

Definitions: [source_reaction_structure.py](source_reaction_structure.py),
[reaction_tensor_data.py](reaction_tensor_data.py),
[reaction_state_data.py](reaction_state_data.py), and
[annotated_spectrum_data.py](annotated_spectrum_data.py).

## Example and index spaces

All numbers below illustrate storage and indexing. Reaction patterns, paths,
feature vectors, and assignments are schematic, not outputs of a chemistry run.
Examples describe individual tables; they are not a complete constructor fixture.

| Spectrum s | Source m | Molecule | Main adduct | CE (eV) | Group g |
|---|---|---|---|---|---|
| 0 | 0 | Ethanol, CCO | [M+H]+ | 10 | 0 |
| 1 | 0 | Ethanol, CCO | [M+H]+ | 30 | 0 |
| 2 | 0 | Ethanol, CCO | [M+Na]+ | 20 | 1 |
| 3 | 1 | Acetic acid, CC(=O)O | [M+H]+ | 40 | 2 |

A group is `(source molecule, main adduct)`; CE does not define a new group.

| Symbol | Meaning | Example |
|---|---|---|
| M | Source molecules | 2 |
| S | Spectrum samples | 4 |
| G | Branch groups | 3 |
| R | Concrete source-specific reactions | 3 |
| V | Candidate reaction-set states | 8 |
| Q | Active states referenced by supervision | 5 |
| P | Observed peaks | 5 |
| C | Unique compounds | Depends on materialization |
| F | Unique charged formulas | Depends on ion generation |
| I | Unique structural ion candidates | Depends on ion generation |
| J | Runtime spectrum/fragment/ion rows | Depends on expansion |

Most indices address the current object or collated batch; canonical reaction-set
values specifically use group-local reaction IDs. A reaction ID is not a
pattern ID; a compound ID is not a fragment occurrence ID. Index tensors use
`torch.int64` (`torch.long`) unless stated otherwise. Empty optional tables are
valid in the appropriate preparation/training stage.

## Reaction-set CSR and Compound-to-set CSR

Reaction sets and Compound nodes have separate identities. Within a branch group,
`A -> B` and `B -> A` give the same set `{A,B}`. Different sets producing the same
Compound remain separate CSR rows, while the Compound stays one node within the
group. Different groups retain their own topology and share normalized Compound
identity and structure features.

Two CSR levels encode the relationship:

1. `states.candidate_reaction_ptr [V+1]` and
   `states.candidate_reaction_index [L]`: set -> reaction IDs. IDs are **group-local,
   strictly ascending within each set**. Local order is the ascending registry
   rows of that group's Source. Collation does not offset these stored values.
2. `annotations.compounds.reaction_set_ptr [C+1]` and
   `annotations.compounds.reaction_set_index [K]`: Compound -> generating set IDs.
   These IDs reference the V rows of the first CSR. This pointer has **C+1**
   entries, not V+1. Unmaterialized sets are omitted from this inverse relation.

For one group, suppose sets `{}`, `{0}`, `{1}`, `{0,1}` produce Compounds
`C0`, `C1`, `C1`, `C2` respectively:

```python
# Set IDs:                    0    1    2      3
# Reactions:                 {}  {0}  {1}  {0,1}
candidate_reaction_ptr = tensor([0, 0, 1, 2, 4])
candidate_reaction_index = tensor([0, 1, 0, 1])

# C0 -> [set 0], C1 -> [set 1, set 2], C2 -> [set 3]
reaction_set_ptr = tensor([0, 1, 3, 4])  # C=3, shape [4]
reaction_set_index = tensor([0, 1, 2, 3])
state_index = tensor([0, 1, 1, 2])       # Set -> Compound
```

C1 is one Compound/node with two generating sets. Given Compound `c`, read
`ids = reaction_set_index[reaction_set_ptr[c]:reaction_set_ptr[c+1]]`, then
slice the reaction-set CSR for each ID. `SourceReactionStructure` provides
`candidate_reaction_registry_index` to convert the stored local reaction IDs to
batch-global reaction registry rows when needed. Teacher transition reaction IDs
remain registry IDs; only the canonical set CSR stores group-local IDs.

The schema is `fredis.annotated-spectrum-training.v3`. Previous prepared data
must be regenerated because reaction-set indices now have group-local meaning
and the inverse CSR has explicit reaction-set field names.

## Spectrum and group fields

The following arrays encode the four input spectra. The adduct vocabulary here
is illustrative and belongs to the model; main-adduct labels are not stored in
`branch_group_adduct_index` itself.

```python
import torch
from torch import tensor, int64, float32

adduct_type_strs = ("[M+H]+", "[M+Na]+")
source_smiles = ("CCO", "CC(=O)O")
sample_tree_index = tensor([0, 0, 0, 1], dtype=int64)
collision_energy_eV = tensor([10., 30., 20., 40.], dtype=float32)
sample_branch_group_index = tensor([0, 0, 1, 2], dtype=int64)
branch_group_tree_index = tensor([0, 0, 1], dtype=int64)
branch_group_adduct_index = tensor([0, 1, 0], dtype=int64)

sample_adduct = branch_group_adduct_index[sample_branch_group_index]
# tensor([0, 0, 1, 0])
assert torch.equal(
    sample_tree_index,
    branch_group_tree_index[sample_branch_group_index],
)
```

| Field | Type / shape | Meaning and example |
|---|---|---|
| `sample_tree_index` | int64 [S] | Spectrum → source; spectrum 2 refers to source 0. |
| `collision_energy_eV` | float32 [S] | Physical CE in eV; spectrum 1 has 30 eV, before CE-encoder normalization. |
| `sample_branch_group_index` | int64 [S] | Spectrum → group; spectra 0 and 1 share group 0. |
| `branch_group_tree_index` | int64 [G] | Group → source; groups 0 and 1 both use source 0. |
| `branch_group_adduct_index` | int64 [G] | Group → model main-adduct vocabulary ID; group 1 uses ID 1. |
| `source_smiles` | tuple[str, ...], normally length M | Source SMILES in source order; default `()`. |

`source_smiles` remains a tuple of Python strings: strings have varying lengths,
and this field is primarily for identification/display. A fixed-width NumPy
`np.str_` array may waste space and truncate longer replacement strings.
`dtype=object` still stores Python string objects.

The three group-index fields default to empty int64 tensors. Fully prepared data
should populate them. `prepare_source_reactions` requires explicit group adduct
IDs; if sample group IDs are omitted, it creates one group per spectrum.

## source_graph: source molecular graphs

A PyG `Batch` combines M source graphs. In the example, assume hydrogens are
features rather than separate nodes: ethanol has 3 nodes and acetic acid has 4.
Global atom rows are ethanol C0–C1–O2 and acetic acid C3–C4(=O5)–O6.

- `ptr [M+1]` delimits source atom rows.
- `batch [7]` maps each atom to its source.
- `x [7, atom_dim]` holds atom features.
- `edge_index [2, 10]` stores both directions of five bonds.
- `edge_attr [10, bond_dim]` holds bond features, including bond order.
- Molecular descriptors may also be present.

Atom rows address `source_graph.x`, not atom-map labels. The actual feature
columns and explicit-hydrogen policy come from `MolGraphBuilder`.

```python
source_graph_ptr = tensor([0, 3, 7])
source_graph_batch = tensor([0, 0, 0, 1, 1, 1, 1])
source_graph_edge_index = tensor([
    [0, 1, 1, 2, 3, 4, 4, 5, 4, 6],
    [1, 0, 2, 1, 4, 3, 5, 4, 6, 4],
])
# Source 1 atoms: x[ptr[1]:ptr[2]] == x[3:7]
```

## reactions: concrete reaction registry

`ReactionTensorData` stores R source-specific reactions. Applying the same pattern
to different matches can produce different reaction IDs. All fields below are
int64. `reaction_type` columns are pattern ID, rule ID, and product molecule slot;
the last column is not an ID into `annotations.compounds`.

The example gives reactions 0/1 to source 0 and reaction 2 to source 1.

```python
reaction_tree_index = tensor([0, 0, 1])               # [R]
reaction_type = tensor([[10, 0, 0], [11, 0, 0], [10, 1, 0]])  # [R, 3]
reaction_source_atom_ptr = tensor([0, 2, 4, 6])        # [R+1]
reaction_source_atom_index = tensor([0, 1, 1, 2, 4, 6])
reaction_retained_atom_ptr = tensor([0, 2, 4, 7])      # [R+1]
reaction_retained_atom_index = tensor([1, 2, 0, 1, 3, 4, 5])
reaction_conflict_index = torch.empty((2, 0), dtype=int64)
reaction_invalidation_index = tensor([[0, 1], [1, 0]]) # [2, pair_count]
```

`*_ptr` and `*_index` encode variable-length lists in CSR form. Reaction 1
matches `index[ptr[1]:ptr[2]] = [1, 2]` and retains atoms `[0, 1]`.

- `reaction_conflict_index`: each column is a pair of reactions changing a common
  source bond; both directions are stored. This example has no such pairs.
- `reaction_invalidation_index`: each column is `[invalidating, invalidated]`.
  Here reaction 0 discards atom 0 required by reaction 1, while reaction 1 discards
  atom 2 required by reaction 0. Invalidation is directional in general.

## states: reaction sets and transitions

`ReactionStateData` separates V candidate states from Q active states used by
supervision. A candidate state is identified by a group and a concrete reaction
set. Distinct states can materialize to the same compound.

| V row | Group | Reaction set |
|---|---|---|
| 0 | 0 | {} |
| 1 | 0 | {0} |
| 2 | 0 | {1} |
| 3 | 1 | {} |
| 4 | 1 | {0} |
| 5 | 1 | {1} |
| 6 | 2 | {} |
| 7 | 2 | {0} |

Candidate fields (int64). Source 1's registry reaction 2 is local reaction 0
within group 2:

```python
candidate_branch_group_index = tensor([0, 0, 0, 1, 1, 1, 2, 2])  # [V]
candidate_reaction_ptr = tensor([0, 0, 1, 2, 2, 3, 4, 4, 5])    # [V+1]
candidate_reaction_index = tensor([0, 1, 0, 1, 0])
candidate_fragment_index = tensor([0, 1, 2, 3, 4, 5, 6, 7])     # [V]
compound_index = tensor([0, 1, 2, 0, 1, 2, 3, 4])               # [V]
```

`candidate_fragment_index` references runtime fragment occurrences; use -1 for
unmaterialized states. `compound_index` references unique compounds, also with -1
for unmaterialized states. It is synchronized from annotations when available.
Occurrence 0 and occurrence 3 can both refer to compound 0.

Suppose Q rows select V rows `[0, 1, 3, 6, 7]`:

```python
candidate_index = tensor([0, 1, 3, 6, 7])          # [Q], Q -> V
branch_group_index = tensor([0, 0, 1, 2, 2])        # [Q]
parent_index = tensor([-1, 0, -1, -1, 3])           # [Q], Q -> Q
added_reaction_index = tensor([-1, 0, -1, -1, 2])   # [Q]
fragment_node_index = tensor([0, 1, 3, 6, 7])       # [Q]
transition_parent_index = tensor([0, 3])            # [E], Q rows
transition_child_index = tensor([1, 4])             # [E], Q rows
transition_reaction_index = tensor([0, 2])          # [E], reaction IDs
next_ptr = tensor([0, 2, 2, 4, 5, 5])              # [Q+1]
next_reaction_index = tensor([0, 1, 0, 1, 2])
next_state_index = tensor([1, 2, 4, 5, 7])          # V rows, NOT Q rows
```

`parent_index` and `added_reaction_index` describe a representative parent;
roots use -1. Full edges use `transition_*`, allowing multiple paths to a state.
`fragment_node_index` uses runtime occurrence IDs, with -1 if unmaterialized.
`next_*` lists chemically valid additions, including candidates not selected into
Q. For q0, valid additions are reaction 0 → v1 and reaction 1 → v2.

Computed properties are not separately stored:
`reaction_count = [0, 1, 0, 0, 1]`,
`reaction_ptr = [0, 0, 1, 1, 1, 2]`, and `reaction_index = [0, 0]`.

## teacher: supervision referencing states and observations

`TeacherData` stores labels and alternative explanation paths. It does not copy
peak intensities or formula vectors. Here q1 and q4 explain observed ions;
reaction 0 from q0 and reaction 2 from q3 are positive additions. The negative
example below is illustrative; preparation determines actual weak negatives.

```python
peak_index = tensor([0, 3])                         # [H], observed peak IDs
positive_reaction_ptr = tensor([0, 1, 1, 1, 2, 2])    # [Q+1]
positive_reaction_index = tensor([0, 2])
observed = tensor([False, True, False, False, True])  # bool [Q]
negative_reaction_ptr = tensor([0, 1, 1, 1, 1, 1])    # [Q+1]
negative_reaction_index = tensor([1])
peak_path_ptr = tensor([0, 1, 2])                    # [H+1]
peak_branch_group_index = tensor([0, 2])             # [H]
path_step_ptr = tensor([0, 1, 2])                    # [path_count+1]
path_step_state_index = tensor([0, 3])               # [step_count], Q rows
path_step_reaction_index = tensor([0, 2])            # [step_count], reactions
```

Peak item 0 references observed peak 0 and has one path: add reaction 0 at q0.
Peak item 1 references observed peak 3 and has one path: add reaction 2 at q3.
The two CSR levels express peak → alternative paths → steps.
`teacher.spectrum_targets(annotations)` derives runtime intensity targets using
peak/formula assignments and maximum assigned intensity per spectrum/formula.
It returns None if `observed` is empty.

## Capacity and formula metadata

| Field | Example | Meaning |
|---|---|---|
| `max_reaction_role_count: int` | 2 | Maximum number of matched source atoms of any concrete reaction; not the reaction count. Default 0. |
| `source_atom_capacity: int` | 4 | Maximum source-local atom capacity in this batch; not the total 7 atoms. Default 0. Used for retained-atom bitsets. |
| `formula_elements: tuple[str, ...]` | ("C", "H", "O", "Na") | Element column order, excluding the final charge column. Default (). |

With that element order, `[2, 6, 1, 1, 1]` means C2H6ONa with charge +1.
Collation takes the maximum capacities and requires identical element orders.
`max_reaction_role_count` is currently populated/collated but has no further
consumer in the ML package.

## annotations: normalized observations and chemical explanations

Optional `AnnotatedSpectrumData` defaults to None. It separates unique chemical
identities from repeated runtime occurrences and spectrum conditions.

### spectra: measured data

Here spectra contain 1, 1, 1, and 2 peaks. m/z values are rounded examples;
intensities are illustrative input relative intensities.

```python
peak_ptr = tensor([0, 1, 2, 3, 5])                  # [S+1]
mz = tensor([31.0184, 31.0184, 69.0311, 43.0184, 61.0284])  # [P]
intensity = tensor([0.4, 1.0, 1.0, 1.0, 0.3])       # [P]
precursor = tensor([False, False, True, False, True]) # bool [P]
source_index = tensor([0, 0, 0, 1])                  # [S]
branch_group_index = tensor([0, 0, 1, 2])            # [S]
precursor_mz = tensor([47.0491, 47.0491, 69.0311, 61.0284])
collision_energy = tensor([10., 30., 20., 40.])
adducts = ("[M+H]+", "[M+H]+", "[M+Na]+", "[M+H]+")
metadata = tuple({"recordIndex": i} for i in [100, 101, 102, 103])
```

`peak_ptr` selects peaks of one spectrum, e.g. spectrum 3 uses `mz[3:5]`.
`precursor` records precursor membership under the configured mass tolerance.
`adducts` are observed precursor labels and may differ from resolved main-adduct
labels. `metadata` excludes peaks, conditions, and assignments.
Normalized measured m/z, precursor m/z, and CE are float64; intensities are
float32 and precursor flags are bool. The code block abbreviates dtype arguments.
`collision_energy` and top-level `collision_energy_eV` describe corresponding
samples; direct construction must keep them consistent.

### compounds: unique structures

`smiles`, `graph`, and `compounds` hold C canonical isomeric SMILES, C packed
molecular graphs, and C molecule objects. Identity includes stereochemistry and
charge. `occurrence_index [occurrences]` maps runtime occurrences to compounds;
`state_index [V]` maps candidate states to compounds (-1 if unmaterialized).
For the state example, both maps can be `[0,1,2,0,1,2,3,4]`.
`reaction_set_ptr [C+1]` and `reaction_set_index` form the inverse compound → V-state relation:
`reaction_set_ptr=[0,2,4,6,7,8]`, `reaction_set_index=[0,3,1,4,2,5,6,7]`.

### formulas: unique charged formulas

`composition [F,D+1]` stores D element counts plus charge. `mz [F]` stores exact
m/z (rounded here). `valid [F]` flags valid elemental counts. `ion_ptr [F+1]` and
`ion_index` map each formula to all structural ion explanations.

```python
composition = tensor([
    [1, 3, 1, 0, 1],  # CH3O+
    [2, 6, 1, 1, 1],  # C2H6ONa+
    [2, 3, 1, 0, 1],  # C2H3O+
    [2, 5, 2, 0, 1],  # C2H5O2+
])
mz = tensor([31.0184, 69.0311, 43.0184, 61.0284])
valid = tensor([True, True, True, True])
```

### ions and assignments

`ions.compound_index`, `formula_index`, `adduct_index`, and
`hydrogen_shift_index` are [I] references defining unique ion candidates.
Adduct IDs refer to `annotations.ion_adducts`, not necessarily the model's
main-adduct vocabulary. `hydrogen_shifts` contains signed shift values;
for `[-1,0,1]`, shift index 1 means a shift of 0, not +1.

`ions.runtime_index [J]` maps runtime rows to unique ion IDs, and
`runtime_sample_index [J]` gives their spectrum IDs. Repeated CE samples can use
the same unique ion ID: runtime IDs `[0,0]` and sample IDs `[0,1]` mean one ion
candidate evaluated for spectra 0 and 1.

`assignments.formula_ptr [P+1]` / `formula_index` map peaks to alternative
formulas. `assignments.ion_ptr [P+1]` / `ion_index` map peaks to admissible
structural ion explanations. For five peaks each assigned one formula,
`formula_ptr=[0,1,2,3,4,5]` and `formula_index=[0,0,1,2,3]`.
Ambiguous peaks can have multiple entries; unmatched peaks have empty rows.

Remaining annotation fields:

- `elements`: formula element order, excluding charge.
- `reaction_state_index`: V candidate-state row IDs; e.g. `[0,1,2,3,4,5,6,7]`.
- `occurrence_group_index`: group for each runtime occurrence; e.g.
  `[0,0,0,1,1,1,2,2]`. It is display/topology information, not chemical identity.

## observations: initialization-only dictionaries

`observations` is an `InitVar`, default None, not a persisted dictionary field.
When nonempty observations and a spectrum batch are supplied, `__post_init__`
normalizes and validates them into `annotations`, then synchronizes
`states.compound_index`. This is one sample dictionary; the full input has four.

```python
observations = ({
    "precursorMz": 47.0491,
    "collisionEnergy": 10.0,
    "adduct": "[M+H]+",
    "recordIndex": 100,
    "peaks": [{
        "mz": 31.0184,
        "intensity": 0.4,
        "precursor": False,
        "matches": [{
            "nodeIndices": [1],
            "fragmentIonAdduct": "[M+H]+",
            "hydrogenShift": 0,
        }],
    }],
},)
```

## spectrum_batch: expanded runtime model input

The annotation is `object | None`, but the runtime value is normally
`SpectrumModelBatch` from `../specgen/fragment_spectrum_model.py`. Default None
allows reaction-only/scorer workflows.

For illustration, give each of the eight fragment occurrences one ion condition.
Group 0 has three occurrences reused by both spectra 0 and 1, so J=11.

```python
state_node_index = tensor([0,1,2, 0,1,2, 3,4,5, 6,7])
state_sample_index = tensor([0,0,0, 1,1,1, 2,2,2, 3,3])
row_ce = collision_energy_eV[state_sample_index]
# tensor([10.,10.,10., 30.,30.,30., 20.,20.,20., 40.,40.])
```

| Runtime field | Meaning |
|---|---|
| `decoded` | Materialized fragment topology, graphs, and compound occurrences. |
| `state_node_index [J]` | Fragment occurrence row, int64. |
| `state_sample_index [J]` | Spectrum ID, int64. |
| `state_formula_tensor [J,D+1]` | Charged formula per row. |
| `state_mz [J]` | m/z per row. |
| `state_adduct_index [J]` | Fragment-ion adduct vocabulary ID, int64. |
| `state_hydrogen_shift [J]` | Signed hydrogen shift value, not a vocabulary index. |
| `target_intensity [J]` | Derived target intensities or None. |
| `precursor_adducts` | Precursor labels in spectrum order. |
| `ion_type_strs` | Fragment-ion adduct vocabulary. |
| `unique_node_graph` | Optional graph batch for unique structures not shared with Source graphs (a placeholder graph may be used when empty). |
| `node_graph_inverse` | Runtime node → unique structure ID in `unique_smiles`. |
| `unique_source_index` | Source ID per unique structure when it matches a Source; -1 otherwise. |
| `unique_smiles` | SMILES per unique structure ID, including structures shared with Sources. |
| `node_smiles` | SMILES per runtime node occurrence. |

For row 4, `state_node_index[4]=1` and `state_sample_index[4]=1`: fragment
occurrence 1 is evaluated for spectrum 1 at 30 eV. It could have
`state_formula_tensor[4]=[1,3,1,0,1]`, `state_mz[4]=31.0184`, and
`target_intensity[4]=1.0`. Multiple adduct/shift conditions produce more rows.
Runtime `state_*` rows are neither Q state rows nor V candidate rows.

## Lifecycle, defaults, and derived views

- The dataclass is frozen: use `dataclasses.replace` for field replacement.
  This does not make tensor contents deeply immutable.
- `states` and `teacher` default to fresh empty tables. `annotations` and
  `spectrum_batch` default to None; metadata tuples default to ().
- `num_samples` is `sample_tree_index.numel()` (4 here).
- `num_branch_groups` is the group-tree table length (3 here), falling back to
  the sample count if that table is empty.
- `device` comes from `reactions.reaction_type.device`; `.to(device)` moves
  fields exposing a `.to()` method and returns a replaced structure.
- `sample_annotations` exports viewer dictionaries from normalized annotations,
  or returns () when annotations are absent.
- `without_teacher()` removes Q-state supervision and teacher labels/targets,
  retaining V candidates and observations.
- `from_structures()` concatenates tables and offsets source, atom, reaction,
  group, state, and peak references as appropriate. Canonical reaction-set values
  stay group-local. Pattern/rule IDs remain
  vocabulary IDs. Capacities use maxima rather than sums.
- When annotations and a runtime batch coexist, serialization omits redundant
  runtime molecular graphs, formulas, m/z, targets, and graph caches. Loading
  restores the needed runtime views from normalized data.
- Files using the old combined `condition_features [S,2]` layout must be
  regenerated. Adduct is now group-level int64; CE is sample-level float32.

Preparation normalizes CE to float32 and validates its [S] shape, sample-group
shape, and group-adduct shape. The dataclass itself does not automatically
validate every semantic relationship shown in this guide.
