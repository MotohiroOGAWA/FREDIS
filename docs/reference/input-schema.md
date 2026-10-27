# Prepared input schema and tensor ownership

The model uses shared registries with batch-local integer indices. A source
molecule, materialized Compound, execution history, formula explanation, and
measured peak occupy different index spaces. Their explicit references preserve
ambiguity and avoid repeated chemical/neural work.

## Training and prediction roots

| Root object | Contents |
|---|---|
| `SpectrumPredictionData` | Shared chemistry and requested measurement conditions |
| `SourceReactionStructure` | The prediction fields plus observed peaks, assignments, BranchTeacher, and spectrum targets |

`SourceReactionStructure.for_prediction()` removes observation/teacher fields and
branch diagnostic labels. The prepared execution graph can remain shared; it
contains chemical/history references rather than observed intensities. Loading
and batching use stored memberships instead of running chemistry again.

```{mermaid}
flowchart TB
    SOURCE["SourceCompound<br/>Original graphs and Compound references"] --> REACTION["ReactionTensorData<br/>Concrete matches and constraints"]
    REACTION --> SETS["ReactionSet<br/>Unique sorted reaction sets"]
    SETS --> COMPOUND["CompoundBatch<br/>Shared materialized structures"]
    COMPOUND --> GRAPH["CompoundReactionGraph / execution histories"]
    COMPOUND --> EXPLAIN["FormulaExplanationData<br/>Compound + ion + shift + formula"]
    ADDUCT["AdductData / BranchGroup / conditions"] --> GRAPH
    ADDUCT --> EXPLAIN
    FORMULA["FormulaData<br/>Composition, m/z and validity"] --> EXPLAIN
    EXPLAIN --> CANDIDATES["SpectrumExplanationData"]
    PEAKS["Measured SpectrumData"] --> ASSIGN["PeakAssignmentData"]
    CANDIDATES --> ASSIGN
    ASSIGN --> TEACHER["BranchTeacher<br/>Peak / depth / history memberships"]
    ASSIGN --> TARGET["SpectrumFormulaTarget<br/>Intensity and masks"]
    GRAPH --> TEACHER
```

## Dimensions

| Symbol | Meaning |
|---|---|
| `U` | Source molecules |
| `A` | Packed source atoms |
| `E` | Directed packed bonds |
| `R` | Concrete source reaction instances |
| `C` | Unique materialized Compounds |
| `S` | Spectrum conditions |
| `F` | Formula rows |
| `X` | Formula explanation rows |
| `K` | Execution states |
| `T` | Execution transitions |

Integer registry indices use `torch.long` unless a field explicitly stores a
boolean mask or diagnostic class label. Atom features use floating-point arrays;
measured m/z values and neural feature dtypes have different ownership.

## Sources, reactions, and sets

| Table / field | Shape and reference |
|---|---|
| `SourceCompound.source_graph.x` | `[A, len(symbols)+26]` atom channels |
| `source_graph.edge_index` | `[2,E]`, references packed source atom rows |
| `source_graph.edge_attr` | `[E,9]` bond channels |
| `source_graph.ptr` | `[U+1]`, source atom offsets |
| `SourceCompound.compound_index` | `[U]`, source roots in CompoundBatch |
| `ReactionTensorData.reaction_type` | `[R,3]`: pattern, rule, product IDs |
| `reaction_source_compound_index` | `[R]`, source owner |
| `reaction_source_atom_ptr/index` | CSR reactant atom references in SMARTS order |
| `reaction_retained_atom_ptr/index` | CSR retained source atom references |
| `reaction_conflict_index`, `reaction_invalidation_index` | Sparse concrete-reaction pairs |
| `ReactionSet.reaction_ptr/index` | CSR sorted unique concrete-reaction IDs |

A two-position pattern's match `[7,3]` preserves SMARTS position order, even
though it is not sorted by atom ID. A ReactionSet is instead canonicalized as
a sorted set of concrete reaction IDs.

CSR represents variable-length rows with one pointer and one flattened index:

```python
ptr = [0, 2, 2, 5]
index = [3, 7, 1, 4, 9]
# Rows are [3, 7], [], [1, 4, 9].
```

## Structures and histories

CompoundBatch stores unique molecular structures, corresponding graphs, and
references to generating ReactionSets. CompoundReactionGraph's
`source_compound_index`, `target_compound_index`, and `reaction_index` encode
physical structural transitions.

ReactionExecutionGraph distinguishes admissible histories:

| Field | Shape / reference |
|---|---|
| `compound_index` | `[K]`, CompoundBatch rows |
| `branch_group_index` | `[K]`, source/main-adduct groups |
| `reaction_set_index` | `[K]`, applied ReactionSets |
| `source_index`, `target_index` | `[T]`, execution states |
| `reaction_index` | `[T]`, added concrete reaction |
| `root_index` | One root execution state per BranchGroup |

Two execution states can share `compound_index` and have different applied sets.
The next compatible reactions and state-conditioned ion probabilities can differ.
Molecular embeddings and physical spectrum GAT nodes remain shared.

## Ion conditions and formula explanations

AdductData owns main/fragment ion vocabulary and hydrogen shifts. BranchGroup
pairs one source with a main adduct. Spectrum conditions reference BranchGroups
and contain CE/observed precursor metadata.

FormulaData stores composition, charge-aware identity, theoretical m/z, and
validity. FormulaExplanationData links each explanation to a Compound, formula,
fragment-ion adduct, and hydrogen shift. SpectrumExplanationData has CSR rows
listing a spectrum's explanations, including valid unobserved candidates.

Several Compounds, sets, or ion conditions can explain the same Formula.
Spectrum probabilities are summed by `(spectrum, Formula)` before loss and final
output aggregation; explanations remain distinct until that step.

## Observations and learning targets

SpectrumData stores measured peak CSR, m/z, intensities, precursor membership,
and condition metadata. PeakAssignmentData's `formula_explanation_ptr/index`
retains each peak's alternative explanations.

BranchTeacher owns an execution graph and peak/depth learning memberships:

| Membership | Purpose |
|---|---|
| `peak_index`, `peak_branch_group_index` | Link learning units to measured peaks and source/adduct groups |
| `node_state_index`, `node_peak_index`, `node_depth` | Map peak-specific history nodes to shared execution states |
| `source_index`, `target_index`, `edge_index` | Connect saved teacher paths to execution transitions |
| `bag_peak_index`, `bag_depth` | Define peak/depth learning bags |
| `node_index`, `node_bag_index` | Positive prefix-node memberships |
| `negative_edge_index`, `negative_bag_index` | Corresponding weak-negative transitions |

Alternative structures for one peak share a MIL unit. Transitions supporting
another observed peak in the BranchGroup are excluded from negatives. Temperature,
class weights, and depth/peak averages are calculated during training.

SpectrumFormulaTarget independently stores spectrum/formula indices, observation
flags, maximum assigned intensity, and a supervision mask. Unmatched valid
candidates have zero target intensity; unavailable/masked candidates do not
participate in measured intensity objectives.

## Persistence, selection, and device transfer

`save()` writes `.preft.pt` on CPU. `load()` validates the stored schema and teacher
memberships without rebuilding them. `from_structures()` remaps table references;
`select_samples()` supports reordering and repeated spectrum indices. `.to(device)`
constructs tables on the requested device.

Execution geometry, packed spectrum batches, and condition/topology indices are
runtime caches. They do not contain learned features and are refreshed for
replaced/device-copied inputs. The [data API](../api/data.md) exposes the relevant
methods. The source-level
[field registry](https://github.com/MotohiroOGAWA/FREDIS/blob/main/fredis/ml/input/SCHEMA.md)
provides further examples of every stored table.
