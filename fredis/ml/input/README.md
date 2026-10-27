# Prepared training and prediction inputs

Detailed field types, values and direct references: [SCHEMA.md](SCHEMA.md).
The saved identifier is `fredis.annotated-spectrum-training`, without a version.
Prepared data and models must be regenerated when their contract changes.

`SpectrumPredictionData` owns the common input tables. `SourceReactionStructure`
adds observed spectra, peak assignments, derived intensity targets and prepared
peak/depth BranchTeacher reaction history graphs for training.
`training_data.for_prediction()` returns the common input type with no annotations,
targets or branch labels. Prediction uses `ReactionStructureBuilder.build_prediction()`
to construct the same components without measured peaks.

| Component | Field | Meaning |
|---|---|---|
| SourceCompound | `source_compounds` | Packed source graphs, SMILES, symbol vocabulary and `compound_index` into CompoundBatch |
| ReactionTensorData | `reactions` | Concrete reactions with Source/atom references and reaction conflicts |
| ReactionSet | `reaction_sets` | Interned sorted sets of global reaction IDs, including one shared empty set |
| CompoundBatch | `compounds` | Unique mmkit Compound objects, molecular graphs and generating ReactionSet CSR |
| AdductData | `adducts` | Main/fragment adduct registries, their CSR relation and hydrogen shifts |
| BranchGroup | `branch_groups` | Unique (Source, main adduct) pairs, independent of CE |
| CompoundReactionGraph | `compound_graph` | Shared source Compound, target Compound and reaction edge columns |
| ReactionContextData | `reaction_contexts` | Root Source and selected-edge CSR per BranchGroup; optional aligned int8 labels |
| FormulaData | `formulas` | Integer elemental composition/charge, theoretical m/z and validity |
| FormulaExplanationData | `formula_explanations` | Formula, Compound, fragment adduct and hydrogen-shift references |
| SpectrumExplanationData | `spectrum_explanations` | Candidate explanation CSR per spectrum, including unobserved candidates |
| SpectrumConditions | `conditions` | BranchGroup, float32 CE and observed precursor adduct label per spectrum |
| AnnotatedSpectrumData | `annotations` (training) | SpectrumData measurements and PeakAssignmentData explanation CSR per peak |
| BranchTeacher | `branch_teacher` (training) | Shared execution graph, exact history labels, peak reaction history graphs, depth bags and negative transition membership |
| SpectrumFormulaTarget | `targets` (training) | Derived (spectrum, Formula) observation, intensity and mask |

Compound structures are shared across histories and adduct groups. Source roots
reference SourceCompound, which maps to CompoundBatch. ReactionSet IDs represent
sets, not paths. No descriptor values are prepared or stored; selected descriptors
are evaluated by the intensity model only when configured.

## Preparation, training and prediction

Preparation enumerates valid chemistry and peak explanations, interns the shared
tables, and labels each group's selected edges: 1 positive, 0 weak negative,
-1 excluded. These diagnostic Compound-edge labels pool evidence across the
group's CE measurements; BranchTeacher stores the exact-history supervision.
Spectral targets remain specific to each measured spectrum. A formula can have
several structural explanations, and one peak can have several assigned formulas.

The branch network scores reaction-history states against concrete reaction
keys with root-molecule and main-adduct conditioning, without CE. Preparation
retains the original peak-to-path assignment as a per-peak reaction history graph referencing shared
execution states. All alternative structures explaining the same observed peak
share one MIL unit. Every depth stores candidate prefix nodes and the negative
outgoing transitions from the union of their parents. Other observed peaks'
positive transitions are excluded from negatives. Terminal negatives form
next-depth bags; an unassigned branch group can retain a negative-only root unit.

Training reads these memberships directly, computing cumulative path scores by
reaction history graph partition functions. `negative_weight`, MIL temperature, class balance,
per-depth means, and per-peak means are applied at training time. No path search
or teacher reconstruction runs in the loss. The saved execution graph contains
only integer references; molecule graphs and state features remain shared.
Prediction derives its own admissible graph without accessing BranchTeacher.

Intensity contributions from explanations of the same Formula are summed before
comparison with the maximum assigned measured intensity for that (spectrum,
Formula). Valid unmatched candidates receive zero; invalid candidates are masked.
The objective combines distribution cross entropy and cosine loss, both with and
without the precursor. `observed` is target metadata, not a separate presence head.

`generator.predict()` and `predict_batches(..., max_samples=...)` prepare teacher-free
inputs, share Source/reaction encodings across conditions, select reachable edges,
and predict spectra. Batched outputs expose `sample_input_index` to restore original
input order. `.select_samples()` supports reordering/repetition; `.from_structures()`
collates inputs while remapping shared registries and foreign keys.

Workbench opens the saved training structure to show the Compound graph and
positive/weak-negative/excluded edges per BranchGroup. Selecting a CE measurement
shows its measured intensities, peak explanations and corresponding Compounds.

`_reaction_preparation.py`, `_preparation_histories.py` and
`_annotation_preparation.py` are transient chemistry-building helpers. Their
history/occurrence tables are not the persisted model input or public schema.
Neural forward passes use prepared tensors without running reaction chemistry.

Weak negatives are limited to admissible outgoing edges whose source Compound
is the precursor root or an endpoint of a positive edge in the same branch group
(including positive terminal nodes). Positive
evidence is pooled across CE measurements before selecting these sources.
By default, preparation saves only positive/weak-negative edges, their endpoint
Compounds, and source roots (including isolated roots). Use `--save-all-edges`
(Workbench: **Save all candidate edges and nodes**, config `saveAllEdges`) to
retain the complete candidate graph. Additional edges have label -1 and do not
change branch supervision or its normalization. Spectrum candidate coverage can
change with graph retention; this option guarantees identical branch labels,
not identical spectrum losses.
