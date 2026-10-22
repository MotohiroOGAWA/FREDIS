# Prepared training and prediction inputs

Detailed field types, values and direct references: [SCHEMA.md](SCHEMA.md).
The saved identifier is `fredis.annotated-spectrum-training`, without a version.
Prepared data and models must be regenerated when their contract changes.

`SpectrumPredictionData` owns the common input tables. `SourceReactionStructure`
adds observed spectra, peak assignments and derived intensity targets for training.
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
| SpectrumFormulaTarget | `targets` (training) | Derived (spectrum, Formula) observation, intensity and mask |

Compound structures are shared across histories and adduct groups. Source roots
reference SourceCompound, which maps to CompoundBatch. ReactionSet IDs represent
sets, not paths. No descriptor values are prepared or stored; selected descriptors
are evaluated by the intensity model only when configured.

## Preparation, training and prediction

Preparation enumerates valid chemistry and peak explanations, interns the shared
tables, and labels each group's selected edges: 1 positive, 0 weak negative,
-1 excluded. Branch labels pool evidence across that group's CE measurements.
Spectral targets remain specific to each measured spectrum. A formula can have
several structural explanations, and one peak can have several assigned formulas.

The branch network scores source/target Compound features, concrete reaction
features and main-adduct features, without CE. Training applies weighted edge BCE;
excluded edges contribute no loss. `negative_weight` controls weak-negative weight.
The intensity model uses CE and a finite runtime expansion of admissible reaction
histories. A reaction cannot be repeated and incompatible additions are rejected.
This prevents structural cycles or merged Compound identities from introducing
invalid paths. Structures remain shared; the expansion is derived, not serialized.

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
has at least one positive outgoing edge in the same branch group. Positive
evidence is pooled across CE measurements before selecting these sources.
By default, preparation saves only positive/weak-negative edges, their endpoint
Compounds, and source roots (including isolated roots). Use `--save-all-edges`
(Workbench: **Save all candidate edges and nodes**, config `saveAllEdges`) to
retain the complete candidate graph. Additional edges have label -1 and do not
change branch supervision or its normalization. Spectrum candidate coverage can
change with graph retention; this option guarantees identical branch labels,
not identical spectrum losses.
