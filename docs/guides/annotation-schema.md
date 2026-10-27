# Annotated spectra and prepared training data

The detailed [input schema](../reference/input-schema.md) follows processing
order and gives stored fields, shapes, examples and references. Prepared files
use `fredis.annotated-spectrum-training`, without a schema version or migration.

## 1. Source molecules and requested conditions

SourceCompound packs original molecular graphs and identifies each source.
AdductData defines the shared adduct/shift vocabulary. BranchGroup identifies
one source/main-adduct pair; spectra at different collision energies share it.
SpectrumData retains measured peaks, intensity, precursor membership and conditions.

## 2. Reactions, sets and generated structures

ReactionTensorData stores source-specific reaction matches and chemical constraints.
ReactionSet interns sorted concrete reaction-ID sets. CompoundBatch shares each
molecular structure and records its generating sets. CompoundReactionGraph stores
structural transitions; ReactionContextData selects candidates and diagnostic
support labels for each BranchGroup.

## 3. Ion candidates and measured-peak assignments

FormulaData stores elemental composition, m/z and validity.
FormulaExplanationData links formulas to Compounds, ion adducts and hydrogen
shifts. SpectrumExplanationData lists all spectrum candidates, including valid
unobserved ones. PeakAssignmentData retains each measured peak's alternative
explanation IDs using `formula_explanation_ptr/index`.

## 4. Branch and spectrum teachers

[BranchTeacher](../reference/input-schema.md#observations-and-learning-targets) stores a shared
reaction history graph and observed-peak/depth learning units. Each unit retains
target fragment endpoints, supported alternative paths and the corresponding
negative transitions. States distinguish applied ReactionSets even when they
produce the same Compound. Molecular graphs and neural features are shared.

At depth two, a positive path score sums the first and second log probabilities.
The loss computes path MIL, class balance, temperature weighting and depth/peak
averages during training. Preparation saves memberships, not numerical weights.
Another peak's positive transition is excluded from negative candidates.

SpectrumFormulaTarget remains independent: it stores observation flags, maximum
assigned measured intensities and masks per (spectrum, Formula). Predicted
contributions for equivalent formulas are summed before cross-entropy and cosine
losses, including precursor-inclusive and precursor-excluded objectives.

## 5. Persistence, batching and prediction

SourceReactionStructure combines these tables and saves them on CPU in `.preft.pt`.
Its BranchTeacher and history graph are persisted; geometry and `spectrum_batch`
are caches. Loading runs validation without chemistry or teacher reconstruction.
Collation and sample selection remap saved memberships, including repeated spectra.

`training.for_prediction()` returns SpectrumPredictionData without annotations,
intensity targets or BranchTeacher, and clears diagnostic branch labels.
Direct prediction prepares the common chemical tables from requested molecules
and conditions. Existing datasets without BranchTeacher must be regenerated.
