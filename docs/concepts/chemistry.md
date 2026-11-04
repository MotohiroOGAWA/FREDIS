# Source reactions, fragments, and ion conditions

FREDIS separates a chemical search definition from a learned scoring model.
The Fragmenter defines which transformations and ion conditions can be proposed.
The model learns their selection and intensities from assigned experimental peaks.

## A pattern, a match, and a history are different objects

| Object | Meaning | Example |
|---|---|---|
| Reaction pattern | Reactant/product SMARTS defining one transformation family | A mapped two-atom single-bond query |
| Concrete reaction | A particular match and product choice on one source molecule | The query applied to atoms 2 and 5 |
| Reaction set | A canonical sorted set of concrete reaction IDs | `{3, 7}` |
| Compound | A unique materialized molecular structure | A canonical-SMILES identity shared by several histories |
| Execution state | A source/main-adduct group, Compound, and applied ReactionSet | Two histories can share a Compound but have different valid extensions |
| Formula explanation | A Compound combined with an ion adduct and hydrogen shift | Several explanations can produce the same formula |

Pattern IDs refer to definitions; concrete reaction IDs refer to source-specific
instances. IDs are local to their registry and can be remapped when datasets or
models are combined. SMARTS atom positions preserve query order and are distinct
from global source atom indices.

## Define a reaction pattern

```json
{
  "name": "single_bond_cleavage",
  "reactant_smarts": "[!#1:1]-[!#1:2]",
  "products": [
    {"name": "retained_side", "smarts": "[!#1:1]"}
  ]
}
```

Mapped atoms establish correspondence between the reactant and retained product.
The complete pattern set lives under
`fragment_ion_tree_builder.reaction_pattern_set`. Use Workbench's visual or text
reaction editor to build and inspect queries before preparing large datasets.
A SMARTS match is a proposed transformation, rather than evidence that a measured
peak follows that mechanism.

## Compatible source-anchored transformations

Concrete reactions are found on the original source. Preparation identifies
conflicts, invalidations, and redundant effects. Valid reaction sets must also
retain a nonempty compatible atom set; pairwise validity alone does not guarantee
validity of the complete set.

```{mermaid}
flowchart LR
    SOURCE["Original source"] --> R1["Concrete match r1"]
    SOURCE --> R2["Concrete match r2"]
    R1 --> SET["Compatible set: r1 + r2"]
    R2 --> SET
    SET --> STRUCTURE["Materialized Compound"]
    STRUCTURE --> ION["Ion adduct + hydrogen shift"]
    ION --> FORMULA["Formula / theoretical m/z"]
```

Equivalent materialized structures share one Compound entry and molecular
encoding. Their admissible histories remain distinct for next-reaction choices
and probability propagation. This sharing prevents repeated neural computation
without merging incompatible paths.

## Main adduct, fragment ion, and hydrogen shift

A main-adduct rule associates a precursor condition with permitted fragment-ion
adducts. The configured hydrogen-shift range expands each ion choice:

```json
{
  "name": "positive_H",
  "adduct_type": "[M+H]+",
  "fragment_ion_adducts": ["[M-H]+", "[M+H]+"]
}
```

With `hydrogen_shift_min=-5` and `hydrogen_shift_max=0`, two ion choices produce
12 ion/shift conditions per molecular structure. This grid exists before formula
validity and positive m/z filtering. The State Head has a slice for each condition
within its main-adduct bank.

`adduct_type` describes the observed precursor relative to the original Source.
It is not an instruction to repeat a loss on every fragment. For example:

```json
{
  "name": "pos_H_H2O_loss",
  "adduct_type": "[M+H-H2O]+",
  "fragment_ion_adducts": ["[M+H]+"]
}
```

The target precursor formula is Source + H − H2O. Precursor resolution searches
for actual reaction-product structures that reach that formula under `[M+H]+`
and an allowed hydrogen shift. It applies the fragment ion to the precursor too.
The original Source remains the structural root; a loss must be represented by
reaction chemistry, and its product must occur on the recorded ancestor path of
an annotated fragment. Formula subtraction alone does not establish a loss site.
If no such product exists, the precursor is unresolved. Do not include
`[M+H-H2O]+` as a blanket fragment-ion condition for this rule: it would allow an
additional loss independently of structural ancestry.

Training and prediction initialize spectrum probability on the matching
precursor execution histories, not on the untransformed Source. Only their
descendants can emit nonzero spectrum intensity. Multiple matching histories
receive equal initial probability; Source stays available as model context and
bookkeeping. Ion/hydrogen-shift states and precursor identity are resolved during
preparation, before neural scoring.

Only adduct types listed in the adduct rule set are accepted. A loss adduct such
as `[M+H-H2O]+` must have its own rule like the one above; it is not resolved to
the main adduct it contains (`[M+H]+`), and records with an unlisted adduct are
reported as invalid during Training Data preparation. Model-guided prediction
never prunes the histories that reach a listed loss precursor.

Register the intended precursor types explicitly in the adduct rule set. Precursor
resolution uses the same `max_reaction_count` budget as fragment-tree generation;
there is no separate precursor reaction limit. An unresolved precursor is reported
rather than synthesized by subtracting a neutral loss from every fragment.
Regenerate prepared datasets and retrain models after switching precursor or
fragment-ion rules; changing labels does not add learned chemistry to old checkpoints.

## Mass and collision energy

Formula and theoretical m/z come from mmkit chemistry. They annotate and group
predictions and connect candidates to measured peaks. The neural model does not
regress m/z from hidden features.

Collision energy conditions spectrum intensities. Branch selection is CE
independent; spectra at different energies share source chemistry and primitive
reaction features while retaining separate spectrum probabilities. The input
adapter parses supported energy representations, and the CE MLP normalizes its
eV input using saved center/scale settings.

See [architecture](architecture.md) for the learned transformations and
[data preparation](../guides/data-preparation.md) for assignment/search controls.
