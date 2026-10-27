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

A neutral-loss precursor can resolve to a registered main-adduct rule through
precursor chemistry. `precursor_candidate_max_reaction_count` controls that
resolution search separately from fragment-tree reaction depth. An unsupported
precursor is reported rather than silently assigned to a different learned bank.

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
