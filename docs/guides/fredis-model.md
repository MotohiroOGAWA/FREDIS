# FREDIS model overview

FREDIS connects explicit reaction definitions to a learned spectrum generator.
A Fragmenter determines source reaction matches, compatible sets, fragment
structures, ion conditions, and assignment tolerance. A checkpoint learns
branch selection and CE-conditioned physical-ion probabilities on that chemistry.

## Three boundaries

| Boundary | Responsibility | Details |
|---|---|---|
| Chemistry and preparation | Define candidates and observed-peak supervision | [Chemistry](../concepts/chemistry.md) · [Preparation](data-preparation.md) |
| Learned model | Encode molecules/reactions, select histories, and distribute ion probability | [Architecture](../concepts/architecture.md) · [Training](training.md) |
| Inference and inspection | Apply a checkpoint to structures and preserve provenance | [Prediction](prediction.md) |

Reaction patterns are externally definable SMARTS. They identify transformations
on the original molecule; compatible sets can share materialized structures
without merging distinct histories. Fragment-ion adducts and hydrogen shifts
translate molecular fragments into formula/mass candidates.

The molecular and reaction encoders consume all atom/bond channels defined by
MolGraphBuilder. Pattern-owned cross attention uses reactant atoms as Query and
all source atoms as Key/Value, with centrality, spatial, and edge encodings.
Local reactions feed a branch Transformer and, independently, the spectrum
GAT/heads. Main-adduct selection is separate from CE-conditioned intensities.

Fresh training learns the full model. [Fine-tuning](fine-tuning.md) imports and
freezes complete local reaction features while training new patterns, MolEncoder,
and downstream networks. Resume continues the same optimizer and architecture.
The [configuration reference](../reference/configuration.md) records the current
parameter names and ownership.
