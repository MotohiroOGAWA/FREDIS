# FREDIS model

**Fragmentation with Reactions that are Externally Definable for In Silico Spectra**

FREDIS keeps the existing source-anchored reaction-combination search and
fragment tree, while replacing the learned representations and ion-state
model.

## Fragment ion rules

Hydrogen shifts are configured once per rule set. Each precursor-adduct rule
only lists permitted fragment-ion adducts:

```json
{
  "fragment_ion_adduct_rule_set": {
    "name": "positive_default",
    "hydrogen_shift_min": -5,
    "hydrogen_shift_max": 0,
    "adduct_rules": [{
      "name": "protonated",
      "adduct_type": "[M+H]+",
      "fragment_ion_adducts": ["[M+H]+", "[M-H]+"]
    }]
  }
}
```

`hydrogen_shift_min`, `hydrogen_shift_max`, and each rule's
`fragment_ion_adducts` are required. Every listed fragment-ion adduct applies
to every neutral fragment; charged fragments get no fragment-ion adducts.

Each structural node has independent output states for every fragment-ion
adduct × hydrogen-shift condition. Each main adduct owns a separate transition
head bank; every layer has one weight matrix and bias per output condition.
Heads consume parent/child molecular embeddings, reaction features and CE,
using GELU and configurable dropout. Ion/adduct-shift embeddings and pooling
are not used. Formula-equivalent states are merged only after propagation for
loss/output. In inference, mmkit computes ion formulas and m/z after prediction;
training preparation computes annotations for assigning observed targets. Neither
formula nor m/z controls the transition network or state graph.

## Molecular and reaction encoding

`MolEncoder` accepts `external_feature_dim` (default 0), not descriptor names.
Supply an aligned finite tensor `[num_graphs, external_feature_dim]` through
`model(batch, external_features=features)` or `batch.external_features` (one
`[1, external_feature_dim]` row per graph before batching). Features are projected
with Linear → GELU → Linear and fused with the Graphormer graph embedding by a
final Linear layer. Atom outputs remain structural. Selection, scaling and
missing-value handling belong to the caller; descriptors are one possible source.
Graph construction does not automatically populate external features. Existing
SMILES-only prediction/preparation uses width 0; using a nonzero width requires
supplying features for every source and generated fragment. Legacy MolEncoder
`descriptors` configurations and descriptor-fusion weights require explicit
conversion before loading; they are not silently reinterpreted.

Reaction encoding uses raw atom/bond features, a shared edge-aware local
context GCN, a SMARTS-sized Graphormer per Reaction Pattern, and explicit
before/after transformation features. Reaction Sequence encoding remains
order invariant and reuses the existing combination-search semantics.

## Training

Fresh models are randomly initialized and trained end to end. The standalone
MolEncoder pretraining and frozen-encoder CLI commands are not part of the
FREDIS workflow. Joint-model checkpoint save, resume, initialization, and
prediction loading remain supported.

Prepared tensors use schema `fredis.fragment-tree-training`, version 2.
Reaction preference weights are neutral unless explicitly enabled, and never
remove chemically valid candidates.
