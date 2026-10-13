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

Each explanation is `(fragment, fragment-ion adduct, ΔH)`. Explanations with
the same final formula and charge share one physical-ion candidate. Their log
probabilities are summed directly; the fragment log probability is then added
exactly once. Attention is used only to pool explanation embeddings for the
intensity head.

## Molecular and reaction encoding

`MolEncoder.forward` returns graph embeddings only. Optional molecule-level
features use late fusion and an explicit missing-value mask. Training-split
statistics are stored as model buffers and in `feature_configuration`.

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
