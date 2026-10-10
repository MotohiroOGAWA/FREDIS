# Fragment-tree branch and intensity training

## Prediction workflow

1. Group spectra by source molecule and normalized main adduct. Collision
   energies in a group share one branch search.
2. Encode the source graph and all primitive reactions once.
3. Starting at Source, score valid next reactions with the current reaction-set
   representation and main-adduct embedding.
4. Accumulate branch probability in log space, deduplicate normalized states,
   and stop at the path threshold, reaction limit, node budget, or a state with no
   valid continuation.
5. Materialize the selected fragment tree outside neural training.
6. Generate physical-ion candidates with current FREDIS ion rules. Equivalent
   final ions share one candidate while retaining all `(adduct, hydrogen shift)`
   explanations.
7. Encode the fragment tree and aggregate equivalent explanations with
   normalized attention. Predict independent physical-ion intensities using
   separate main-adduct and collision-energy features.
8. Max-normalize raw intensities for the final spectrum.

## Main settings

| Argument | Default | Purpose |
|---|---:|---|
| `--reaction-main-adduct-dim` | 128 | Width of the Branch Scorer's main-adduct input |
| `--main-adduct-embedding-dim` | 128 | Width of the dedicated main-adduct conditioning path |
| `--collision-energy-feature-dim` | 128 | Width of the intensity-only collision-energy features |
| `--branch-weight` | 1 | Top-level branch-loss weight |
| `--negative-weight` | 0.2 | Weak-negative term inside branch loss |
| `--branch-mil-temperature` | 0.1 | Smooth-max temperature for alternative positive paths |
| `--fragment-loss-weight` | 0.5 | Class-balanced fragment BCE weight |
| `--state-loss-weight` | 0.5 | Class-balanced ion-state BCE weight |
| `--intensity-loss-weight` | 1 | Physical-ion intensity-loss weight |
| `--cosine-loss-weight` | 0.5 | Spectrum cosine term added to log intensity MSE |
| `--intensity-power` | 0.5 | Target/prediction power used only during loss calculation |
| `--precursor-free-loss-weight` | 0.5 | Precursor-free spectrum-loss weight |

Loss settings are training settings: they are stored in the checkpoint's
training settings, not in `model_config`, and a resumed run keeps them.

### Generation settings

`generation_params` is a separate `model_config` section. It is saved with the
model, used by validation, and used by prediction unless a prediction option
overrides it.

| Argument | Default | Purpose |
|---|---:|---|
| `--branch-path-threshold` | 0 | Minimum cumulative path probability; 0 disables pruning |
| `--max-fragment-nodes` | 100 | Shared node budget per branch group |
| `--ion-prediction-threshold` | 0.5 | Minimum physical-ion confidence for an output peak |
| `--peak-intensity-threshold` | 0 | Minimum max-normalized intensity for an output peak |

`python -m fredis.ml.specgen.predict_spectrum` accepts the same four options;
when omitted, the values saved with the model are used.

`max_reaction_count` and chemistry configuration are inherited from prepared data.
The Workbench shows architecture (Mol Encoder, Reaction Model, Spectrum Model)
and Generation as separate sections; loss weights are in Optimization and Loss.

## Supervision

Preparation unions teacher paths across every collision energy in a branch
group. For one peak, alternative paths use normalized smooth-max MIL. At every
teacher-supported depth, chemically valid reactions absent from all supported
continuations become weak negatives. Thus a fragment missing at low collision
energy is not marked negative when another spectrum in the group supports it.

The intensity objective uses one output for both full-spectrum and
precursor-free losses, plus physical-ion assignment supervision. Several ion
states can have non-zero intensity simultaneously. There is no hydrogen-shift
classification loss and no sum over equivalent explanation intensities.

## Prepared-data boundary

Preparation stores source and fragment graphs, primitive reactions, transition
states, valid/positive/weak-negative reactions, peak-to-path MIL indices, tree
connectivity, formulas, masses, physical candidates, explanation indices, and
targets. Training performs embedding lookup, gather/scatter, attention,
transformers, loss, and backpropagation only. Prepared files are regenerated
when this contract changes.

Regression coverage includes CE-shared branch trees, cumulative probability,
multi-depth negatives, cross-CE protection, equivalent-candidate grouping,
candidate-count invariance, simultaneous ion peaks, CE-dependent intensities,
and a strict forward/loss/backward test with chemistry entry points disabled.
