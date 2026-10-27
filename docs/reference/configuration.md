# Configuration reference

FREDIS separates chemical definitions, neural architecture, optimization, and
inference limits. Their files can share the `.pft` suffix; the owning workflow
and document contents determine their meaning.

## Configuration ownership

| Configuration | Chosen during | Contains |
|---|---|---|
| Fragmenter / chemistry | Preparation | SMARTS, ion rules, depth, mass tolerance, symbols |
| Reaction-model selections | Preparation | Imported pattern source files and definitions |
| Model architecture | Fresh training | Molecular/reaction/Transformer/GAT/head dimensions |
| Training settings | Training or resume | Epochs, batches, optimizer, validation and loss weights |
| Generation settings | Prediction, or saved defaults | Branch path threshold, structure budget, output threshold |

Dataset preparation bundles imported models. Fresh training inherits their
architectures and decides new trainable architectures. Resume restores the
checkpoint's architecture and optimizer state.

## Shared training arguments

`fredis train fragment-tree` resolves the following shared settings into the
existing detailed training arguments. The defaults below belong to the shared
command; the component tables later in this page describe detailed/API defaults.

| Shared flag (default) | Expanded detailed settings |
|---|---|
| `--hidden-dim 128` | `node-dim`, `graph-dim`, `reaction-hidden-dim`, `reaction-main-adduct-dim`, `adduct-embedding-dim`, `collision-energy-feature-dim`, each new `pattern-hidden-dim` |
| `--num-heads 4` | `num-heads`, `reaction-num-heads`, `reaction-transformer-heads`, `state-set-heads`, each new `pattern-num-heads` |
| `--num-layers 2` | `num-layers`, `spectrum-gat-layers`, `state-set-layers`, each new `pattern-num-layers` |
| `--ffn-mult 4` | `ffn-mult`, `reaction-transformer-ffn-mult`, `spectrum-gat-ffn-mult`, `state-set-ffn-mult`, each new `pattern-ffn-mult` |
| `--max-degree 4` | Molecular and each new pattern's centrality limit |
| `--max-spatial-dist 4`, `--max-edge-dist 4` | Molecular and each new pattern's atom-distance/path limits |
| `--reaction-encoder-layers 2`, `--reaction-decoder-layers 2` | Reaction Transformer encoder/decoder depths |
| `--reaction-instance-max-distance 4` | `reaction_transformer.max_reaction_distance` |
| `--fc-layers 3`, `--fc-mult 2` | `ce-fc-dims`, `fragment-fc-dims`, `state-fc-dims` as `[128,256,128]` |

`num-heads` divides `hidden-dim`. The one-hop reaction memory GNN and single-head
spectrum GAT keep their architecture. Shared settings apply to new patterns;
frozen imports retain their architecture. For imports, the shared molecular
settings and reaction output width must match the saved model. Use
`fragment-tree-detailed` for independent dimensions or architecture documents.
Resume always restores the saved architecture.

## Model architecture envelope

Architecture files have a `model_config` object containing these components:

```json
{
  "model_config": {
    "mol_encoder_params": {
      "node_dim": 16, "graph_dim": 64, "num_layers": 2,
      "num_heads": 4, "ffn_mult": 4, "max_degree": 4,
      "max_spatial_dist": 3, "max_edge_dist": 3
    },
    "reaction_model_params": {
      "hidden_dim": 128, "num_heads": 4,
      "branch_main_adduct_dim": 128,
      "reaction_transformer": {
        "encoder_layers": 2, "decoder_layers": 2,
        "num_heads": 4, "ffn_mult": 4, "max_reaction_distance": 4
      },
      "pattern_model_params": {}
    },
    "spectrum_model_params": {
      "fragment_fc_dims": [128, 64], "state_fc_dims": [128, 64],
      "gat_num_layers": 2, "state_set_num_heads": 1, "state_set_num_layers": 2,
      "collision_energy_dim": 16, "ce_fc_dims": [32], "ce_center": 0, "ce_scale": 100
    }
  }
}
```

The empty `pattern_model_params` uses pattern defaults. Supplying explicit
pattern IDs selects per-pattern overrides. Reaction definitions and import
sources come from the prepared dataset, rather than this architecture envelope.

## Molecular features

| Parameter | Default | Meaning |
|---|---:|---|
| `node_dim` | 16 | Atom/attention hidden width |
| `graph_dim` | 64 | Whole-molecule readout width |
| `num_layers` | 2 | Molecular Graphormer blocks |
| `num_heads` | 4 | Divides `node_dim` |
| `ffn_mult` | 4 | Molecular FFN intermediate width multiplier |
| `max_degree` | 4 | Highest degree bucket; larger represented degrees are clipped |
| `max_spatial_dist` | 3 | Molecular shortest-path encoding limit |
| `max_edge_dist` | 3 | Molecular bond-path encoding limit |
| `external_feature_dim` | 0 | Optional API-supplied whole-molecule feature width |

Graph token count is `ceil(graph_dim / node_dim)`; `graph_dim` need not be a
multiple of `node_dim`. External features require explicit values for every
encoded source and fragment. Workbench uses external width 0. Spectrum descriptors
are a separate spectrum-only option, rather than external MolEncoder features.

## Reaction atom encoder for each pattern

These fields live in `reaction_model_params.pattern_model_params["ID"]`.

| Parameter | Default | Meaning |
|---|---:|---|
| `hidden_dim` | Shared reaction width (128 normally) | Query, molecular memory, and FFN hidden width |
| `num_heads` | Reaction `num_heads` (4 normally) | Divides the pattern's hidden width |
| `num_layers` | 2 | Repeated cross attention + FFN blocks |
| `ffn_mult` | 4 | FFN intermediate width multiplier |
| `max_degree` | **4** | Centrality clipping bucket |
| `max_spatial_dist` | 4 | Reactant-atom-to-all-atom distance limit |
| `max_edge_dist` | 4 | Bond-path encoding limit |

Detailed CLI overrides use `--pattern-PARAMETER ID=VALUE` with underscores changed to
hyphens. Each pattern has independent parameters, even under Workbench Shared
settings. All readouts produce the shared `reaction_model_params.hidden_dim`.
Distance limits control structural encodings, rather than restricting attention
to a local window. The one-hop memory GNN always uses one message-passing step.

## Reaction memory and state decoding

`reaction_model_params.reaction_transformer` accepts:

| Parameter | Default | Meaning |
|---|---:|---|
| `encoder_layers` | 2 | Source reaction memory blocks |
| `decoder_layers` | 2 | Applied-set state decoder blocks |
| `num_heads` | Reaction `num_heads` | Divides the common reaction width |
| `ffn_mult` | 4 | Memory/decoder FFN multiplier |
| `max_reaction_distance` | 4 | Minimum substructure-distance encoding cutoff |

The CLI name is `--reaction-instance-max-distance`. Distance is the minimum
source-molecule bond distance between the two matched atom sets. Overlap is 0;
farther or disconnected pairs use an overflow bucket. It is separate from the
atom encoders' `max_spatial_dist`. Older checkpoints using `max_spatial_distance`
are accepted as an alias; conflicting old/new values are rejected. Every tied nearest SMARTS
endpoint is projected to the common reaction width and summed. No additional
endpoint-width setting is needed.

## Spectrum prediction and energy

| Field in `spectrum_model_params` | Default | Meaning |
|---|---|---|
| `gat_num_layers` | 2 | Sparse fragment GAT blocks |
| `gat_ffn_mult` | 4 | Spectrum GAT FFN intermediate width multiplier |
| `state_set_ffn_mult` | 2 | State SetTransformer FFN intermediate width multiplier |
| `fragment_fc_dims` | `[128,64]` | Fragment Head hidden widths |
| `state_fc_dims` | `[128,64]` | State Head hidden widths |
| `state_set_num_heads` | 1 | Applied-set SetTransformer heads; divides reaction width |
| `state_set_num_layers` | 2 | Applied-set SetTransformer layers |
| `adduct_embedding_dim` | 32 | Branch main-adduct embedding width |
| `collision_energy_dim` | 16 | CE feature output width |
| `ce_fc_dims` | `[32]` | CE MLP hidden widths; empty selects a Linear map |
| `ce_center`, `ce_scale` | 0, 100 | Normalization `(CE_eV - center) / scale` |
| `descriptors` | `[]` | Selected spectrum-only molecular descriptors |

The shared command generates symmetric FC hidden widths from H, layer count and
multiplier: `H * mult ** min(i, layers - 1 - i)`. Thus H=256, layers=3, mult=2
produces `[256,512,256]`. Zero selects an empty hidden list and a direct Linear
map. The detailed command accepts separate `--ce-fc-layers/--ce-fc-mult`,
`--fragment-fc-layers/--fragment-fc-mult` and `--state-fc-layers/--state-fc-mult`;
the base H is `--reaction-hidden-dim`. An explicit layer count takes precedence
over that network's raw `--*-fc-dims`. Each head has a final output Linear in
addition to these hidden layers.

`ce_scale` must be positive. The CE MLP uses ReLU hidden layers and no dropout
or final LayerNorm. Fragment and State Head banks are independent for each
main adduct and share their weights across that bank's fragments.

## Shared dropout and optimization

The training CLI defaults to `--dropout 0.5`. The Python generator API defaults
to 0. Newly trained neural networks use the run's shared rate; imported frozen
reaction modules preserve their saved architecture/rate. Evaluation disables
dropout.

Loss settings such as `branch_weight`, `negative_weight`, `intensity_loss_weight`,
`cosine_loss_weight`, and `intensity_exponent` affect the training objective.
They are not primitive atom features or chemical preparation fields. Complete
CLI defaults are generated in the [command reference](cli.md).

## Preserve the resolved configuration

Use the checkpoint's `model_config` and `fragment-tree.training-config.pft` to
recover an effective run. Architecture choices inferred from an old command
line can miss imported modules, pattern-ID remapping, or saved overrides.
[Fine-tuning](../guides/fine-tuning.md) and [resume](../guides/training.md#resume-optimization)
have different ownership rules.
