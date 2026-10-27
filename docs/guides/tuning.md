# Tune model and optimization parameters

The tuning orchestrator runs ordinary training CLI trials in separate processes.
The prepared dataset fixes the chemical definitions. Search dimensions can change
optimization settings and trainable neural architectures, rather than redefine
reactions or frozen imported modules.

## Define a small grid

Save this as `search.json`:

```json
{
  "outputDir": "runs/search",
  "baseArgs": [
    "--dataset-dir", "data/prepared",
    "--epochs", "20",
    "--device", "cuda"
  ],
  "parameters": {
    "--lr": [0.0001, 0.0003],
    "--batch-size": [4, 8]
  }
}
```

```bash
python -m fredis.cli train tune-fragment-tree --plan search.json
```

The default training command is `fragment-tree`, so shared settings such as
`--hidden-dim`, `--num-layers` and `--ffn-mult` can also be searched. Set
`"trainingCommand": "fragment-tree-detailed"` in the plan for independent
component flags or architecture documents. Workbench selects the corresponding
command from its architecture mode.

This grid creates four trials. Candidate lists form a complete Cartesian product;
the orchestrator does not truncate or prune it. Choose a new search output root.
Each trial receives a distinct subdirectory, and a failed trial is recorded while
later trials continue. The overall command reports failure if any trial fails.

`--plan-json` accepts the same document inline. Advanced plans can provide explicit
`trials` with configuration and full training argument arrays; Workbench uses this
form after expanding its architecture controls.

## Tune pattern architecture in Workbench

On Training, enable Tuning mode and enter candidates beside the required setting.
The default **Shared settings** architecture searches common dimensions, depths,
FFN multipliers and FC settings. Select **Detailed settings** to search component
parameters; its reaction-only **Shared** choice expands one pattern architecture
to all trainable patterns, while **Individual** searches patterns separately.
Switching the global mode removes candidates that no longer belong to that mode.
The learned weights remain separate in both cases.

Pattern parameters include `hidden_dim`, `num_layers`, `num_heads`, `ffn_mult`,
`max_degree`, `max_spatial_dist`, and `max_edge_dist`. Hidden widths must remain
divisible by head counts. Frozen imported patterns keep their saved architecture
and are excluded from editing/search.

## Read the results

| File | Contents |
|---|---|
| `tuning-plan.json` | Submitted search definition |
| `fragment-tree.tuning.pft` | Trial settings, statuses, timestamps, exit codes, and validation observations |
| `tuning-scores.tsv` | Trial comparison columns |
| Trial subdirectories | Ordinary checkpoints, training results, spectra, and logs |

The result retains numeric observations from completed validation epochs,
including cosine/loss distribution summaries. Q1/Q3/median/mean describe spectra
within an epoch. Maximum, final-epoch, and loss minimum values are distinct
statistics; a missing final observation stays missing.

Open `fragment-tree.tuning.pft` in Workbench to select a metric/statistic, inspect
trial details, and open its training report. The viewer's ranking choice does
not change which trials execute. Compare prediction behavior and held-out data,
rather than treating a short tuning run as an accuracy claim.
