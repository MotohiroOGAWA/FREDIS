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
the orchestrator does not truncate or prune it. Choose a new output root for a new search; reuse the same plan and root to resume.
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
| `fragment-tree.tuning-plan.pft` | Submitted search definition |
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

`fragment-tree.tuning-plan.pft` is the saved search plan and editable Workbench
configuration. Workbench plans retain the original form settings and candidate
order alongside executable trials. Click the file to open the Training Workbench;
editing and saving regenerates the trials. Choose a fresh output directory when changing the plan. Reopen the same plan
and keep the same output directory to continue an interrupted search. No separate tuning-plan.json or tuning-config file is
written. The CLI still accepts legacy JSON search plans through `--plan`, and
Workbench can restore legacy explicit trial plans through Load Configuration.

## Resume an interrupted search

Run the saved plan again, or open it in Workbench and start Tuning with the same
settings and output directory (keep **New model** selected; ordinary single-model
Resume mode is separate):

```bash
python -m fredis.cli train tune-fragment-tree --plan runs/search/fragment-tree.tuning-plan.pft
```

The orchestrator checks the saved plan against the result report and
`tuning-scores.tsv`. A row is completed only when its status is `completed` and
exit code is zero; a `running`, `failed` or missing row does not skip the trial.
Completed trial reports can recover scores if the orchestrator was interrupted
before committing its summary. Completed trials are retained and skipped.

Unfinished trials resume from `last.pt`; a valid periodic checkpoint is used if
`last.pt` is missing or corrupt. The checkpoint restores architecture, weights,
optimizer, scheduler, random state and training history. Only the remaining
planned epochs run, since the ordinary training CLI counts resumed `--epochs`
as additional epochs. Trials already at the epoch limit or early stopping
threshold are marked completed without extra training. Unattempted trials start
normally. New checkpoint writes use atomic replacement.

A non-empty trial directory without a usable checkpoint is reported failed,
with its files retained; progress before the first checkpoint cannot be restored.
Changed plans are rejected for an existing search, and a directory lock prevents
concurrent orchestrators from writing the same search outputs. The original plan
arguments and actual continuation arguments are both retained in the result.
