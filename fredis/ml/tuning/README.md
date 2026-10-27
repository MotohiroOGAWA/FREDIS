# Fragment-tree parameter tuning

Run the dedicated orchestrator:

```bash
python -m fredis.cli train tune-fragment-tree --plan search.json
```

`search.json` can define a grid over ordinary training CLI flags:

```json
{
  "outputDir": "runs/search-001",
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

Grid plans default to `fragment-tree` and accept shared architecture flags.
Set `"trainingCommand": "fragment-tree-detailed"` for independent component
flags and architecture files.

Alternatively supply `trials`, a list of `{config, args}` records. Each config
must include its outputDir; args must begin with
`["-m", "fredis.cli", "train", "fragment-tree"]` or use `fragment-tree-detailed`
in the last position, and specify that same output
via `--output-dir`. This is the format produced by Workbench's architecture
search planner, including per-pattern atom cross-attention parameters.
`--plan-json` accepts the same JSON inline.

The orchestrator invokes the normal training CLI in a separate process for each
trial, using the same Python interpreter. Outputs must be distinct
subdirectories of outputDir; the complete Cartesian product of candidate values is run, with no truncation. A failed trial
is recorded and later trials continue. The overall exit code is nonzero if any
trial fails. Reuse the identical plan and output directory to resume an interrupted search.

`fragment-tree.tuning.pft` is a `fredis.tuning-result` JSON document.
It records trial configurations, CLI arguments, status, timestamps, exit codes,
and **all numeric completed-epoch validation observations**, not just the selected
score. It also merges per-epoch observations from `metrics/distributions/` (or legacy
`metric_distributions.tsv` logs), preserving
validation cosine Q1 (q25), Q3 (q75), median, mean and other recorded distributions.
For every statistic it saves maximum and final-epoch values; loss statistics
also retain minimum values. Maxima are calculated independently for each
statistic across completed epochs. Last refers to the final completed epoch,
with missing values left empty rather than taken from an earlier epoch.
The result is written atomically before and after each trial.
`fragment-tree.tuning-plan.pft` retains the submitted search. `tuning-scores.tsv` provides
one row per trial and a separate column for every maximum/last/loss minimum.
There is no preselected evaluation criterion or single best configuration.

Open the `.tuning.pft` document in Workbench, or choose Results on the training
job. Its scatter plot supports execution order or score rank on the horizontal
axis, cosine and loss metric selection, and maximum/final-epoch evaluation.
Cosine ranks higher values first; losses rank lower values first and also offer
minimum. These choices affect comparison only and never prune the grid search.
The table shows every recorded cosine/loss comparison column together; Q1/Q3/
median/mean describe validation spectra within an epoch, not the distribution of
epoch averages. Failed and incomplete trials remain in the table with any
available observations but are excluded from chart ranking. Click a point or
Details to inspect its configuration and validation history or open its normal
training report. Viewer choices are retained in editor state.

New preparation outputs use `fragment-tree.training-data.pft`. Readers retain
compatibility with old `fragment-tree.pft` dataset manifests. Prepared tensors
still use `.preft.pt`.

The reaction pattern set is fixed when preparing the training dataset and is
not a search dimension. Per-pattern atom cross-attention architecture parameters
remain eligible for tuning; they do not change the dataset reaction definitions.

Workbench offers **Shared** and **Individual** architecture modes. Shared mode
applies the same architecture settings and tuning candidates to all trainable
patterns. Individual mode edits and tunes each trainable pattern separately.
Imported, frozen patterns retain their saved architectures and are excluded.
Weights remain separately learned for each pattern in both modes.

Tunable parameters are `hidden_dim`, `num_layers`, `num_heads`, `ffn_mult`,
`max_degree`, `max_spatial_dist` and `max_edge_dist`. For example, enable Tune
for `num_layers` and enter `1, 2, 3`. In Shared mode this creates three trials
with the same layer count across trainable patterns. Separate Individual
parameters form independent search dimensions. Workbench expands the selected
candidates into explicit trials before submitting the plan to the orchestrator.

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
