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
    "--train-dir", "data/train_structures",
    "--val-dir", "data/validation_structures",
    "--epochs", "20",
    "--device", "cuda"
  ],
  "parameters": {
    "--lr": [0.0001, 0.0003],
    "--batch-size": [4, 8]
  }
}
```

Alternatively supply `trials`, a list of `{config, args}` records. Each config
must include its outputDir; args must begin with
`["-m", "fredis.cli", "train", "fragment-tree"]` and specify that same output
via `--output-dir`. This is the format produced by Workbench's architecture
search planner, including per-pattern Reactant Graphormer parameters.
`--plan-json` accepts the same JSON inline.

The orchestrator invokes the normal training CLI in a separate process for each
trial, using the same Python interpreter. Outputs must be distinct, fresh
subdirectories of outputDir; the complete Cartesian product of candidate values is run, with no truncation. A failed trial
is recorded and later trials continue. The overall exit code is nonzero if any
trial fails. Do not reuse an existing search output directory.

`fragment-tree.tuning.pft` is a versioned `fredis.tuning-result` JSON document.
It records trial configurations, CLI arguments, status, timestamps, exit codes,
and **all numeric completed-epoch validation observations**, not just the selected
score. It also merges the per-epoch `metric_distributions.tsv` observations, preserving
validation cosine Q1 (q25), Q3 (q75), median, mean and other recorded distributions.
For every statistic it saves maximum and final-epoch values; loss statistics
also retain minimum values. Maxima are calculated independently for each
statistic across completed epochs. Last refers to the final completed epoch,
with missing values left empty rather than taken from an earlier epoch.
The result is written atomically before and after each trial.
`tuning-plan.json` retains the submitted search. `tuning-scores.tsv` provides
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
not a search dimension. Per-pattern Reactant Graphormer architecture parameters
remain eligible for tuning; they do not change the dataset reaction definitions.
