# Training and reaction-asset API

The CLI is the main entry point for dataset discovery, configuration inheritance,
checkpoint resume, and result publication. Direct Python use exposes the same
model and reaction-asset components for integrations.

## Construct a generator from configuration

```python
import json
from pathlib import Path
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator

architecture = json.loads(Path("docs/examples/compact.architecture.pft").read_text())["model_config"]
architecture["fragmenter_params"] = json.loads(Path("docs/examples/chemistry.fragmenter.pft").read_text())
architecture["mol_encoder_params"]["symbols"] = ["C", "N", "O", "P", "S", "F", "Cl", "Br", "I"]
generator = create_spectrum_generator(architecture)
```

This initializes weights. Use [load_generator](prediction.md#load-a-trained-model)
for a trained checkpoint. Prepared input vocabularies and definitions must match
the effective model configuration.

```{eval-rst}
.. autofunction:: fredis.ml.specgen.spectrum_generator.create_spectrum_generator
```

## Train through the prepared-data wrapper

```{eval-rst}
.. autoclass:: fredis.ml.training.fragment_tree_training.model.ReactionFragmentTreeTrainingModel
   :members: forward
   :undoc-members:
   :exclude-members: __new__
```

The wrapper returns branch and spectrum loss/metrics and retains autograd.
Optimization settings, scheduling, checkpoint writing, and epoch validation are
provided by the higher-level training function:

```{eval-rst}
.. autofunction:: fredis.ml.training.fragment_tree_training.training.train_reactions
```

`train_dir` and `val_dir` refer to prepared split directories. The CLI's
`--dataset-dir` resolves them from one preparation root and inherits its chemistry
and import selections. Resume restores optimization state; importing patterns
starts a new run. See the [training](../guides/training.md) and
[fine-tuning](../guides/fine-tuning.md) guides before calling this function directly.

## Inspect and export a complete reaction pattern

```python
from fredis.ml.training.fragment_tree_training.pattern_models import inspect_model, export_pattern

report = inspect_model("runs/demo-model/last.pt")
pattern_id = report["patterns"][0]["pattern"]["pattern_id"]
export_pattern("runs/demo-model/last.pt", pattern_id, "runs/demo.reaction-model.pt")
```

The exported feature asset includes the pattern-specific atom encoder, rule/product
readout, and nearest-endpoint projection together with the required shared
configuration/dependencies. It is selected during destination dataset preparation.

```{eval-rst}
.. autofunction:: fredis.ml.training.fragment_tree_training.pattern_models.inspect_model
```

```{eval-rst}
.. autofunction:: fredis.ml.training.fragment_tree_training.pattern_models.export_pattern
```

```{eval-rst}
.. autofunction:: fredis.ml.training.fragment_tree_training.pattern_models.pattern_config
```

Destination IDs can differ from source IDs. A complete import preserves inherited
features and remaps their parameter namespaces. Checkpoint
`model_config.fine_tuning.pattern_imports` records those bindings.
