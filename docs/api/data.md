# Prepared-data API

Use `SourceReactionStructure` for observed/annotated training input and
`SpectrumPredictionData` for chemistry plus requested conditions. The
[input-schema reference](../reference/input-schema.md) describes stored fields,
CSR relationships, and distinct index spaces.

## Load, inspect, and move a prepared structure

```python
from pathlib import Path
import torch
from fredis.ml.input.source_reaction_structure import SourceReactionStructure

path = next(Path("runs/demo-prepared/train_structures/data").glob("*.preft.pt"))
structure = SourceReactionStructure.load(path)
structure.validate()
print(structure.source_smiles)
print(structure.num_samples)
print(structure.reactions.reaction_type.shape)

device_structure = structure.to("cuda" if torch.cuda.is_available() else "cpu")
prediction_input = structure.for_prediction()
```

The device-transfer line uses CUDA when available. The loaded structure is on CPU;
`for_prediction()` shares chemical registries and an execution graph while
removing observed peaks, intensity targets, and BranchTeacher.

## Select and collate conditions

```python
selected = structure.select_samples([1, 0, 1])
combined = SourceReactionStructure.from_structures([structure, selected])
combined.validate()
combined.save("runs/combined.preft.pt")
```

Selection can reorder and repeat spectra. Collation remaps table references and
teacher memberships, including shared Compounds and ReactionSets. It does not
recompute chemistry or construct a new teacher from structure labels. Input
vocabularies must match.

## Build from a source and measured peaks

The model supplies Fragmenter settings and graph-feature vocabulary. Its weights
are not trained by data preparation:

```python
import torch
from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.specgen.predict_spectrum import load_generator

generator = load_generator(
    model_path="runs/demo-model/last.pt", params_path=None,
    device=torch.device("cpu"), strict=True,
)
builder = ReactionStructureBuilder(generator)
structure, retained = builder.build(
    Compound.from_smiles("CCO"),
    [Adduct.parse("[M+H]+")],
    [20.0],
    [[47.04914, 31.01839]],
    [[1.0, 0.6]],
)
```

`retained` maps retained requested conditions. `build_prediction()` accepts the
source, adducts, and energies without measured peak lists. Use the CLI for
multi-file loading, metadata validation, split handling, subprocess preparation,
and manifest/report publication.

## References

```{eval-rst}
.. autoclass:: fredis.ml.input.structure_builder.ReactionStructureBuilder
   :members: build, build_prediction
   :undoc-members:
   :exclude-members: __new__
```

```{eval-rst}
.. autoclass:: fredis.ml.input.source_reaction_structure.SourceReactionStructure
   :members: load, save, validate, select_samples, from_structures, for_prediction
   :undoc-members:
   :exclude-members: __new__
```

```{eval-rst}
.. autoclass:: fredis.ml.input.source_reaction_structure.SpectrumPredictionData
   :members: validate, select_samples, from_structures, for_prediction
   :undoc-members:
   :exclude-members: __new__
```

`execution_graph` and `spectrum_batch` are parameter-independent runtime views.
Stored BranchTeacher history graphs and memberships remain explicit persisted
supervision. Geometry and packed spectrum caches are excluded from saved input
state and rebuilt for replaced/device-copied objects.
