# Installation

FREDIS has a Python package, command-line workflows, and an optional VS Code
Workbench. Install the scientific environment before installing the editable
checkout. Python **3.10 or newer** is required.

## Native dataset library

This fresh source tree includes source snapshots in `upstream/` and chemistry
code in `fredis/libs/mmkit`; no submodule initialization is needed. See
`upstream/versions.json` for the pinned dataset and similarity revisions.
Install CMake, a C++20 compiler, HDF5 development files, `nlohmann-json3-dev`,
and `pyarrow>=20`, then build/install the native Python bindings:

```bash
python scripts/build-native.py
```

For spectrum prediction, build the separate similarity library as well with
`python scripts/build-native.py --similarity`, then install `'.[similarity]'`.
The viewer belongs to the separate Workbench repository.

## Python and scientific dependencies

Use the Python environment in which you intend to prepare data, train, and run
prediction. The model needs PyTorch, PyTorch Geometric, RDKit, and the chemistry library's
`bidict` dependency. Choose a PyTorch
build compatible with the available CPU/CUDA environment before installing FREDIS.

```bash
python -m pip install torch torch-geometric rdkit bidict
python -m pip install -e .
```

The editable install adds the declared NumPy, pandas, matplotlib, HDF5, PyArrow,
and progress-display dependencies. It does not choose a CUDA build or install a
GPU driver. The example command above uses the package index's default PyTorch
build; an existing cluster/container environment can supply its own build.

Verify the environment using the same executable that will run the CLI:

```bash
python -c "import torch, torch_geometric; from rdkit import Chem; print(torch.__version__, torch.cuda.is_available(), Chem.MolFromSmiles('CCO') is not None)"
python -m fredis.cli --help
python -m fredis.ml.specgen.predict_spectrum --help
```

The first command prints the installed PyTorch version, CUDA availability, and
whether RDKit parsed ethanol. CPU execution works with `--device cpu` even when
CUDA is unavailable. Continue with the [quickstart](../quickstart.md).

## Command names

| Entry point | Purpose |
|---|---|
| `python -m fredis.cli` / `fredis` | Data preparation, training, tuning, reaction analysis, and chemistry utilities |
| `python -m fredis.ml.specgen.predict_spectrum` | Single and dataset prediction |
| `fredis-predict-spectrum` | Installed prediction entry point |
| `msentity-predict-fredis` | Equivalent installed prediction entry point |

The module forms remain useful when the environment's scripts directory is not
on `PATH`. Each workflow's complete options appear in the
[CLI reference](../reference/cli.md).

## Optional GUI

FREDIS Workbench is a separate VS Code extension repository. Install its VSIX
and set `fredis.pythonPath` to the Python environment containing FREDIS and
`fredis.applicationRoot` to this backend checkout when needed. Backend install,
tests and documentation do not require the GUI repository.

## Build and read the documentation

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open `http://localhost:8000` locally. Sphinx uses Furo, MyST Markdown, rendered
Mermaid diagrams, selected autodoc API signatures, and CLI tables derived from
actual argument parsers. The source checkout and scientific dependencies must
be importable for those references to build. Notebook execution is disabled;
examples are executed explicitly during verification.

From inside `docs`, `make html` runs the same strict HTML build. See
[development](../development.md) for documentation checks and example verification.
