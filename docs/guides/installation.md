# Installation

FREDIS has a Python package, command-line workflows, and an optional VS Code
Workbench. Install the scientific environment before installing the editable
checkout. Python **3.10 or newer** is required.

## Recommended: Docker

From the FREDIS repository root:

```bash
docker build -t fredis .
docker run --rm -v "$PWD/data:/data" fredis python -m fredis.cli --help
```

The image builds and installs the native libraries and their Python bindings,
FREDIS, PyTorch, PyTorch Geometric, RDKit, and pytest. It supports CPU workflows.
Data under the mounted `/data` directory persists on the host. For example,
pass input/output arguments using `/data/...` paths. No msentity server is needed.

For the existing GPU workspace, rebuild `env/docker/Dockerfile` using the
workspace's usual image build procedure. It now includes CMake, C++ tools,
HDF5 headers and nlohmann-json. Dev Container creation runs
`python mnt/fredis/scripts/install.py`; Compose startup runs the same installer
from the backend directory. Existing containers can run the command manually.

## Existing Python or GPU environment

The sources in `libs/msentity` and `libs/msentity-similarity` are bundled snapshots;
`libs/versions.json` records their base revisions. The chemistry code remains in
`fredis/libs/mmkit`. No submodule initialization is needed.

Use Python 3.10+, CMake 3.21+, a C++20 compiler, HDF5 development files and
nlohmann-json 3.10+. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libhdf5-dev nlohmann-json3-dev
python scripts/install.py
```

Run this inside the virtual environment or container that will run FREDIS. The
installer sets up Python dependencies, compiles both libraries, installs their
Python bindings and installs FREDIS with similarity support. Existing PyTorch
is reused; choose your CPU/CUDA build before running the installer if needed.
On macOS install the corresponding build prerequisites with your package
manager; Windows users should use Docker.

The binding packages bundle the compiled shared libraries, so normal imports
work without PYTHONPATH or manually configured library paths:

```bash
python -c "import msentity, msentity_similarity, fredis"
python -m fredis.cli --help
python -m pytest tests -v
```

`python scripts/install.py --jobs 2` reduces build parallelism. `--no-deps`
skips Python dependency installation when the environment is already prepared.
Native-only development can use `python scripts/build-native.py --similarity`
with PyArrow 20+, setuptools and wheel already installed. Build files are under
`build/`. Re-run the installer after moving the checkout, changing PyArrow,
or updating native sources. `pip install -e .` alone does not build native code.

Continue with the [quickstart](../quickstart.md).

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
