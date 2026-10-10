# Installation

FREDIS requires Python 3.10 or newer, PyTorch and RDKit. Use a Python environment with the appropriate CPU or CUDA PyTorch build and RDKit available, then install the project:

```bash
pip install -e .
python -m fredis.cli --help
```

The scientific environment also needs the libraries used by the project, including NumPy, pandas, HDF5 and PyArrow. Workbench's **Settings → Environment** checks the configured Python executable, PyTorch, RDKit and CUDA.

## Build the documentation

```bash
pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The docs extra installs Sphinx, Furo, sphinx-autodoc-typehints, MyST Parser, MyST-NB and jupyter-sphinx. MyST-NB provides Markdown parsing and notebook support; notebook execution is disabled during the documentation build so building the guide does not launch scientific jobs.
