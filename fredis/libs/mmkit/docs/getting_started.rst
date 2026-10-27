Getting Started
===============

``mmkit`` is a lightweight Python toolkit for representing chemical formulas,
molecular structures, and adduct ions in a consistent and programmatic way.

It is designed for cheminformatics and mass spectrometry workflows, providing:

- Formula handling with elemental composition, charge, and exact mass
- Compound representation with canonical SMILES and atom mapping
- Adduct modeling and m/z calculation

Requirements
------------

- Python 3.10 or later
- RDKit (tested with 2024.03.5)

Installation
------------

Install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/MotohiroOGAWA/mmkit.git

For development, you can clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/MotohiroOGAWA/mmkit.git
   cd mmkit
   pip install -e .

Testing
-------

Run tests to verify the installation:

.. code-block:: bash

   python -m run_tests

Basic Usage
-----------

.. code-block:: python

   from mmkit import Formula, Compound, Adduct

   # Formula
   f = Formula.parse("C6H12O6")

   # Compound
   c = Compound.from_smiles("CCO")

   # Adduct
   a = Adduct.parse("[M+H]+")

   # m/z calculation
   mz = a.apply_to_mz(c.exact_mass)