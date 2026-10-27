API Reference
=============

The ``mmkit`` API provides lightweight utilities for representing molecular
formulas, adduct ions and chemical compounds in a consistent and programmatic way.

This reference documents the core modules of the package:

- ``Formula``: tools for handling elemental compositions, exact masses, and charge-aware formula objects
- ``Adduct``: tools for representing adduct ions, parsing adduct strings, and converting neutral molecules into observed ion forms
- ``Compound``: tools for representing chemical structures, canonical SMILES, and atom-level mappings

These classes are designed to support mass spectrometry workflows such as
formula manipulation, structure handling, adduct processing, and m/z calculation.

Classes
=======

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`Formula <api/generated/mmkit.Formula>`
     - Represents a molecular formula with elemental composition, charge, and exact mass.
   * - :doc:`Compound <api/generated/mmkit.Compound>`
     - Represents a chemical structure with canonical SMILES, molecular formula, and atom mapping support.
   * - :doc:`Adduct <api/generated/mmkit.Adduct>`
     - Represents an adduct ion and provides conversion from neutral molecules to ion forms.


.. toctree::
   :maxdepth: 2
   :hidden:

   api/formula
   api/adduct
   api/compound