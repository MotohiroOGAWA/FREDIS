mmkit.Compound
==============

.. currentmodule:: mmkit

.. autoclass:: Compound




Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :attr:`~mmkit.Compound.atom_index_mapped_smiles`
     - SMILES string with atom indices used as atom map numbers.

   * - :attr:`~mmkit.Compound.atom_map_to_index`
     - Mapping from atom map number to canonical atom index.

   * - :attr:`~mmkit.Compound.charge`
     - Formal charge of the compound.

   * - :attr:`~mmkit.Compound.exact_mass`
     - Monoisotopic exact mass of the compound.

   * - :attr:`~mmkit.Compound.formula`
     - Molecular formula of the compound.

   * - :attr:`~mmkit.Compound.mapped_smiles`
     - Canonical SMILES string with atom map numbers.

   * - :attr:`~mmkit.Compound.mol`
     - RDKit molecule without atom map numbers.

   * - :attr:`~mmkit.Compound.mapped_mol`
     - RDKit molecule with atom map numbers.

   * - :attr:`~mmkit.Compound.smiles`
     - Canonical SMILES string without atom map numbers.




Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :meth:`~mmkit.Compound.__init__`
     - Initialize a compound from an RDKit molecule.

   * - :meth:`~mmkit.Compound.assign_atom_map`
     - Assign atom map numbers to atoms.

   * - :meth:`~mmkit.Compound.copy`
     - Create a copy of the compound.

   * - :meth:`~mmkit.Compound.from_smiles`
     - Create a compound from a SMILES string.

   * - :meth:`~mmkit.Compound.get_atom_index_from_map`
     - Get the atom index corresponding to an atom map number.




Property Details
----------------


.. autoattribute:: Compound.atom_index_mapped_smiles


.. autoattribute:: Compound.atom_map_to_index


.. autoattribute:: Compound.charge


.. autoattribute:: Compound.exact_mass


.. autoattribute:: Compound.formula


.. autoattribute:: Compound.mapped_smiles


.. autoattribute:: Compound.mol


.. autoattribute:: Compound.mapped_mol


.. autoattribute:: Compound.smiles





Method Details
--------------


.. automethod:: Compound.__init__


.. automethod:: Compound.assign_atom_map


.. automethod:: Compound.copy


.. automethod:: Compound.from_smiles


.. automethod:: Compound.get_atom_index_from_map


