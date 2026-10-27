mmkit.Adduct
============

.. currentmodule:: mmkit

.. autoclass:: Adduct




Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :attr:`~mmkit.Adduct.adduct_formulas`
     - Signed counts of adduct formulas.

   * - :attr:`~mmkit.Adduct.charge`
     - Net charge of the adduct.

   * - :attr:`~mmkit.Adduct.element_diff`
     - Total element count difference introduced by the adduct.

   * - :attr:`~mmkit.Adduct.formula_diff`
     - Net formula difference introduced by the adduct.

   * - :attr:`~mmkit.Adduct.ion_type`
     - Ion type of the adduct.

   * - :attr:`~mmkit.Adduct.mass_shift`
     - Exact mass shift introduced by the adduct.

   * - :attr:`~mmkit.Adduct.n_molecules`
     - Number of neutral molecules.




Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :meth:`~mmkit.Adduct.__init__`
     - Initialize an Adduct.

   * - :meth:`~mmkit.Adduct.add`
     - Combine this adduct with another adduct.

   * - :meth:`~mmkit.Adduct.add_prefer_self`
     - Combine two adducts while keeping this adduct's metadata.

   * - :meth:`~mmkit.Adduct.apply_to_formula`
     - Apply the adduct to a neutral formula.

   * - :meth:`~mmkit.Adduct.apply_to_mass`
     - Apply the adduct to a neutral mass.

   * - :meth:`~mmkit.Adduct.apply_to_mz`
     - Apply the adduct to a neutral mass and compute m/z.

   * - :meth:`~mmkit.Adduct.copy`
     - Create a copy of the adduct.

   * - :meth:`~mmkit.Adduct.split_by_reference_adducts`
     - Split an adduct into components matched to reference adducts and a residual component.

   * - :meth:`~mmkit.Adduct.get_formula_count`
     - Get the signed count of a specific formula.

   * - :meth:`~mmkit.Adduct.parse`
     - Parse an adduct string.

   * - :meth:`~mmkit.Adduct.set_charge`
     - Set the net charge of the adduct.

   * - :meth:`~mmkit.Adduct.split`
     - Split the adduct into independent components.




Property Details
----------------


.. autoattribute:: Adduct.adduct_formulas


.. autoattribute:: Adduct.charge


.. autoattribute:: Adduct.element_diff


.. autoattribute:: Adduct.formula_diff


.. autoattribute:: Adduct.ion_type


.. autoattribute:: Adduct.mass_shift


.. autoattribute:: Adduct.n_molecules





Method Details
--------------


.. automethod:: Adduct.__init__


.. automethod:: Adduct.add


.. automethod:: Adduct.add_prefer_self


.. automethod:: Adduct.apply_to_formula


.. automethod:: Adduct.apply_to_mass


.. automethod:: Adduct.apply_to_mz


.. automethod:: Adduct.copy


.. automethod:: Adduct.split_by_reference_adducts


.. automethod:: Adduct.get_formula_count


.. automethod:: Adduct.parse


.. automethod:: Adduct.set_charge


.. automethod:: Adduct.split


