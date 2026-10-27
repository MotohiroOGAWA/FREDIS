mmkit.Formula
=============

.. currentmodule:: mmkit

.. autoclass:: Formula




Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :attr:`~mmkit.Formula.charge`
     - Net charge of the formula.

   * - :attr:`~mmkit.Formula.elements`
     - Element-count mapping in Hill order.

   * - :attr:`~mmkit.Formula.exact_mass`
     - Monoisotopic exact mass of the formula.

   * - :attr:`~mmkit.Formula.is_nonnegative`
     - Whether all element counts are non-negative.

   * - :attr:`~mmkit.Formula.normalized`
     - Copy with ``raw_formula`` cleared.

   * - :attr:`~mmkit.Formula.normalized_plain`
     - Copy with zero charge and empty ``raw_formula``.

   * - :attr:`~mmkit.Formula.plain`
     - Copy of the formula with zero charge.

   * - :attr:`~mmkit.Formula.plain_value`
     - Canonical formula string without charge.

   * - :attr:`~mmkit.Formula.raw_formula`
     - Original formula string, if available.

   * - :attr:`~mmkit.Formula.value`
     - Canonical formula string including charge.




Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :meth:`~mmkit.Formula.__init__`
     - Initialize a Formula.

   * - :meth:`~mmkit.Formula.copy`
     - Create a copy of the formula.

   * - :meth:`~mmkit.Formula.empty`
     - Return an empty formula.

   * - :meth:`~mmkit.Formula.eq`
     - Compare two Formula objects with optional raw-string comparison.

   * - :meth:`~mmkit.Formula.from_mol`
     - Create a Formula from an RDKit molecule.

   * - :meth:`~mmkit.Formula.parse`
     - Parse a formula string into a Formula object.

   * - :meth:`~mmkit.Formula.to_string`
     - Convert the formula to a string.




Property Details
----------------


.. autoattribute:: Formula.charge


.. autoattribute:: Formula.elements


.. autoattribute:: Formula.exact_mass


.. autoattribute:: Formula.is_nonnegative


.. autoattribute:: Formula.normalized


.. autoattribute:: Formula.normalized_plain


.. autoattribute:: Formula.plain


.. autoattribute:: Formula.plain_value


.. autoattribute:: Formula.raw_formula


.. autoattribute:: Formula.value





Method Details
--------------


.. automethod:: Formula.__init__


.. automethod:: Formula.copy


.. automethod:: Formula.empty


.. automethod:: Formula.eq


.. automethod:: Formula.from_mol


.. automethod:: Formula.parse


.. automethod:: Formula.to_string


