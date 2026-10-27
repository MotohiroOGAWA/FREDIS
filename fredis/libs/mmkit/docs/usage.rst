Usage
=====

This section introduces the main functionality of ``mmkit`` with simple examples.

Formula
-------

Parsing a formula string:

.. code-block:: python

   from mmkit import Formula

   f = Formula.parse("C6H12O6")
   print(f.value)

Arithmetic operations:

.. code-block:: python

   water = Formula.parse("H2O")

   print((water * 2).value)
   print((Formula.parse("C6H12O6") + water).value)


Compound
--------

Creating a compound from SMILES:

.. code-block:: python

   from mmkit import Compound

   c = Compound.from_smiles("CCO")
   print(c.smiles)
   print(c.formula)

Accessing properties:

.. code-block:: python

   print(c.exact_mass)
   print(c.charge)


Adduct
------

Parsing an adduct:

.. code-block:: python

   from mmkit import Adduct

   a = Adduct.parse("[M+H]+")
   print(a)


m/z calculation
---------------

Compute the observed m/z from a neutral molecule:

.. code-block:: python

   from mmkit import Formula, Compound, Adduct

   c = Compound.from_smiles("CCO")
   a = Adduct.parse("[M+H]+")

   mz = a.apply_to_mz(c.exact_mass)
   print(mz)