Adduct
======
.. currentmodule:: mmkit

.. autoclass:: Adduct
   :members:
   
Represents an adduct ion in mass spectrometry.

The :class:`~mmkit.Adduct` class describes how a neutral molecule is
transformed into an observed ion through adduct formation. It tracks the number
of molecules, added and removed formulas, and resulting charge.

This class supports parsing standard adduct notation such as ``[M+H]+`` or
``[2M+Na]+``, computing formula and mass differences, and converting neutral
molecules into ion formulas or m/z values.

Class
-----

Core class for representing and applying adduct transformations.

.. autosummary::
   :toctree: generated/

   mmkit.Adduct