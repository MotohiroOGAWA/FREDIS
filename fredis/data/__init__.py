"""Pandas-facing FREDIS model; file I/O is provided by native msentity.

The core model preserves FREDIS's mutable pandas/NumPy view semantics. It is
adapted from the former Python msentity (see LICENSE-msentity), not a second
installation of that package. Similarity is imported only when requested.
"""
from .core.MSDataset import MSDataset, SpectrumRecord, MSDatasetMeta
from .core.PeakSeries import PeakSeries, Spectrum
from .core.Peak import Peak
from .native import load_ms_dataset

__all__ = ['MSDataset', 'SpectrumRecord', 'MSDatasetMeta', 'PeakSeries',
           'Spectrum', 'Peak', 'load_ms_dataset']
