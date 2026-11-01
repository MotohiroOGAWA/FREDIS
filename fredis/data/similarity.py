"""Prediction scoring through msentity-similarity."""
from msentity_similarity import parse_mz_tolerance, similarity_by_key

from .native import to_native

# Peaks match within either term, as in msentity-similarity.
DEFAULT_MZ_TOLERANCE = '0.01Da,10ppm'


def calculate_similarity(dataset1, dataset2, *, mz_tolerance=DEFAULT_MZ_TOLERANCE, **options):
    with to_native(dataset1) as first, to_native(dataset2) as second:
        return similarity_by_key(first, second, **parse_mz_tolerance(mz_tolerance), **options)
