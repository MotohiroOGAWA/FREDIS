"""Optional prediction scoring through the separate msentity-similarity package."""
from .native import to_native


def calculate_similarity(dataset1, dataset2, **options):
    try:
        from msentity_similarity import similarity_by_key
    except ImportError as exc:
        raise ImportError('Prediction scoring requires msentity-similarity; install fredis[similarity].') from exc
    with to_native(dataset1) as first, to_native(dataset2) as second:
        return similarity_by_key(first, second, **options)
