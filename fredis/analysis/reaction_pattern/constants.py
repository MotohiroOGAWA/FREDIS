"""Names shared by the Reaction Pattern Analysis writer, reader and Workbench backend."""
from __future__ import annotations

SCHEMA = "fredis.reaction-pattern-analysis"
SCHEMA_VERSION = 3
RESULT_FORMAT = "hdf5-tree-v1"

RESULT_NAME = "result.reactionpatternanalysis.pft"
RESULT_SUFFIX = ".reactionpatternanalysis.pft"
RUN_CONFIG_NAME = "config.reaction-pattern-analysis.pft"
MODEL_CONFIG_NAME = "reaction-pattern-analysis-model-config.json"
TREE_CONFIG_NAME = "tree.reaction-pattern-analysis.json"

COMPOUND_DIR = "compounds"
PEAKS_SUFFIX = ".peaks.parquet"
DETAIL_DIR = "details"
CONDITION_DETAIL_DIR = "details/conditions"

DEFAULT_COLUMNS = {
    "smiles": "SMILES",
    "adduct": "AdductType",
    "collision_energy": "CollisionEnergy",
    "precursor_mz": "PrecursorMZ",
}

COMPLETED = "completed"
SKIPPED = "skipped"
FAILED = "failed"

# Skip categories match Training Data preparation (create_training_data.py).
SKIP_LIMIT_EXCEEDED = "limit_exceeded"
SKIP_UNRESOLVED_PRECURSOR = "unresolved_precursor"
SKIP_ERROR = "error"
SKIP_WORKER_CRASH = "worker_crash"
SKIP_PARENT_UNAVAILABLE = "parent_unavailable"

# Per-sample metrics of one tree node.  ``parent_*``/``delta_*``/``newly_*``
# exist only for edges (nodes with a parent).  Coverage metrics use a fixed
# 0-1 histogram.
DISTRIBUTION_METRICS = (
    "peak_coverage",
    "intensity_coverage",
    "parent_peak_coverage",
    "parent_intensity_coverage",
    "delta_peak_coverage",
    "delta_intensity_coverage",
    "newly_assigned_peak_count",
    "newly_assigned_intensity",
)
EDGE_METRICS = frozenset(name for name in DISTRIBUTION_METRICS
                         if name.startswith(("parent_", "delta_", "newly_")))
COVERAGE_METRICS = frozenset(name for name in DISTRIBUTION_METRICS if name.endswith("_coverage"))
HISTOGRAM_BINS = 20
HISTOGRAM_RANGE = (0.0, 1.0)


def assigned_column(condition_id: int) -> str:
    """Peak table column: assigned under this cumulative condition."""
    return f"assigned::{int(condition_id)}"


def count_column(condition_id: int) -> str:
    """Peak table column: number of assignment paths under this condition."""
    return f"count::{int(condition_id)}"
