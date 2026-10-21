"""Dataset-scale reaction pattern contribution analysis over a PatternSet tree (A/B/C result format).

A  ``result.reactionpatternanalysis.pft``  HDF5 summaries, distributions and indexes
B  ``compounds/*.peaks.parquet``     one compound, all spectra, binary peak annotations per condition
C  ``details/conditions/**/*.preft.pt``  fragment/reaction/assignment detail per compound tree
"""

from __future__ import annotations

from pathlib import Path

from .constants import RESULT_NAME, RESULT_SUFFIX, SCHEMA, SCHEMA_VERSION
from .reader import ReactionPatternAnalysisResult
from .runner import generate_reaction_pattern_analysis
from .sources import inspect_analysis_source, is_result_file
from .tree import (
    PatternSetSpec,
    NodeSpec,
    TreeConfiguration,
    load_pattern_set_file,
    load_tree_configuration,
)


def open_reaction_pattern_analysis(path: str | Path) -> ReactionPatternAnalysisResult:
    """Open ``result.reactionpatternanalysis.pft``, given the file or its result directory."""
    path = Path(path)
    if path.is_dir():
        path = path / RESULT_NAME
    if not is_result_file(path):
        raise ValueError(f"Not a Reaction Pattern Analysis result: {path}")
    return ReactionPatternAnalysisResult(path)


__all__ = [
    "ReactionPatternAnalysisResult",
    "NodeSpec",
    "PatternSetSpec",
    "RESULT_NAME",
    "RESULT_SUFFIX",
    "SCHEMA",
    "SCHEMA_VERSION",
    "TreeConfiguration",
    "generate_reaction_pattern_analysis",
    "inspect_analysis_source",
    "is_result_file",
    "load_pattern_set_file",
    "load_tree_configuration",
    "open_reaction_pattern_analysis",
]
