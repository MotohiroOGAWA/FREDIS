"""C detail files: on-demand peak-assignment extraction."""

from __future__ import annotations

from pathlib import Path


def peak_assignments(
    file_path: str | Path,
    sample_id: int,
    peak_index: int,
    fragmenter_params: dict | None,
) -> list[dict]:
    """Every saved assignment path of one peak in one C file."""
    from fredis.analysis.prepared_result import (
        iter_prepared_samples,
        load_prepared_structure,
    )

    _, metadata = load_prepared_structure(file_path)
    sample_ids = list(metadata.get("sample_ids") or ())
    if int(sample_id) not in sample_ids:
        return []
    position = sample_ids.index(int(sample_id))
    for sample in iter_prepared_samples(file_path, fragmenter_params):
        if sample["sampleIndex"] != position:
            continue
        peak = next(
            (
                row
                for row in sample["peaks"]
                if int(row.get("index", -1)) == int(peak_index)
            ),
            None,
        )
        if peak is None:
            return []
        return [
            {
                "fragmentSmiles": match.get("smiles"),
                "formula": match.get("formula"),
                "theoreticalMz": match.get("theoreticalMz"),
                "massErrorPpm": match.get("massErrorPpm"),
                "ionAdduct": match.get("adduct"),
                "hydrogenShift": match.get("hydrogenShift"),
                "reactionCount": match.get("reactionCount"),
                "reactionPatterns": list(match.get("reactionPatterns", ())),
                "reactions": list(match.get("reactions", ())),
            }
            for match in peak.get("matches", ())
        ]
    return []
