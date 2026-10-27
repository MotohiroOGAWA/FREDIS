"""MSDataset export for the union of validation quantile representatives."""

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

from fredis.data import MSDataset, PeakSeries


def save_validation_msdataset(
    records, representatives, path, *, epoch=None, global_step=None, summary=None
):
    metadata, peaks, peak_metadata, offsets = [], [], [], [0]
    selections = {index: [] for index in range(len(records))}
    for metric, levels in representatives.items():
        for quantile, item in levels.items():
            selections[item["spectrum_index"]].append(
                f"{metric}:{'q50' if quantile == 'median' else quantile}"
            )
    for index, record in enumerate(records):
        scalar = {
            key: value
            for key, value in record.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
        for role, field in (
            ("generated", "generated_peaks"),
            ("observed", "original_peaks"),
        ):
            metadata.append(
                {
                    **scalar,
                    "SMILES": record.get("smiles", ""),
                    "RepresentativeIndex": index,
                    "SpectrumRole": role,
                    "Epoch": epoch,
                    "GlobalStep": global_step,
                    "PrecursorMZ": record.get("precursor_mz"),
                    "AdductType": record.get("adduct", record.get("main_adduct", "")),
                    "CollisionEnergy": record.get("collision_energy"),
                    "SelectedBy": json.dumps(selections[index]),
                }
            )
            for peak in record.get(field, []):
                peaks.append((float(peak["mz"]), float(peak["intensity"])))
                peak_metadata.append(
                    {
                        "precursor": bool(peak.get("precursor")),
                        "depth": (
                            peak.get("depth") if peak.get("depth") is not None else -1
                        ),
                        "assigned": bool(peak.get("assigned")),
                    }
                )
            offsets.append(len(peaks))
    frame = pd.DataFrame(metadata)
    dataset = MSDataset(
        spectrum_metadata=frame,
        peak_series=PeakSeries(
            data=np.asarray(peaks, dtype=np.float64).reshape(-1, 2),
            offsets=np.asarray(offsets, dtype=np.int64),
            metadata=pd.DataFrame(
                peak_metadata, columns=["precursor", "depth", "assigned"]
            ),
            metadata_columns=["precursor", "depth", "assigned"],
            sort_by_mz=True,
        ),
        columns=frame.columns.tolist(),
        description="FREDIS validation quantile representatives: paired generated and observed spectra",
        attributes={
            "selection": "Union of with/without precursor × assignment/cosine × q10/q25/q50/q75/q90",
            "validation_summary": json.dumps(summary or {}),
            "validation_representatives": json.dumps(
                {
                    metric: {
                        level: {k: v for k, v in item.items() if k != "spectrum"}
                        for level, item in levels.items()
                    }
                    for metric, levels in representatives.items()
                }
            ),
        },
    )
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        dataset.save(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_validation_msdataset(path):
    """Read paired representative spectra and their selection metadata from MSDS."""
    dataset = MSDataset.load(str(path))
    records = {}
    selected = {}
    for position, row in dataset.metadata.iterrows():
        record = json.loads(row.to_json())
        index = int(record["RepresentativeIndex"])
        target = records.setdefault(index, dict(record))
        spectrum = dataset.peaks[position]
        peaks = []
        metadata = spectrum.metadata
        for offset, (mz, intensity) in enumerate(zip(spectrum.mz, spectrum.intensity)):
            extra = (
                json.loads(metadata.iloc[offset].to_json())
                if metadata is not None
                else {}
            )
            if extra.get("depth") == -1:
                extra["depth"] = None
            peaks.append({**extra, "mz": float(mz), "intensity": float(intensity)})
        target[
            (
                "generated_peaks"
                if record["SpectrumRole"] == "generated"
                else "original_peaks"
            )
        ] = peaks
        selected[index] = json.loads(record.get("SelectedBy") or "[]")
    representatives = json.loads(
        dataset.attributes.get("validation_representatives", "{}")
    )
    if not representatives:
        for index, choices in selected.items():
            for choice in choices:
                metric, level = choice.rsplit(":", 1)
                level = "median" if level == "q50" else level
                representatives.setdefault(metric, {})[level] = {
                    "spectrum_index": index,
                    "value": records[index].get(metric),
                }
    for levels in representatives.values():
        for item in levels.values():
            item["spectrum"] = records[int(item["spectrum_index"])]
    return {
        "summary": json.loads(dataset.attributes.get("validation_summary", "{}")),
        "representatives": representatives,
    }


def export_legacy_preview(preview, path, *, run_directory=None):
    """Convert a bounded preview, recovering Source SMILES from prepared metadata."""
    import re

    representatives = {}
    records, indices = [], {}
    smiles = {}
    step = preview.get("global_step")
    match = re.match(r"epoch_(\d+)", Path(path).stem)
    epoch = int(match[1]) if match else preview.get("epoch")
    if run_directory is not None:
        root = Path(run_directory)
        from .configuration import CONFIG_NAME

        config = json.loads((root / CONFIG_NAME).read_text())
        from .prepared_dataset import prepared_piece_sequence

        pieces = prepared_piece_sequence(
            config["valDir"],
            max_samples=int(config.get("maxSamples", 128)),
            minimum_assignment_score=float(config.get("minimumAssignmentScore", 0)),
            minimum_assignment_score_without_precursor=float(
                config.get("minimumAssignmentScoreWithoutPrecursor", 0)
            ),
            cache_in_memory=False,
        )
        summary_file = root / "dataset_summary.json"
        expected = (
            json.loads(summary_file.read_text()).get("validation", {}).get("chunks")
            if summary_file.exists()
            else None
        )
        if expected is not None and len(pieces) != expected:
            raise ValueError(
                "Prepared validation indexing differs from the saved run; SMILES cannot be recovered safely"
            )
        smiles = {
            index: piece.compound_smiles for index, piece in enumerate(pieces._pieces)
        }
        if step is None:
            import csv

            legacy = root / "metrics.tsv"
            if legacy.exists():
                with legacy.open() as stream:
                    events = list(csv.DictReader(stream, delimiter="\t"))
            else:
                from .metric_store import read_scalar_events

                events = read_scalar_events(root)
            step = next(
                (
                    int(float(row["global_step"]))
                    for row in events
                    if row.get("event") == "epoch_end"
                    and (
                        float(row["epoch"])
                        if row.get("epoch") not in (None, "")
                        else -1
                    )
                    == epoch
                ),
                None,
            )
    for metric, levels in preview.get("representatives", {}).items():
        representatives[metric] = {}
        for quantile, item in levels.items():
            key = item.get("source_spectrum_index", item["spectrum_index"])
            if key not in indices:
                record = dict(item["spectrum"])
                record["smiles"] = (
                    record.get("smiles") or smiles.get(record.get("piece_index")) or ""
                )
                indices[key] = len(records)
                records.append(record)
            index = indices[key]
            representatives[metric][quantile] = {
                **item,
                "spectrum_index": index,
                "source_spectrum_index": key,
                "spectrum": records[index],
            }
    if not records:
        raise ValueError("No representative spectra in preview")
    save_validation_msdataset(
        records,
        representatives,
        path,
        epoch=epoch,
        global_step=step,
        summary=preview.get("summary", {}),
    )
    return {
        "file": str(path),
        "spectra": len(records),
        "smiles": sum(bool(record["smiles"]) for record in records),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export representative validation spectra from a compact preview"
    )
    parser.add_argument("preview", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-directory", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export_legacy_preview(
                json.loads(args.preview.read_text()),
                args.output,
                run_directory=args.run_directory,
            )
        )
    )
