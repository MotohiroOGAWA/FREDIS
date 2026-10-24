"""Per-chart TSV storage and a small, atomically published series catalog."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re

INDEX_NAME = "metrics/index.json"
HEADER = ("series", "global_step", "epoch", "event", "value")
SCHEMA = "fredis.metric-index"


def chart_key(series):
    if series.startswith("distributions/"):
        metric = series.split("/", 2)[2]
        metric = re.sub(r"_(min|q1|q10|q25|median|q3|q75|q90|max|mean)$", "", metric)
    else:
        metric = re.sub(
            r"^(train_window|train|validation|val|intermediate_validation)_",
            "",
            series.removeprefix("metrics/"),
        )
    base, _, facet = metric.partition("@")
    base = re.sub(
        r"(peak_(?:recall|precision)_top)(\d+)", lambda m: m[1] + m[2].zfill(2), base
    )
    if not re.match(
        r"^(spectrum_loss|cross_entropy_loss|cosine_loss|transition_cross_entropy)(_|$)",
        base,
    ):
        base = re.sub(r"_without_precursor$", "", base)
    return base + ("@" + facet.split(":", 1)[0] if facet else "")


def _catalog(output):
    path = Path(output) / INDEX_NAME
    if not path.exists():
        return {"schema": SCHEMA, "schemaVersion": 1, "series": []}
    result = json.loads(path.read_text())
    if result.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported metric index: {path}")
    return result


def _publish(output, catalog):
    target = Path(output) / INDEX_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(catalog, separators=(",", ":")) + "\n")
    temporary.replace(target)


def _discard_partial_row(target):
    if not target.exists() or not target.stat().st_size:
        return
    with target.open("rb+") as stream:
        end = stream.seek(0, 2)
        stream.seek(end - 1)
        if stream.read(1) == b"\n":
            return
        while end:
            start = max(0, end - 8192)
            stream.seek(start)
            chunk = stream.read(end - start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                stream.truncate(start + newline + 1)
                return
            end = start
        stream.truncate(0)


def write_series_rows(output, rows, *, replace=False, legacy_sources=None):
    """Write scalar snapshots or append distribution rows, grouped by chart.

    Each row contains (series name, global step, epoch, event, value).
    Files are published before their catalog entries so readers never see a
    catalog pointing to an unfinished new file. Appended partial rows are ignored
    by readers until complete.
    """
    output = Path(output)
    catalog = _catalog(output)
    entries = {entry["name"]: entry for entry in catalog["series"]}
    grouped = {}
    for series, step, epoch, event, value in rows:
        if value is None or not math.isfinite(float(value)):
            continue
        family = (
            "distributions"
            if series.startswith("distributions/")
            else (
                "scalars/intermediate"
                if event == "intermediate_validation"
                else "scalars"
            )
        )
        key = chart_key(series)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:100]
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        relative = f"metrics/{family}/{stem}-{digest}.tsv"
        grouped.setdefault(relative, []).append((series, step, epoch, event, value))
        entries[series] = {"name": series, "file": relative}
    for relative, items in grouped.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not replace:
            _discard_partial_row(target)
        destination = target.with_suffix(".tsv.tmp") if replace else target
        with destination.open("w" if replace else "a", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            if replace or stream.tell() == 0:
                writer.writerow(HEADER)
            writer.writerows(items)
        if replace:
            destination.replace(target)
    catalog["series"] = sorted(entries.values(), key=lambda entry: entry["name"])
    if legacy_sources is not None:
        catalog["legacySources"] = legacy_sources
    if grouped:
        _publish(output, catalog)


def scalar_rows(events):
    for row in events:
        event = row.get("event", "epoch_end")
        for key, value in row.items():
            key = key.replace("/", "_")
            if key in ("event", "epoch", "global_step", "branch_id") or value in (
                None,
                "",
            ):
                continue
            if event == "train_step" and key in (
                "train_batches",
                "train_window_batches",
            ):
                continue
            if (
                event == "train_step"
                and key.startswith("train_")
                and not key.startswith("train_window_")
            ):
                key = "train_window_" + key[len("train_") :]
            if event == "intermediate_validation":
                key = "intermediate_validation_" + key.removeprefix("validation_")
            yield "metrics/" + key, row.get("global_step", 0), row.get(
                "epoch", ""
            ), event, value


def iter_series_rows(output, prefix):
    """Read only partitions containing the requested series family."""
    output = Path(output)
    entries = [
        item for item in _catalog(output)["series"] if item["name"].startswith(prefix)
    ]
    wanted = {item["name"] for item in entries}
    for relative in sorted({item["file"] for item in entries}):
        target = (output / relative).resolve()
        if not target.is_relative_to(output.resolve()):
            raise ValueError("Metric file escapes the training directory")
        with target.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row.get("series") in wanted and row.get("value"):
                    yield row


def read_scalar_events(output):
    output = Path(output)
    catalog = _catalog(output)
    files = {
        entry["file"]
        for entry in catalog["series"]
        if entry["name"].startswith("metrics/")
    }
    events = {}
    for relative in sorted(files):
        target = (output / relative).resolve()
        if not target.is_relative_to(output.resolve()):
            raise ValueError("Metric file escapes the training directory")
        with target.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if not row.get("value"):
                    continue
                event = row["event"]
                if event == "intermediate_validation":
                    continue
                step = int(float(row["global_step"]))
                epoch = row.get("epoch", "")
                item = events.setdefault(
                    (event, step, epoch), {"event": event, "global_step": step}
                )
                if epoch:
                    item["epoch"] = int(float(epoch))
                key = row["series"].removeprefix("metrics/")
                if event == "train_step" and key.startswith("train_window_"):
                    key = "train_" + key[len("train_window_") :]
                item[key] = float(row["value"])
    return sorted(
        events.values(),
        key=lambda item: (item["global_step"], item["event"] != "train_step"),
    )


def normalize_legacy_scalar(row, intensity_weight):
    """Retain historical totals while exposing the current per-loop metrics."""
    row = dict(row)
    for split in ("train", "validation", "val"):
        prefix = split + "_"
        batches = float(row.get(prefix + "batches") or 0)
        if (
            batches > 0
            and row.get(prefix + "samples")
            and not row.get(prefix + "total_samples")
        ):
            row[prefix + "total_samples"] = row[prefix + "samples"]
            row[prefix + "total_batches"] = batches
            row[prefix + "samples"] = float(row[prefix + "samples"]) / batches
        if row.get(prefix + "total_batches") not in (None, ""):
            row[prefix + "batches"] = row[prefix + "total_batches"]
        if row.get(prefix + "intensity_loss") and not row.get(prefix + "spectrum_loss"):
            row[prefix + "spectrum_loss"] = row[prefix + "intensity_loss"]
        for suffix in ("with_precursor", "without_precursor"):
            ce_name = "transition_cross_entropy" + (
                "_without_precursor" if suffix == "without_precursor" else ""
            )
            ce, loss = row.get(prefix + ce_name), row.get(
                prefix + "spectrum_loss_" + suffix
            )
            if ce in (None, "") or loss in (None, "") or intensity_weight is None:
                continue
            ce_key = prefix + "cross_entropy_loss_" + suffix
            cosine_key = prefix + "cosine_loss_" + suffix
            if row.get(ce_key) in (None, ""):
                row[ce_key] = float(ce) * intensity_weight
            if row.get(cosine_key) in (None, ""):
                row[cosine_key] = float(loss) - float(row[ce_key])
        for metric in ("cross_entropy_loss", "cosine_loss"):
            full, excluded = row.get(prefix + metric + "_with_precursor"), row.get(
                prefix + metric + "_without_precursor"
            )
            if (
                full not in (None, "")
                and excluded not in (None, "")
                and not row.get(prefix + metric)
            ):
                row[prefix + metric] = float(full) + float(excluded)
    return row


def migrate_legacy(output):
    """Convert existing TSV logs without removing or changing the originals."""
    output = Path(output)
    rows = []
    sources = []
    intensity_weight = None
    from .configuration import CONFIG_NAME

    config_path = output / CONFIG_NAME
    if config_path.exists():
        config = json.loads(config_path.read_text())
        intensity_weight = float(
            config.get("intensityLossWeight", config.get("intensity_loss_weight", 1))
        )
    for filename in (
        "metrics.tsv",
        "metric_distributions.tsv",
        "intermediate_validation.tsv",
    ):
        target = output / filename
        if not target.exists():
            continue
        before = target.stat()
        with target.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if None in row or any(value is None for value in row.values()):
                    continue
                if filename == "metric_distributions.tsv":
                    if row.get("value"):
                        rows.append(
                            (
                                "distributions/" + row["split"] + "/" + row["metric"],
                                row["global_step"],
                                row.get("epoch", ""),
                                "",
                                row["value"],
                            )
                        )
                else:
                    if filename == "intermediate_validation.tsv":
                        row["event"] = "intermediate_validation"
                    rows.extend(
                        scalar_rows([normalize_legacy_scalar(row, intensity_weight)])
                    )
        after = target.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(
                f"{filename} changed during migration; retry after writing finishes"
            )
        sources.append(
            {
                "file": filename,
                "size": after.st_size,
                "mtimeMs": after.st_mtime_ns / 1_000_000,
            }
        )
    if not rows:
        raise ValueError("No legacy metrics found")
    if (output / INDEX_NAME).exists():
        raise ValueError(
            "A metric index already exists; existing partitioned data was left untouched"
        )
    write_series_rows(output, rows, replace=True, legacy_sources=sources)
    catalog = _catalog(output)
    return {
        "index": str(output / INDEX_NAME),
        "series": len(catalog["series"]),
        "files": len({entry["file"] for entry in catalog["series"]}),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split legacy training metrics into per-chart TSV files"
    )
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate_legacy(args.directory)))
