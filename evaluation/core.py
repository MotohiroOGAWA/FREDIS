from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from fredis.domain.mass.parse_ce import parse_ce_to_ev
from fredis.domain.molecule.descriptors import (
    DEFAULT_DESCRIPTOR_NAMES,
    compute_descriptor_values,
)

from .io import read_table


MISSING_LABEL = "(missing)"


@dataclass(frozen=True)
class EvaluationRequest:
    input_path: str
    group_column: str
    metadata: str | None = None
    join_column: str | None = None
    metadata_join_column: str | None = None
    mode: str = "auto"
    bins: tuple[float, ...] = ()
    include: tuple[str, ...] | None = None
    order: tuple[str, ...] = ()
    transform: str = "none"
    precursor_mz_column: str = "PrecursorMZ"
    instrument_column: str | None = None
    smiles_column: str = "SMILES"
    chemical_descriptor: str = "HeavyAtomCount"
    smarts_queries: tuple[tuple[str, str], ...] = ()


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def inspect_dataset(
    input_path: str, metadata: str | None = None, join_column: str | None = None,
    metadata_join_column: str | None = None,
) -> dict[str, Any]:
    frame = read_table(
        input_path, metadata=metadata, join_column=join_column,
        metadata_join_column=metadata_join_column,
    )
    columns = []
    for name in frame.columns:
        series = frame[name]
        numeric = pd.to_numeric(series, errors="coerce")
        non_missing = int(series.notna().sum())
        numeric_count = int(numeric.notna().sum())
        is_numeric = non_missing > 0 and numeric_count == non_missing
        entry: dict[str, Any] = {
            "name": str(name),
            "kind": "numeric" if is_numeric else "categorical",
            "nonMissing": non_missing,
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if is_numeric:
            entry["min"] = _json_value(numeric.min())
            entry["max"] = _json_value(numeric.max())
        columns.append(entry)
    return {
        "path": input_path, "metadata": metadata, "joinColumn": join_column,
        "metadataJoinColumn": metadata_join_column,
        "rows": int(len(frame)), "columns": columns,
    }


def _number_label(value: float) -> str:
    return f"{value:g}"


def _numeric_groups(series: pd.Series, edges: Iterable[float]) -> pd.Series:
    cuts = sorted(set(float(value) for value in edges))
    if not cuts:
        raise ValueError("Numeric grouping requires at least one bin boundary.")
    if not all(np.isfinite(cuts)):
        raise ValueError("Bin boundaries must be finite numbers.")
    values = pd.to_numeric(series, errors="coerce")
    bounds = [-np.inf, *cuts, np.inf]
    labels = [f"[-~,{_number_label(cuts[0])})"]
    labels.extend(
        f"[{_number_label(left)},{_number_label(right)})"
        for left, right in zip(cuts, cuts[1:])
    )
    labels.append(f"[{_number_label(cuts[-1])},~)")
    grouped = pd.cut(values, bins=bounds, labels=labels, right=False, ordered=True)
    return grouped.astype(object).where(values.notna(), MISSING_LABEL)


def _numeric_group_order(edges: Iterable[float]) -> list[str]:
    cuts = sorted(set(float(value) for value in edges))
    if not cuts:
        return [MISSING_LABEL]
    return [
        f"[-~,{_number_label(cuts[0])})",
        *(
            f"[{_number_label(left)},{_number_label(right)})"
            for left, right in zip(cuts, cuts[1:])
        ),
        f"[{_number_label(cuts[-1])},~)",
        MISSING_LABEL,
    ]


def _categorical_groups(series: pd.Series) -> pd.Series:
    return series.map(lambda value: MISSING_LABEL if pd.isna(value) else str(value))


def _collision_energy_values(frame: pd.DataFrame, request: EvaluationRequest) -> pd.Series:
    precursor = (
        frame[request.precursor_mz_column]
        if request.precursor_mz_column in frame.columns
        else pd.Series([None] * len(frame), index=frame.index)
    )
    instrument = (
        frame[request.instrument_column]
        if request.instrument_column and request.instrument_column in frame.columns
        else pd.Series([None] * len(frame), index=frame.index)
    )
    return pd.Series([
        parse_ce_to_ev(ce, mz, inst)
        for ce, mz, inst in zip(frame[request.group_column], precursor, instrument)
    ], index=frame.index, dtype=float)


def _chemical_descriptor_values(
    frame: pd.DataFrame, request: EvaluationRequest,
    progress: Callable[[int, int], None] | None = None,
) -> pd.Series:
    if request.smiles_column not in frame.columns:
        raise ValueError(f"SMILES column was not found: {request.smiles_column}")
    if request.chemical_descriptor not in DEFAULT_DESCRIPTOR_NAMES:
        raise ValueError(
            f"Unsupported chemical descriptor: {request.chemical_descriptor}"
        )
    cache: dict[str, float | None] = {}
    values = []
    total = len(frame)
    report_every = max(1, total // 20)
    for index, raw_smiles in enumerate(frame[request.smiles_column], start=1):
        smiles = "" if pd.isna(raw_smiles) else str(raw_smiles)
        if smiles not in cache:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            cache[smiles] = (
                compute_descriptor_values(mol, (request.chemical_descriptor,))[0]
                if mol is not None else None
            )
        values.append(cache[smiles])
        if progress and (index == total or index % report_every == 0):
            progress(index, total)
    return pd.Series(values, index=frame.index, dtype=float)


def _boxplot_rows(
    frame: pd.DataFrame, group_key: pd.Series,
    progress: Callable[[int, int], None] | None = None,
    outlier_limit: int = 250,
) -> list[dict[str, Any]]:
    if "cosine_similarity" not in frame.columns:
        raise ValueError("Input must contain the .mssim cosine_similarity column.")
    values = pd.to_numeric(frame["cosine_similarity"], errors="coerce")
    grouped = pd.DataFrame({"__group": group_key, "__value": values}).groupby(
        "__group", sort=False, dropna=False
    )
    rows = []
    total = grouped.ngroups
    for index, (category, group) in enumerate(grouped, start=1):
        rows.append(_boxplot_statistics(str(category), group["__value"], outlier_limit))
        if progress:
            progress(index, total)
    return rows


def _boxplot_statistics(
    category: str, values: pd.Series, outlier_limit: int = 250,
) -> dict[str, Any]:
    scores = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if scores.empty:
        return {
            "category": category, "count": 0, "min": None, "q1": None,
            "median": None, "q3": None, "max": None, "whiskerLow": None,
            "whiskerHigh": None, "outliers": [], "outlierCount": 0,
            "density": [],
        }
    q1, median, q3 = (float(scores.quantile(q)) for q in (0.25, 0.5, 0.75))
    iqr = q3 - q1
    central = scores[(scores >= q1 - 1.5 * iqr) & (scores <= q3 + 1.5 * iqr)]
    outliers = scores[~scores.index.isin(central.index)].tolist()
    if len(outliers) > outlier_limit:
        indices = np.linspace(0, len(outliers) - 1, outlier_limit, dtype=int)
        displayed_outliers = [float(outliers[index]) for index in indices]
    else:
        displayed_outliers = [float(value) for value in outliers]
    return {
        "category": category, "count": int(len(scores)),
        "min": float(scores.min()), "q1": q1, "median": median, "q3": q3,
        "max": float(scores.max()), "whiskerLow": float(central.min()),
        "whiskerHigh": float(central.max()), "outliers": displayed_outliers,
        "outlierCount": len(outliers), "density": _density_profile(scores),
    }


def _smarts_boxplot_rows(
    frame: pd.DataFrame, request: EvaluationRequest,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Evaluate each SMARTS independently; matching groups may overlap."""
    if request.smiles_column not in frame.columns:
        raise ValueError(f"SMILES column was not found: {request.smiles_column}")
    if "cosine_similarity" not in frame.columns:
        raise ValueError("Input must contain the .mssim cosine_similarity column.")
    if not request.smarts_queries:
        raise ValueError("SMARTS grouping requires at least one SMARTS query.")

    labels: list[str] = []
    patterns: list[str] = []
    queries = []
    for index, item in enumerate(request.smarts_queries, start=1):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"SMARTS query #{index} must contain a label and SMARTS value.")
        label, pattern = (str(value).strip() for value in item)
        if not label or not pattern:
            raise ValueError(f"SMARTS query #{index} requires a non-empty label and SMARTS value.")
        with rdBase.BlockLogs():
            query = Chem.MolFromSmarts(pattern)
        if query is None or query.GetNumAtoms() == 0:
            raise ValueError(f"Invalid or empty SMARTS for {label!r}: {pattern}")
        labels.append(label)
        patterns.append(pattern)
        queries.append(query)
    if len(labels) != len(set(labels)):
        raise ValueError("SMARTS query labels must be unique.")

    masks = [np.zeros(len(frame), dtype=bool) for _ in queries]
    unique_matches = [set() for _ in queries]
    all_unique: set[str] = set()
    matched_unique: set[str] = set()
    cache: dict[str, tuple[str, tuple[bool, ...]] | None] = {}
    missing = invalid = valid = 0
    total = len(frame)
    report_every = max(1, total // 20)
    for row_index, raw_smiles in enumerate(frame[request.smiles_column], start=0):
        if raw_smiles is None or pd.isna(raw_smiles) or not str(raw_smiles).strip():
            missing += 1
        else:
            smiles = str(raw_smiles).strip()
            if smiles not in cache:
                with rdBase.BlockLogs():
                    mol = Chem.MolFromSmiles(smiles)
                cache[smiles] = None if mol is None or not mol.GetNumAtoms() else (
                    Chem.MolToSmiles(mol),
                    tuple(mol.HasSubstructMatch(query) for query in queries),
                )
            entry = cache[smiles]
            if entry is None:
                invalid += 1
            else:
                valid += 1
                canonical, hits = entry
                all_unique.add(canonical)
                if any(hits):
                    matched_unique.add(canonical)
                for query_index, matched in enumerate(hits):
                    if matched:
                        masks[query_index][row_index] = True
                        unique_matches[query_index].add(canonical)
        done = row_index + 1
        if progress and (done == total or done % report_every == 0):
            progress(done, total)

    rows = []
    for label, pattern, mask, compounds in zip(labels, patterns, masks, unique_matches):
        row = _boxplot_statistics(label, frame.loc[mask, "cosine_similarity"])
        row.update({
            "smarts": pattern,
            "matchedRows": int(mask.sum()),
            "uniqueCompounds": len(compounds),
        })
        rows.append(row)
    return rows, {
        "rows": total, "validSmilesRows": valid, "missingSmilesRows": missing,
        "invalidSmilesRows": invalid, "uniqueValidCompounds": len(all_unique),
        "uniqueMatchedCompounds": len(matched_unique),
    }


def _density_profile(values: pd.Series, points: int = 65) -> list[list[float]]:
    """Return a normalized Gaussian KDE sampled across the similarity axis."""
    samples = values.to_numpy(dtype=float)
    if len(samples) > 5000:
        samples = np.sort(samples)[np.linspace(0, len(samples) - 1, 5000, dtype=int)]
    grid = np.linspace(0.0, 1.0, points)
    spread = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    bandwidth = max(0.025, 1.06 * spread * len(samples) ** -0.2)
    density = np.exp(-0.5 * ((grid[:, None] - samples[None, :]) / bandwidth) ** 2).sum(axis=1)
    maximum = float(density.max())
    normalized = density / maximum if maximum else density
    return [[float(value), float(weight)] for value, weight in zip(grid, normalized)]


def summarize(
    request: EvaluationRequest,
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    if progress:
        progress(5, "Reading evaluation data…")
    frame = read_table(
        request.input_path, metadata=request.metadata, join_column=request.join_column,
        metadata_join_column=request.metadata_join_column,
    )
    if progress:
        progress(20, f"Loaded {len(frame):,} similarity pairs")
    if request.transform in {"none", "collision-energy"} and request.group_column not in frame.columns:
        raise ValueError(f"Group column was not found: {request.group_column}")
    if request.transform == "collision-energy":
        if progress:
            progress(28, "Converting collision energies…")
        source = _collision_energy_values(frame, request)
        effective_column = f"{request.group_column} (eV)"
    elif request.transform == "chemical":
        source = _chemical_descriptor_values(
            frame, request,
            progress=(lambda done, total: progress(
                25 + int(30 * done / max(1, total)),
                f"Computing chemical descriptors… {done:,}/{total:,}",
            )) if progress else None,
        )
        effective_column = request.chemical_descriptor
    elif request.transform == "smarts":
        if progress:
            progress(25, "Matching SMARTS against compounds…")
        rows, structure_summary = _smarts_boxplot_rows(
            frame, request,
            progress=(lambda done, total: progress(
                25 + int(30 * done / max(1, total)),
                f"Matching SMARTS against compounds… {done:,}/{total:,}",
            )) if progress else None,
        )
        effective_column = "SMARTS match"
        mode = "categorical"
    elif request.transform == "none":
        source = frame[request.group_column]
        effective_column = request.group_column
    else:
        raise ValueError("Transform must be none, collision-energy, chemical, or smarts.")
    if request.transform != "smarts":
        structure_summary = None
        detected_numeric = pd.api.types.is_numeric_dtype(source)
        mode = request.mode if request.mode != "auto" else ("numeric" if detected_numeric else "categorical")
        if mode not in {"categorical", "numeric"}:
            raise ValueError("Grouping mode must be auto, categorical, or numeric.")
        group_key = _numeric_groups(source, request.bins) if mode == "numeric" else _categorical_groups(source)
        if progress:
            progress(58, "Calculating group statistics…")
        rows = _boxplot_rows(
            frame, group_key,
            progress=(lambda done, total: progress(
                58 + int(24 * done / max(1, total)),
                f"Calculating group statistics… {done}/{total}",
            )) if progress else None,
        )
    if request.order:
        positions = {name: index for index, name in enumerate(request.order)}
        rows.sort(key=lambda row: (positions.get(row["category"], len(positions)), row["category"]))
    elif mode == "numeric":
        numeric_order = {
            name: index for index, name in enumerate(_numeric_group_order(request.bins))
        }
        rows.sort(key=lambda row: numeric_order.get(row["category"], len(numeric_order)))
    category_rows = rows
    if request.include is not None:
        included = set(request.include)
        rows = [row for row in rows if row["category"] in included]
    result = {
        "input": request.input_path,
        "groupColumn": effective_column,
        "sourceGroupColumn": request.group_column,
        "transform": request.transform,
        "groupMode": mode,
        "scoreColumn": "cosine_similarity",
        "sourceRows": int(len(frame)),
        "categoryRows": category_rows,
        "rows": rows,
    }
    if structure_summary is not None:
        result["structureSummary"] = structure_summary
        result["smartsQueries"] = [
            {"label": label, "smarts": smarts}
            for label, smarts in request.smarts_queries
        ]
    return result
