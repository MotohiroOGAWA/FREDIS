"""Query API over an A/B/C PatternSet-tree result with lazy loading.

Opening a result reads only A (HDF5) metadata and small tables.  A compound's
B Parquet is read only when that compound's peaks are requested, and a C
``.preft.pt`` only when a peak/path detail is requested.  ``loaded_files``
records every B/C file read, for tests and diagnostics.

Nodes are addressed by ``node_row`` (their position in the stored tree).  A
root reports absolute metrics; an edge reports parent, child and delta
metrics for the compounds it applies to.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import json
import math
from pathlib import Path

import numpy as np

from .constants import COMPLETED, SCHEMA, SCHEMA_VERSION, assigned_column, count_column
from .hdf5_storage import Table, _plain

SMALL_TABLES = ("datasets", "pattern_sets", "patterns", "conditions", "nodes", "node_summary", "node_distribution")
COMPOUND_COLUMNS = (
    "compound_id", "canonical_smiles", "original_smiles", "dataset_count", "dataset_ids", "sample_count", "peak_count",
    "total_intensity", "evaluated_node_count", "applicable_node_rows", "improving_edge_count", "new_peak_union_count",
    "new_intensity_union", "status", "reason")
NODE_ROW_COLUMNS = (
    "node_row", "applicable", "status", "reason", "sample_count", "improved_sample_count",
    "newly_assigned_peak_count_total", "newly_assigned_intensity_total",
    "weighted_peak_coverage", "weighted_intensity_coverage",
    "weighted_peak_coverage_without_precursor", "weighted_intensity_coverage_without_precursor",
    "parent_weighted_peak_coverage", "parent_weighted_intensity_coverage",
    "weighted_delta_peak_coverage", "weighted_delta_intensity_coverage",
    "weighted_delta_peak_coverage_without_precursor", "weighted_delta_intensity_coverage_without_precursor",
    "mean_peak_coverage", "mean_intensity_coverage", "mean_peak_coverage_without_precursor",
    "mean_intensity_coverage_without_precursor", "mean_delta_peak_coverage", "mean_delta_intensity_coverage",
    "mean_delta_peak_coverage_without_precursor", "mean_delta_intensity_coverage_without_precursor")
FREQUENCY_METRICS = {"intensity_coverage": "intensity_coverage", "peak_coverage": "peak_coverage",
                     "assigned_peak_count": "assigned_peak_count"}


def _number(value):
    if value is None or value == "":
        return None
    return float(value)


def _split(value) -> list[str]:
    return [item for item in str(value or "").split(",") if item != ""]


class ReactionPatternAnalysisResult:
    def __init__(self, path: str | Path):
        import h5py

        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.loaded_files: list[str] = []
        self._cache: dict = {}
        with h5py.File(self.path, "r") as handle:
            attrs = dict(handle["metadata"].attrs)
        self.metadata = {key: _plain(value) for key, value in attrs.items()}
        if self.metadata.get("schema") != SCHEMA:
            raise ValueError(f"Not a Reaction Pattern Analysis result: {self.path}")
        version = int(self.metadata.get("schema_version") or 0)
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported Reaction Pattern Analysis schema version {version}: {self.path}")

    # -- plumbing ---------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self._cache.clear()

    @contextmanager
    def _file(self):
        import h5py

        with h5py.File(self.path, "r") as handle:
            yield handle

    def table_records(self, name: str, columns=None, rows=None) -> list[dict]:
        with self._file() as handle:
            return Table(handle[name]).records(columns, rows)

    def _small(self, name: str) -> list[dict]:
        if name not in self._cache:
            self._cache[name] = self.table_records(name)
        return self._cache[name]

    def configuration(self) -> dict:
        if "configuration" not in self._cache:
            with self._file() as handle:
                self._cache["configuration"] = json.loads(handle["configuration/json"].asstr()[()])
        return self._cache["configuration"]

    def _range(self, handle, index: str, compound_row: int) -> slice:
        group = handle["indexes"][index]
        start, count = int(group["start"][compound_row]), int(group["count"][compound_row])
        return slice(start, start + count)

    def _key_rows(self, handle, index: str, key: int) -> np.ndarray:
        group = handle["indexes"][index]
        keys = group["key"][()]
        position = int(np.searchsorted(keys, key))
        if position >= len(keys) or keys[position] != key:
            return np.asarray([], dtype=np.int64)
        start, count = int(group["start"][position]), int(group["count"][position])
        return group["rows"][start:start + count]

    def _compound_ids(self, handle) -> np.ndarray:
        if "compound_ids" not in self._cache:
            self._cache["compound_ids"] = Table(handle["compounds"]).column("compound_id")
        return self._cache["compound_ids"]

    def compound_row(self, handle, compound_id: str) -> int:
        ids = self._compound_ids(handle)  # compounds are stored sorted by compound_id
        position = int(np.searchsorted(ids, compound_id))
        if position >= len(ids) or ids[position] != compound_id:
            raise KeyError(f"Unknown compound: {compound_id}")
        return position

    # -- tree -----------------------------------------------------------------------
    def nodes(self) -> list[dict]:
        return self._small("nodes")

    def node(self, node_row: int) -> dict:
        return self.nodes()[int(node_row)]

    def node_path(self, node_row: int) -> list[dict]:
        result, current = [], self.node(node_row)
        while True:
            result.append(current)
            if current["parent_node_row"] < 0:
                return list(reversed(result))
            current = self.node(current["parent_node_row"])

    def tree(self) -> dict:
        patterns = {row["pattern_id"]: row for row in self._small("patterns")}
        sets = []
        for row in self._small("pattern_sets"):
            ids = [int(value) for value in _split(row["pattern_ids"])]
            sets.append(dict(row, patterns=[dict(id=pid, name=patterns[pid]["name"],
                                                  reactant_smarts=patterns[pid]["reactant_smarts"]) for pid in ids]))
        return {"patternSets": sets, "patterns": self._small("patterns"), "conditions": self._small("conditions"),
                "nodes": self.nodes(), "configuration": self.configuration().get("tree")}

    # -- A-only queries -----------------------------------------------------------
    def overall_summary(self) -> dict:
        if "overall" not in self._cache:
            with self._file() as handle:
                self._cache["overall"] = json.loads(handle["metadata/overall_summary"].asstr()[()])
        return {key: (None if isinstance(value, float) and math.isnan(value) else value)
                for key, value in self._cache["overall"].items()}

    def info(self) -> dict:
        with self._file() as handle:
            counts = {name: int(handle[name].attrs["row_count"]) for name in (
                "datasets", "pattern_sets", "patterns", "nodes", "compounds", "compound_node_summary",
                "sample_summary", "sample_node_summary", "compound_condition")}
        return {"format": "hdf5", "path": str(self.path), "metadata": self.metadata,
                "configuration": self.configuration(), "datasets": self._small("datasets"), "tree": self.tree(),
                "counts": counts, "overall": self.overall_summary(), "nodeSummary": self.node_summary()}

    def node_summary(self) -> list[dict]:
        summaries = {row["node_row"]: row for row in self._small("node_summary")}
        conditions = {row["condition_id"]: row for row in self._small("conditions")}
        patterns = {row["pattern_id"]: row["name"] for row in self._small("patterns")}
        result = []
        for node in self.nodes():
            row = {**summaries.get(node["node_row"], {}), **node}
            row["added_patterns"] = [patterns.get(int(pid), pid) for pid in _split(node["added_pattern_ids"])]
            row["condition_pattern_ids"] = _split(conditions.get(node["condition_id"], {}).get("pattern_ids"))
            row["parent_condition_label"] = (self.node(node["parent_node_row"])["condition_label"]
                                             if node["parent_node_row"] >= 0 else "")
            result.append(row)
        return result

    def node_distribution(self, node_row: int) -> list[dict]:
        return [row for row in self._small("node_distribution") if row["node_row"] == int(node_row)]

    def _node_values(self, handle, field: str) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
        """Per-node absolute values for every population sample.

        Compounds an edge does not apply to inherit the parent's value; an
        applicable compound whose child condition did not complete is NaN.
        """
        samples = Table(handle["sample_summary"])
        sample_ids = np.asarray(samples.column("sample_id"), dtype=np.int64)
        sample_rows = np.asarray(samples.column("compound_row"), dtype=np.int64)
        pairs = Table(handle["sample_node_summary"])
        pair_ids = np.asarray(pairs.column("sample_id"), dtype=np.int64)
        pair_nodes = np.asarray(pairs.column("node_row"), dtype=np.int64)
        pair_values = np.asarray(pairs.column(field), dtype=float)
        summaries = Table(handle["compound_node_summary"])
        summary_nodes = np.asarray(summaries.column("node_row"), dtype=np.int64)
        summary_rows = np.asarray(summaries.column("compound_row"), dtype=np.int64)
        summary_status = np.asarray(summaries.column("status"))
        positions = np.searchsorted(sample_ids, pair_ids)
        values: dict[int, np.ndarray] = {}
        for node in self.nodes():
            row = node["node_row"]
            base = (np.full(len(sample_ids), np.nan) if node["parent_node_row"] < 0
                    else values[node["parent_node_row"]].copy())
            if node["parent_node_row"] >= 0:
                failed = np.zeros(int(sample_rows.max()) + 1 if len(sample_rows) else 0, dtype=bool)
                failed[summary_rows[(summary_nodes == row) & (summary_status != COMPLETED)]] = True
                if len(failed):
                    base[failed[sample_rows]] = np.nan
            own = pair_nodes == row
            base[positions[own]] = pair_values[own]
            values[row] = base
        return sample_ids, sample_rows, values

    def score_frequency(self, metric: str = "intensity_coverage", *, without_precursor: bool = False,
                        bins: int = 20, filters: dict | None = None) -> dict:
        """Filtered sample-frequency series of every node's absolute value."""
        filters = dict(filters or {})
        if metric not in FREQUENCY_METRICS:
            raise ValueError("metric must be intensity_coverage, peak_coverage or assigned_peak_count")
        field = FREQUENCY_METRICS[metric] + ("_without_precursor" if without_precursor else "")
        bins = min(100, max(2, int(bins)))
        with self._file() as handle:
            compound_mask = self._compound_mask(handle, filters)
            sample_ids, sample_rows, values = self._node_values(handle, field)
            sample_mask = compound_mask[sample_rows] & self._sample_mask(handle, filters)
        series_data = []
        for node in self.nodes():
            data = values[node["node_row"]][sample_mask]
            series_data.append({"key": f"node:{node['node_row']}", "label": node["condition_label"],
                                "nodeRow": node["node_row"], "pathLabel": node["path_label"],
                                "isRoot": bool(node["is_root"]), "excludedSamples": int(np.count_nonzero(~np.isfinite(data))),
                                "data": data[np.isfinite(data)]})
        finite = [item["data"] for item in series_data if len(item["data"])]
        merged = np.concatenate(finite) if finite else np.asarray([], dtype=float)
        if metric.endswith("coverage"):
            edges = np.linspace(0.0, 1.0, bins + 1)
        else:
            maximum = float(merged.max()) if merged.size else 1.0
            exponent = math.floor(math.log10(maximum)) if maximum > 0 else 0
            scale = 10.0 ** exponent
            fraction = maximum / scale
            nice = 1.0 if fraction <= 1 else 2.0 if fraction <= 2 else 5.0 if fraction <= 5 else 10.0
            edges = np.linspace(0.0, max(1.0, nice * scale), bins + 1)
        output = []
        for item in series_data:
            data = item.pop("data")
            counts, _ = np.histogram(np.clip(data, edges[0], edges[-1]), bins=edges)
            output.append({**item, "n": int(data.size), "counts": counts.astype(int).tolist()})
        return {"metric": metric, "withoutPrecursor": bool(without_precursor), "bins": bins,
                "filteredSamples": int(np.count_nonzero(sample_mask)), "filters": filters,
                "edges": [float(value) for value in edges], "series": output}

    def _compound_mask(self, handle, filters: dict, node_values: dict | None = None) -> np.ndarray:
        compounds = Table(handle["compounds"])
        count = len(compounds)
        mask = np.ones(count, dtype=bool)
        without = "_without_precursor" if filters.get("withoutPrecursor") else ""
        node_row = filters.get("nodeRow")
        node_row = None if node_row in (None, "", "all") else int(node_row)
        if node_row is not None:
            rows = np.sort(self._key_rows(handle, "compound_node_by_node", node_row))
            data = Table(handle["compound_node_summary"]).read(
                ["compound_row", "status", "mean_delta_peak_coverage" + without, "mean_delta_intensity_coverage" + without,
                 "mean_peak_coverage" + without, "mean_intensity_coverage" + without], rows)
            in_node = np.zeros(count, dtype=bool)
            in_node[data["compound_row"]] = True
            mask &= in_node
            wanted = COMPLETED if filters.get("completedOnly") else filters.get("nodeStatus")
            if wanted:
                keep = np.zeros(count, dtype=bool)
                keep[data["compound_row"][data["status"] == wanted]] = True
                mask &= keep
            for key, column in (("deltaPeakCoverageMin", "mean_delta_peak_coverage"),
                                ("deltaIntensityCoverageMin", "mean_delta_intensity_coverage")):
                threshold = _number(filters.get(key))
                if threshold is not None:
                    keep = np.zeros(count, dtype=bool)
                    keep[data["compound_row"][np.nan_to_num(data[column + without], nan=-np.inf) >= threshold]] = True
                    mask &= keep
            for key, column in (("peakCoverage", "mean_peak_coverage"), ("intensityCoverage", "mean_intensity_coverage")):
                low, high = _number(filters.get(key + "Min")), _number(filters.get(key + "Max"))
                for bound, compare in ((low, np.greater_equal), (high, np.less_equal)):
                    if bound is None:
                        continue
                    values = np.nan_to_num(data[column + without], nan=-np.inf if compare is np.greater_equal else np.inf)
                    keep = np.zeros(count, dtype=bool)
                    keep[data["compound_row"][compare(values, bound)]] = True
                    mask &= keep
        if filters.get("smiles"):
            needle = str(filters["smiles"]).lower()
            canonical, original = compounds.column("canonical_smiles"), compounds.column("original_smiles")
            mask &= np.fromiter(((needle in str(a).lower()) or (needle in str(b).lower())
                                 for a, b in zip(canonical, original)), dtype=bool, count=count)
        if filters.get("completedOnly"):
            mask &= compounds.column("status") == COMPLETED
        elif filters.get("status"):
            mask &= compounds.column("status") == filters["status"]
        if filters.get("improvedOnly"):
            mask &= compounds.column("new_peak_union_count") > 0
        if filters.get("datasetId") not in (None, "", "all"):
            keep = np.zeros(count, dtype=bool)
            keep[self._key_rows(handle, "compound_by_dataset", int(filters["datasetId"]))] = True
            mask &= keep
        return mask

    def _sample_mask(self, handle, filters: dict) -> np.ndarray:
        samples = Table(handle["sample_summary"])
        mask = np.ones(len(samples), dtype=bool)
        if filters.get("datasetId") not in (None, "", "all"):
            mask &= samples.column("dataset_id") == int(filters["datasetId"])
        if filters.get("adduct"):
            needle = str(filters["adduct"]).lower()
            mask &= np.fromiter((needle in str(value).lower() for value in samples.column("adduct")),
                                dtype=bool, count=len(samples))
        for key, column in (("collisionEnergy", "collision_energy"), ("precursorMz", "precursor_mz")):
            low, high = _number(filters.get(key + "Min")), _number(filters.get(key + "Max"))
            if low is not None or high is not None:
                values = np.asarray(samples.column(column), dtype=float)
                if low is not None:
                    mask &= values >= low
                if high is not None:
                    mask &= values <= high
        return mask

    def compound_list(self, filters: dict | None = None, *, sort: str = "compound_id", descending: bool = False,
                      offset: int = 0, limit: int = 200) -> dict:
        filters = dict(filters or {})
        node_row = filters.get("nodeRow")
        node_row = None if node_row in (None, "", "all") else int(node_row)
        with self._file() as handle:
            compounds = Table(handle["compounds"])
            count = len(compounds)
            mask = self._compound_mask(handle, filters)
            sample_filters = [key for key in ("adduct", "collisionEnergyMin", "collisionEnergyMax", "precursorMzMin",
                                              "precursorMzMax") if filters.get(key) not in (None, "")]
            if sample_filters:
                samples = Table(handle["sample_summary"])
                keep = np.zeros(count, dtype=bool)
                keep[samples.column("compound_row")[self._sample_mask(handle, filters)]] = True
                mask &= keep
            node_values = {}
            if node_row is not None:
                rows = np.sort(self._key_rows(handle, "compound_node_by_node", node_row))
                data = Table(handle["compound_node_summary"]).read(["compound_row", *NODE_ROW_COLUMNS[1:]], rows)
                position = np.full(count, -1, dtype=np.int64)
                position[data["compound_row"]] = np.arange(len(rows))
                node_values = {"position": position, **data}
            selected = np.flatnonzero(mask)
            sort_values = None
            if sort in compounds.columns:
                sort_values = compounds.column(sort, selected)
            elif sort in node_values and node_row is not None:
                sort_values = node_values[sort][node_values["position"][selected]]
            if sort_values is not None and len(selected):
                if sort_values.dtype.kind in "fc":
                    sort_values = np.nan_to_num(sort_values, nan=-np.inf)
                order = np.argsort(sort_values.astype(str) if sort_values.dtype.kind == "O" else sort_values,
                                   kind="stable")
                selected = selected[order[::-1] if descending else order]
            page = selected[max(0, int(offset)):max(0, int(offset)) + max(1, int(limit))]
            rows = compounds.records(list(COMPOUND_COLUMNS), np.sort(page)) if len(page) else []
            by_row = dict(zip(np.sort(page).tolist(), rows))
            result = []
            for row_index in page.tolist():
                row = dict(by_row[row_index], compound_row=row_index)
                if node_row is not None:
                    position = int(node_values["position"][row_index])
                    row["node"] = {name: _plain(node_values[name][position]) for name in NODE_ROW_COLUMNS[1:]}
                result.append(row)
        return {"total": int(len(selected)), "offset": int(offset), "rows": result}

    def _conditions_of(self, handle, compound_row: int) -> dict[int, dict]:
        rows = Table(handle["compound_condition"]).records(
            rows=self._range(handle, "compound_condition_by_compound", compound_row))
        for row in rows:
            row["kept"] = [int(value) for value in _split(row["kept"])]
        return {int(row["condition_id"]): row for row in rows}

    def compound_summary(self, compound_id: str) -> dict:
        with self._file() as handle:
            row = self.compound_row(handle, compound_id)
            compound = Table(handle["compounds"]).records(rows=slice(row, row + 1))[0]
            node_rows = Table(handle["compound_node_summary"]).records(
                rows=self._range(handle, "compound_node_by_compound", row))
            conditions = self._conditions_of(handle, row)
        for item in node_rows:
            node = self.node(item["node_row"])
            item.update(node_id=node["node_id"], path_label=node["path_label"], condition_label=node["condition_label"],
                        pattern_set_name=node["pattern_set_name"], depth=node["depth"],
                        parent_node_row=node["parent_node_row"])
        return {"compound": compound, "nodes": node_rows, "conditions": list(conditions.values())}

    def sample_summary(self, compound_id: str) -> list[dict]:
        with self._file() as handle:
            row = self.compound_row(handle, compound_id)
            return Table(handle["sample_summary"]).records(rows=self._range(handle, "sample_by_compound", row))

    def compound_node_distribution(self, compound_id: str, node_row: int) -> dict:
        node_row = int(node_row)
        with self._file() as handle:
            row = self.compound_row(handle, compound_id)
            distributions = [item for item in Table(handle["compound_node_distribution"]).records(
                rows=self._range(handle, "compound_node_distribution_by_compound", row)) if item["node_row"] == node_row]
            samples = {item["sample_id"]: item for item in Table(handle["sample_summary"]).records(
                ["sample_id", "dataset_id", "record_index", "sample_index", "adduct", "collision_energy",
                 "precursor_mz"], self._range(handle, "sample_by_compound", row))}
            values = [dict(samples.get(item["sample_id"], {}), **item) for item in Table(
                handle["sample_node_summary"]).records(rows=self._range(handle, "sample_node_by_compound", row))
                if item["node_row"] == node_row]
        for item in distributions:
            item["histogram"] = json.loads(item.pop("histogram_json") or "{}")
        return {"compoundId": compound_id, "nodeRow": node_row, "default": "mean",
                "distributions": distributions, "samples": values}

    # -- B --------------------------------------------------------------------------
    def compound_peaks(self, compound_id: str, filters: dict | None = None, *, sort: str | None = None,
                       descending: bool = False, offset: int = 0, limit: int = 2000) -> dict:
        """Peaks with one state per node: ``new`` (edges), ``assigned``, ``missing`` or None (not evaluated)."""
        import pandas as pd

        filters = dict(filters or {})
        with self._file() as handle:
            row = self.compound_row(handle, compound_id)
            compound = Table(handle["compounds"]).records(rows=slice(row, row + 1))[0]
            conditions = self._conditions_of(handle, row)
            node_rows = {item["node_row"]: item for item in Table(handle["compound_node_summary"]).records(
                ["node_row", "status", "applicable"], self._range(handle, "compound_node_by_compound", row))}
        path = self.root / compound["peak_summary_file"]
        self.loaded_files.append(str(path))
        frame = pd.read_parquet(path)
        nodes = []
        for node in self.nodes():
            summary = node_rows.get(node["node_row"])
            if summary is None or summary["status"] != COMPLETED:
                continue
            condition = conditions.get(node["condition_id"])
            parent = conditions.get(node["parent_condition_id"]) if node["parent_condition_id"] >= 0 else None
            if condition is None:
                continue
            column = f"node::{node['node_row']}"
            population = set((parent or condition)["kept"])
            evaluated = frame["sample_index"].isin(population).to_numpy()
            current = frame[assigned_column(node["condition_id"])].to_numpy(dtype=bool)
            state = np.where(current, "assigned", "missing").astype(object)
            if parent is not None:
                before = frame[assigned_column(node["parent_condition_id"])].to_numpy(dtype=bool)
                state[current & ~before] = "new"
            state[~evaluated] = None
            frame[column] = state
            nodes.append({"node_row": node["node_row"], "label": node["pattern_set_name"], "column": column,
                          "pathLabel": node["path_label"], "isRoot": bool(node["is_root"])})
        for key, column in (("datasetId", "dataset_id"), ("recordIndex", "record_index"),
                            ("sampleId", "sample_id"), ("sampleIndex", "sample_index")):
            if filters.get(key) not in (None, "", "all"):
                frame = frame[frame[column] == int(filters[key])]
        if filters.get("adduct"):
            frame = frame[frame["adduct"].str.contains(str(filters["adduct"]), case=False, regex=False)]
        for key, column in (("collisionEnergy", "collision_energy"), ("mz", "mz"), ("intensity", "intensity"),
                            ("relativeIntensity", "relative_intensity")):
            low, high = _number(filters.get(key + "Min")), _number(filters.get(key + "Max"))
            if low is not None:
                frame = frame[frame[column] >= low]
            if high is not None:
                frame = frame[frame[column] <= high]
        if filters.get("precursor") in ("precursor", "non-precursor"):
            frame = frame[frame["is_precursor"] == (filters["precursor"] == "precursor")]
        node_filter = filters.get("nodeState")
        if node_filter not in (None, "", "all"):
            # "<node_row>:<state>" (state: new | assigned | missing), or "any-new" / "no-new".
            edge_columns = [item["column"] for item in nodes if not item["isRoot"]]
            if node_filter in ("any-new", "no-new"):
                hit = (frame[edge_columns] == "new").any(axis=1) if edge_columns else pd.Series(False, index=frame.index)
                frame = frame[hit if node_filter == "any-new" else ~hit]
            else:
                node_key, _, state = str(node_filter).partition(":")
                column = f"node::{int(node_key)}"
                if column not in frame.columns:
                    frame = frame.iloc[0:0]
                elif state == "assigned":
                    frame = frame[frame[column].isin(["assigned", "new"])]
                else:
                    frame = frame[frame[column] == state]
        if sort in frame.columns:
            order = [sort, *(column for column in ("sample_index", "mz") if column != sort)]
            frame = frame.sort_values(order, ascending=[not descending, *([True] * (len(order) - 1))],
                                      kind="stable", na_position="last")
        else:
            frame = frame.sort_values(["sample_index", "mz"], kind="stable", na_position="last")
        total = len(frame)
        page = frame.iloc[max(0, int(offset)):max(0, int(offset)) + max(1, int(limit))]
        drop = [column for column in page.columns if column.startswith(("assigned::", "count::"))]
        rows = [{key: _plain(value) for key, value in record.items()} for record in page.drop(columns=drop).to_dict("records")]
        return {"compoundId": compound_id, "total": total, "offset": int(offset), "rows": rows, "nodes": nodes}

    # -- C --------------------------------------------------------------------------
    def _fragmenter_params(self) -> dict | None:
        return (self.configuration().get("modelConfig") or {}).get("fragmenter_params")

    def _path_files(self, compound_id: str, node_row: int) -> list[tuple[dict, Path]]:
        """Own tree files of the conditions on a node's path, root first."""
        with self._file() as handle:
            conditions = self._conditions_of(handle, self.compound_row(handle, compound_id))
        result, seen = [], set()
        for node in self.node_path(node_row):
            row = conditions.get(node["condition_id"])
            if row is None or not row["detail_path"] or row["detail_path"] in seen:
                continue
            seen.add(row["detail_path"])
            result.append((node, self.root / row["detail_path"]))
        return result

    def peak_detail(self, compound_id: str, sample_id: int, peak_index: int, node_row: int) -> dict:
        """Assignments of one peak under a node's condition, labelled with the set that introduced each tree."""
        from .details import peak_assignments

        assignments, seen = [], set()
        for node, path in self._path_files(compound_id, int(node_row)):
            self.loaded_files.append(str(path))
            for item in peak_assignments(path, sample_id, peak_index, self._fragmenter_params()):
                identity = (item["fragmentSmiles"], item["formula"], item["ionAdduct"], tuple(item["reactionPatterns"]))
                if identity in seen:
                    continue
                seen.add(identity)
                assignments.append(dict(item, origin=node["pattern_set_name"]))
        with self._file() as handle:
            row = self.compound_row(handle, compound_id)
            smiles = Table(handle["compounds"]).column("canonical_smiles", slice(row, row + 1))[0]
        node = self.node(int(node_row))
        return {"compoundId": compound_id, "smiles": smiles, "sampleId": int(sample_id), "peakIndex": int(peak_index),
                "nodeRow": int(node_row), "conditionLabel": node["condition_label"], "assignments": assignments}

    def compound_reaction_combinations(self, compound_id: str, node_row: int) -> dict:
        """Reaction-combination and reaction-count breakdown under one node's condition (loads its C files only)."""
        from fredis.analysis.prepared_result import iter_prepared_samples

        combinations: dict[str, set] = defaultdict(set)
        counts: dict[str, set] = defaultdict(set)
        label = self.node(int(node_row))["condition_label"]
        for _, path in self._path_files(compound_id, int(node_row)):
            self.loaded_files.append(str(path))
            for sample in iter_prepared_samples(path, self._fragmenter_params()):
                for peak in sample["peaks"]:
                    for match in peak.get("matches", ()):
                        identity = (sample["sampleIndex"], peak["index"])
                        combinations[" + ".join(match.get("reactionPatterns") or ()) or "Source"].add(identity)
                        bucket = "4+" if match.get("reactionCount", 0) >= 4 else str(match.get("reactionCount", 0))
                        counts[bucket].add(identity)
        return {"combinations": sorted(({"reactionPatterns": key, "assignedPeaks": len(value)}
                                        for key, value in combinations.items()),
                                       key=lambda item: (-item["assignedPeaks"], item["reactionPatterns"])),
                "reactionCounts": [{"condition": label, "reactionCount": bucket, "assignedPeaks": len(value)}
                                   for bucket, value in sorted(counts.items())]}
