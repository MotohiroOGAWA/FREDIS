"""JSON bridge from the Workbench webview to Reaction Pattern Analysis results.

PatternSet-tree results (``*.reactionpatternanalysis.pft``, HDF5) are answered by
``ReactionPatternAnalysisResult`` with lazy B/C loading behind ``run()``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

# Native libraries loaded with fredis (e.g. UCX via torch) may print warnings
# straight to file descriptor 1, which would corrupt the JSON-lines protocol.
# Keep a private copy of stdout for responses and send fd 1 to stderr before
# those libraries are imported.
# UCX's optional VFS service needs an inotify watch on /tmp; without a free
# watch it prints warnings. The result reader never uses it.
os.environ.setdefault("UCX_VFS_ENABLE", "n")

if __name__ == "__main__":
    _PROTOCOL = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
else:
    _PROTOCOL = sys.stdout

from fredis.analysis.reaction_pattern import (
    ReactionPatternAnalysisResult, inspect_analysis_source, is_result_file, open_reaction_pattern_analysis,
)

QUERY_LABELS = {
    "info": "Reading result metadata",
    "overallSummary": "Reading overall summary",
    "nodeSummary": "Reading tree edges",
    "nodeDistribution": "Reading edge distribution",
    "scoreFrequency": "Calculating score frequencies",
    "compoundList": "Filtering compounds",
    "compoundSummary": "Reading compound summary",
    "compoundNodeDistribution": "Reading compound × edge distribution",
    "sampleSummary": "Reading sample summary",
    "compoundPeaks": "Loading compound peaks",
    "peakDetail": "Loading peak detail",
    "compoundReactionCombinations": "Loading compound reaction combinations",
}


def _svg(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    molecule = Chem.MolFromSmiles(smiles or "")
    if molecule is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(360, 240)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def run(request: dict, cache: dict | None = None, progress=None) -> dict:
    if request.get("action") == "inspectSource":
        return inspect_analysis_source(request["path"], request.get("kind", "auto"))
    if request.get("action") == "detect":
        return {"isResult": is_result_file(request["path"])}
    if cache is None:
        store = open_reaction_pattern_analysis(request["path"])
        try:
            return _query(store, request, progress)
        finally:
            store.close()
    path = Path(request["path"]).resolve()
    modified = path.stat().st_mtime_ns
    if cache.get("path") != str(path) or cache.get("modified") != modified:
        if cache.get("store") is not None:
            cache["store"].close()
        cache.update(path=str(path), modified=modified,
                     store=open_reaction_pattern_analysis(path))
    return _query(cache["store"], request, progress)


def _query(store, request: dict, progress=None) -> dict:
    action = request.get("action", "info")
    labels = QUERY_LABELS
    if action not in labels:
        raise ValueError(f"Unknown reaction pattern analysis action: {action}")
    if progress is not None:
        progress({"phase": "query", "message": labels[action], "indeterminate": True})
    result = _result_query(store, request, action)
    if progress is not None:
        progress({"phase": "complete", "message": labels[action], "percent": 100.0})
    return result


def _result_query(store: ReactionPatternAnalysisResult, request: dict, action: str) -> dict:
    """A-only actions never read B or C; compoundPeaks reads one B; peakDetail/compoundReactionCombinations read C."""
    if action == "info":
        return {**store.info(), "loadedFiles": list(store.loaded_files)}
    if action == "overallSummary":
        return {"overall": store.overall_summary()}
    if action == "nodeSummary":
        return {"rows": store.node_summary()}
    if action == "nodeDistribution":
        return {"nodeRow": int(request["nodeRow"]), "rows": store.node_distribution(int(request["nodeRow"]))}
    if action == "scoreFrequency":
        return store.score_frequency(str(request.get("metric") or "intensity_coverage"),
                                     without_precursor=bool(request.get("withoutPrecursor")),
                                     bins=int(request.get("bins", 20)), filters=request.get("filters"))
    if action == "compoundList":
        return store.compound_list(request.get("filters"), sort=str(request.get("sort") or "compound_id"),
                                   descending=bool(request.get("descending")),
                                   offset=max(0, int(request.get("offset", 0))),
                                   limit=min(2000, max(1, int(request.get("limit", 200)))))
    if action == "compoundSummary":
        return store.compound_summary(str(request["compoundId"]))
    if action == "compoundNodeDistribution":
        return store.compound_node_distribution(str(request["compoundId"]), int(request["nodeRow"]))
    if action == "sampleSummary":
        return {"rows": store.sample_summary(str(request["compoundId"]))}
    if action == "compoundPeaks":
        return store.compound_peaks(str(request["compoundId"]), request.get("filters"),
                                    sort=request.get("sort"), descending=bool(request.get("descending")),
                                    offset=max(0, int(request.get("offset", 0))),
                                    limit=min(20000, max(1, int(request.get("limit", 5000)))))
    if action == "peakDetail":
        result = store.peak_detail(str(request["compoundId"]), int(request["sampleId"]), int(request["peakIndex"]),
                                   int(request["nodeRow"]))
        result["precursorSvg"] = _svg(result["smiles"])
        for assignment in result["assignments"]:
            assignment["fragmentSvg"] = _svg(assignment["fragmentSmiles"])
        return result
    if action == "compoundReactionCombinations":
        return store.compound_reaction_combinations(str(request["compoundId"]), int(request["nodeRow"]))
    raise ValueError(f"Unknown reaction pattern analysis action: {action}")


def _response(request: dict, cache: dict | None = None, progress=None) -> dict:
    request_id = request.pop("requestId", None)
    try:
        return {"requestId": request_id, "ok": True, "result": run(request, cache, progress)}
    except Exception as exc:
        return {"requestId": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _dumps(value) -> str:
    return json.dumps(_finite(value), ensure_ascii=False, default=_json_default)


def _finite(value):
    """NaN is not valid JSON for the webview; report it as null."""
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


if __name__ == "__main__":
    if "--server" in sys.argv:
        state = {}
        try:
            for line in sys.stdin:
                if line.strip():
                    request = json.loads(line)
                    request_id = request.get("requestId")

                    def emit_progress(value):
                        print(_dumps({"requestId": request_id, "progress": value}), file=_PROTOCOL, flush=True)

                    print(_dumps(_response(request, state, emit_progress)), file=_PROTOCOL, flush=True)
        finally:
            if state.get("store") is not None:
                state["store"].close()
    else:
        print(_dumps(_response(json.load(sys.stdin))), file=_PROTOCOL, flush=True)
