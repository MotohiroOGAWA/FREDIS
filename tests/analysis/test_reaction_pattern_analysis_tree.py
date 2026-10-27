"""PatternSet-tree Reaction Pattern Analysis: tree configuration, cumulative node
conditions, edge comparisons, cached Source reactions, A/B/C result layout,
lazy reading and adding nodes to an existing result.

Fixture chemistry (every set holds one pattern of nonring_single_bond):

    CCO  4 spectra: [M+H]+ CE10/CE20 + [M+Na]+ CE20 (dataset 1), [M+H]+ CE40 (dataset 2)
         ("OCC" in dataset 1 is the same compound)
    CC   1 spectrum with only its precursor peak

    H (halogen)  matches nothing    → root trees assign only the precursor
    O            matches CCO        → new peaks
    N            matches nothing    → edge never applies
    C            matches CCO and CC

Tree::

    H ─┬─ O ─── C        (H, H+O, H+O+C)
       └─ N              (H+N == H for every compound)
    C ─── O              (C, C+O)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from fredis.analysis.reaction_pattern import ReactionPatternAnalysisResult, TreeConfiguration, generate_reaction_pattern_analysis
from fredis.analysis.reaction_pattern import applicability, chemistry
from fredis.analysis.reaction_pattern.compounds import compound_id_for
from fredis.analysis.reaction_pattern.constants import assigned_column, count_column
from fredis.analysis.reaction_pattern.summary import distribution
from fredis.analysis.reaction_pattern.tree import PatternRegistry, resolve_tree

PATTERN_SET = Path("fredis/presets/fragment/reaction_pattern_sets/nonring_single_bond/nonring_single_bond.reactionset.pft")
PEAKS = [[47.049, 1.0], [31.018, 0.6], [29.039, 0.3], [19.018, 0.2], [15.023, 0.1]]
CCO, CC = compound_id_for("CCO"), compound_id_for("CC")
NAMES = {"H": "nonring_halogen_single_bond", "O": "nonring_o_single_bond", "N": "nonring_n_single_bond",
         "C": "nonring_c_single_bond"}


def _patterns() -> dict:
    return {row["name"]: row for row in json.loads(PATTERN_SET.read_text())["reaction_pattern_set"]["patterns"]}


def _set(key: str, *names: str) -> dict:
    patterns = _patterns()
    return {"id": key, "name": key, "patterns": [patterns[NAMES[name]] for name in (names or (key,))]}


def _tree(*extra_nodes: dict, extra_sets=()) -> dict:
    return {"patternSets": [_set("H"), _set("O"), _set("N"), _set("C"), *extra_sets],
            "nodes": [{"id": "h", "patternSet": "H", "parent": None},
                      {"id": "ho", "patternSet": "O", "parent": "h"},
                      {"id": "hoc", "patternSet": "C", "parent": "ho"},
                      {"id": "hn", "patternSet": "N", "parent": "h"},
                      {"id": "c", "patternSet": "C", "parent": None},
                      {"id": "co", "patternSet": "O", "parent": "c"}, *extra_nodes]}


def _dataset(path: Path, rows: list[tuple[dict, list]]) -> Path:
    from fredis.data import MSDataset
    from fredis.data.core.PeakSeries import PeakSeries

    peaks = [peak for _, values in rows for peak in values]
    offsets = np.cumsum([0] + [len(values) for _, values in rows])
    MSDataset(pd.DataFrame([row for row, _ in rows]),
              PeakSeries(np.asarray(peaks, dtype=float), np.asarray(offsets, dtype=np.int64))).save(str(path))
    return path


def _spectrum(smiles, adduct, energy, precursor):
    return {"SMILES": smiles, "AdductType": adduct, "CollisionEnergy": energy, "PrecursorMZ": precursor}


def _chemistry(max_reaction_count: int = 2) -> dict:
    from tests.ml.reaction.TestReactionPipeline import chemistry

    value = chemistry()
    builder = value["fragmenter_params"]["fragment_ion_tree_builder"]
    builder["max_reaction_count"] = max_reaction_count
    # Patterns of the chemistry configuration never enter the analysis.
    builder["reaction_pattern_set"] = {"name": "ignored", "patterns": [_patterns()[NAMES["O"]]]}
    return value


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("analysis-inputs")
    first = _dataset(root / "first.msds", [
        (_spectrum("CCO", "[M+H]+", 10, 47.049), PEAKS),
        (_spectrum("OCC", "[M+H]+", 20, 47.049), PEAKS[:3]),
        (_spectrum("CC", "[M+H]+", 20, 31.054), [[31.054, 1.0]]),
        (_spectrum("CCO", "[M+Na]+", 20, 69.031), [[69.031, 1.0], [53.0, 0.5]]),
    ])
    second = _dataset(root / "second.msds", [(_spectrum("CCO", "[M+H]+", 40, 47.049), PEAKS)])
    return [first, second]


def _run(output: Path, inputs, tree=None, **kwargs) -> Path:
    return generate_reaction_pattern_analysis(sources=[{"path": str(path)} for path in inputs], output_dir=output,
                                      chemistry_config=_chemistry(), tree=tree or _tree(), overwrite=True, **kwargs)


@pytest.fixture(scope="module")
def analysis(tmp_path_factory, inputs):
    """One run with spies on worker tasks and Source reaction enumeration."""
    from fredis.domain.fragment import reaction as reaction_package

    tasks, reactions = [], []
    original_task, original_reactions = chemistry.run_task, reaction_package.create_reaction_instances
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chemistry, "run_task", lambda task: tasks.append(
            (task.compound.compound_id, [(step.condition_id, step.parent_condition_id) for step in task.steps]))
            or original_task(task))
        patch.setattr(reaction_package, "create_reaction_instances", lambda source, pattern_set: reactions.append(
            (source.smiles, tuple(pattern.name for pattern in pattern_set))) or original_reactions(source, pattern_set))
        result = _run(tmp_path_factory.mktemp("analysis-result") / "result", inputs)
    with ReactionPatternAnalysisResult(result) as reader:
        nodes = {row["node_id"]: row for row in reader.node_summary()}
    return {"path": result, "tasks": tasks, "reactions": reactions, "nodes": nodes}


# -- Tree Configuration ----------------------------------------------------------------
def test_tree_configuration_validates_structure():
    with pytest.raises(ValueError, match="unknown pattern set"):
        TreeConfiguration.from_dict({"patternSets": [_set("H")], "nodes": [{"id": "a", "patternSet": "X"}]})
    with pytest.raises(ValueError, match="cycle"):
        TreeConfiguration.from_dict({"patternSets": [_set("H")], "nodes": [
            {"id": "a", "patternSet": "H", "parent": "b"}, {"id": "b", "patternSet": "H", "parent": "a"}]})
    with pytest.raises(ValueError, match="Duplicate tree node"):
        TreeConfiguration.from_dict({"patternSets": [_set("H")], "nodes": [
            {"id": "a", "patternSet": "H"}, {"id": "a", "patternSet": "H"}]})
    clash = dict(_patterns()[NAMES["O"]], reactant_smarts="[#8]")
    with pytest.raises(ValueError, match="share the name"):
        TreeConfiguration.from_dict({"patternSets": [_set("O"), {"id": "X", "name": "X", "patterns": [clash]}],
                                     "nodes": [{"id": "a", "patternSet": "O"}]})
    with pytest.raises(ValueError, match="at least one node"):
        resolve_tree(TreeConfiguration.from_dict({"patternSets": [_set("H")], "nodes": []}))


def test_nodes_are_cumulative_conditions_and_identical_conditions_are_shared():
    tree = resolve_tree(TreeConfiguration.from_dict(_tree(
        {"id": "ch", "patternSet": "H", "parent": "co"}, {"id": "hc", "patternSet": "C", "parent": "h"},
        {"id": "hco", "patternSet": "O", "parent": "hc"})))
    nodes = {node.node_id: node for node in tree.nodes}
    labels = {key: node.condition_label for key, node in nodes.items()}
    assert labels["hoc"] == "H + O + C" and labels["co"] == "C + O" and labels["ch"] == "C + O + H"
    # One pattern set may appear at any number of nodes; H→O→C and C→O→H and
    # H→C→O reach the same condition, which is calculated once.
    assert nodes["hoc"].condition_id == nodes["ch"].condition_id == nodes["hco"].condition_id
    assert nodes["ho"].added_pattern_ids == tree.set_pattern_ids["O"]
    assert nodes["ho"].parent_condition_id == nodes["h"].condition_id
    assert [node.node_id for node in tree.nodes][:4] == ["h", "ho", "hoc", "hn"]


def test_pattern_ids_are_kept_from_a_previous_registry():
    registry = PatternRegistry([{"pattern_id": 7, "definition": _patterns()[NAMES["C"]]}])
    tree = resolve_tree(TreeConfiguration.from_dict(_tree()), registry)
    assert tree.set_pattern_ids["C"] == (7,)
    assert 7 not in {tree.set_pattern_ids[key][0] for key in ("H", "O", "N")}


# -- calculation -----------------------------------------------------------------------------
def test_source_reactions_are_enumerated_once_per_compound_and_pattern(analysis):
    # C and O are each used at two nodes, but every (compound, pattern) is matched once.
    counts = pd.Series(analysis["reactions"]).value_counts()
    assert counts.max() == 1
    assert {pattern for _, (pattern,) in analysis["reactions"]} == set(NAMES.values()) - {NAMES["N"]}


def test_each_compound_is_one_task_and_unaffected_edges_are_not_calculated(analysis):
    tasks = dict(analysis["tasks"])
    nodes = analysis["nodes"]
    condition = lambda key: nodes[key]["condition_id"]
    assert sorted(tasks) == sorted([CCO, CC])
    # CC: H, C and H+O+C (O does not match CC, so H+O equals H and H+O+C is seeded from H).
    assert sorted(step for step, _ in tasks[CC]) == sorted([condition("h"), condition("hoc"), condition("c")])
    assert dict(tasks[CC])[condition("hoc")] == condition("h")
    # H+N never applies; O applies to CCO only.
    assert condition("hn") not in dict(tasks[CCO]) and condition("hn") not in dict(tasks[CC])
    assert dict(tasks[CCO])[condition("ho")] == condition("h")


def test_child_condition_equals_a_full_build_of_the_cumulative_condition(analysis, inputs, tmp_path):
    full = _run(tmp_path / "full", inputs, {"patternSets": [_set("HOC", "H", "O", "C")],
                                           "nodes": [{"id": "r", "patternSet": "HOC"}]})
    hoc = analysis["nodes"]["hoc"]["condition_id"]
    for compound in (CCO, CC):
        incremental = pd.read_parquet(analysis["path"].parent / "compounds" / f"{compound}.peaks.parquet")
        complete = pd.read_parquet(full.parent / "compounds" / f"{compound}.peaks.parquet")
        assert incremental[assigned_column(hoc)].tolist() == complete[assigned_column(0)].tolist()


# -- edge comparison ----------------------------------------------------------------------------
def test_roots_have_absolute_metrics_only_and_edges_report_parent_child_and_delta(analysis):
    nodes = analysis["nodes"]
    root, edge = nodes["h"], nodes["ho"]
    assert root["is_root"] and root["parent_weighted_peak_coverage"] is None
    assert root["weighted_delta_peak_coverage"] is None and root["applicable_compound_count"] == 2
    assert not edge["is_root"] and edge["applicable_compound_count"] == 1 and edge["calculated_compound_count"] == 1
    assert edge["weighted_peak_coverage"] > edge["parent_weighted_peak_coverage"]
    assert math.isclose(edge["weighted_delta_peak_coverage"],
                        edge["weighted_peak_coverage"] - edge["parent_weighted_peak_coverage"])
    assert edge["newly_assigned_peak_count"] > 0 and edge["parent_condition_label"] == "H"
    assert nodes["hn"]["applicable_compound_count"] == 0 and nodes["hn"]["newly_assigned_peak_count"] == 0
    # Overall coverage spans the whole population: H+N equals H.
    assert nodes["hn"]["overall_peak_coverage"] == root["overall_peak_coverage"]
    assert nodes["hoc"]["overall_peak_coverage"] >= nodes["ho"]["overall_peak_coverage"] >= root["overall_peak_coverage"]


def test_sample_rows_follow_delta_definitions(analysis):
    with ReactionPatternAnalysisResult(analysis["path"]) as reader:
        rows = reader.table_records("sample_node_summary")
    by_node = {row["node_id"]: row["node_row"] for row in analysis["nodes"].values()}
    roots = [row for row in rows if row["node_row"] == by_node["h"]]
    edges = [row for row in rows if row["node_row"] == by_node["ho"]]
    assert len(roots) == 5 and len(edges) == 4  # every spectrum; only CCO for the O edge
    assert all(row["parent_peak_coverage"] is None and row["delta_peak_coverage"] is None for row in roots)
    for row in edges:
        assert math.isclose(row["delta_peak_coverage"], row["peak_coverage"] - row["parent_peak_coverage"])
        assert math.isclose(row["delta_intensity_coverage_without_precursor"],
                            row["intensity_coverage_without_precursor"] - row["parent_intensity_coverage_without_precursor"])
        assert row["newly_assigned_peak_count"] == row["assigned_peak_count"] - row["parent_assigned_peak_count"]


def test_compound_node_distribution_is_aggregated_from_sample_rows(analysis):
    node_row = analysis["nodes"]["ho"]["node_row"]
    with ReactionPatternAnalysisResult(analysis["path"]) as reader:
        result = reader.compound_node_distribution(CCO, node_row)
        summary = next(row for row in reader.compound_summary(CCO)["nodes"] if row["node_row"] == node_row)
    values = [row["delta_peak_coverage"] for row in result["samples"]]
    stats = next(row for row in result["distributions"] if row["metric"] == "delta_peak_coverage"
                 and row["without_precursor"] == 0)
    assert stats["n"] == 4 and math.isclose(stats["mean"], float(np.mean(values)))
    assert math.isclose(summary["mean_delta_peak_coverage"], stats["mean"])
    assert sum(stats["histogram"]["counts"]) == 4


def test_distribution_helper_handles_empty_and_count_metrics():
    assert distribution([], "delta_peak_coverage")["n"] == 0
    counts = distribution([0, 2, 4], "newly_assigned_peak_count")
    assert counts["max"] == 4 and json.loads(counts["histogram_json"])["edges"][-1] == 4


# -- A/B/C layout -------------------------------------------------------------------------------
def test_result_a_is_hdf5_with_tree_tables_and_indexes(analysis):
    with h5py.File(analysis["path"], "r") as handle:
        assert handle.attrs["schema_version"] == 1 and handle.attrs["analysis_status"] == "completed"
        for name in ("pattern_sets", "patterns", "conditions", "nodes", "compound_node_summary", "node_summary",
                     "node_distribution", "sample_node_summary", "compound_condition"):
            assert name in handle
        for name in ("compound_node_by_compound", "compound_node_by_node", "sample_node_by_node",
                     "compound_condition_by_compound"):
            assert name in handle["indexes"]
        configuration = json.loads(handle["configuration/json"].asstr()[()])
    assert [node["id"] for node in configuration["tree"]["nodes"]] == ["h", "ho", "hoc", "hn", "c", "co"]
    assert {row["name"] for row in configuration["chemistryConfig"]["fragmenter_params"]["fragment_ion_tree_builder"]
            ["reaction_pattern_set"]["patterns"]} == set(NAMES.values())
    assert not any("baseline" in name for name in configuration["run"])
    assert (analysis["path"].parent / "tree.reaction-pattern-analysis.json").exists()


def test_peak_tables_hold_one_mask_per_resolved_condition(analysis):
    frame = pd.read_parquet(analysis["path"].parent / "compounds" / f"{CCO}.peaks.parquet")
    nodes = analysis["nodes"]
    for key in ("h", "ho", "hoc", "c", "co"):
        assert assigned_column(nodes[key]["condition_id"]) in frame.columns
        assert count_column(nodes[key]["condition_id"]) in frame.columns
    child, parent = (frame[assigned_column(nodes[key]["condition_id"])] for key in ("ho", "h"))
    assert (child | ~parent).all() and (child & ~parent).sum() > 0
    assert len(frame) == 5 + 3 + 5 + 2


def test_condition_trees_are_saved_once_with_seeds(analysis):
    from fredis.analysis.prepared_result import load_prepared_structure

    with ReactionPatternAnalysisResult(analysis["path"]) as reader:
        rows = reader.table_records("compound_condition")
    files = [row["detail_path"] for row in rows if row["detail_path"]]
    assert len(files) == len(set(files))
    hn = analysis["nodes"]["hn"]["condition_id"]
    assert all(row["alias_of"] == analysis["nodes"]["h"]["condition_id"] for row in rows if row["condition_id"] == hn)
    root_c = next(row["detail_path"] for row in rows
                  if row["compound_id"] == CCO and row["condition_id"] == analysis["nodes"]["c"]["condition_id"])
    _, metadata = load_prepared_structure(analysis["path"].parent / root_c)
    assert metadata["reaction_sequence_compounds"]  # seeds for children added later


# -- reader ---------------------------------------------------------------------------------------
def test_reader_loads_b_and_c_lazily_and_labels_node_states(analysis):
    nodes = analysis["nodes"]
    with ReactionPatternAnalysisResult(analysis["path"]) as reader:
        info = reader.info()
        assert [row["node_id"] for row in info["nodeSummary"]] == ["h", "ho", "hoc", "hn", "c", "co"]
        assert {row["pattern_set_id"] for row in info["tree"]["patternSets"]} == {"H", "O", "N", "C"}
        reader.compound_list({"nodeRow": nodes["ho"]["node_row"]})
        reader.score_frequency()
        assert reader.loaded_files == []
        peaks = reader.compound_peaks(CCO)
        assert len(reader.loaded_files) == 1
        states = {item["node_row"]: item["column"] for item in peaks["nodes"]}
        assert nodes["hn"]["node_row"] not in states  # not applicable to CCO
        new = [row for row in peaks["rows"] if row[states[nodes["ho"]["node_row"]]] == "new"]
        assert new and all(row[states[nodes["h"]["node_row"]]] == "missing" for row in new)
        filtered = reader.compound_peaks(CCO, {"nodeState": f"{nodes['ho']['node_row']}:new"})
        assert filtered["total"] == len([row for row in reader.compound_peaks(CCO, limit=1000)["rows"]
                                         if row[states[nodes["ho"]["node_row"]]] == "new"])
        detail = reader.peak_detail(CCO, new[0]["sample_id"], new[0]["peak_index"], nodes["ho"]["node_row"])
        assert detail["assignments"] and {row["origin"] for row in detail["assignments"]} <= {"H", "O"}
        assert any(path.endswith(".preft.pt") for path in reader.loaded_files)
        combinations = reader.compound_reaction_combinations(CCO, nodes["hoc"]["node_row"])
        assert combinations["combinations"] and combinations["reactionCounts"][0]["condition"] == "H + O + C"


def test_score_frequency_has_one_absolute_series_per_node(analysis):
    with ReactionPatternAnalysisResult(analysis["path"]) as reader:
        result = reader.score_frequency("peak_coverage")
        filtered = reader.score_frequency("peak_coverage", filters={"smiles": "CCO"})
    series = {row["label"]: row for row in result["series"]}
    assert set(series) == {"H", "H + O", "H + O + C", "H + N", "C", "C + O"}
    # Compounds an edge does not apply to inherit the parent's value.
    assert series["H + N"]["counts"] == series["H"]["counts"] and series["H + O"]["n"] == 5
    assert filtered["filteredSamples"] == 4


def test_result_directory_can_be_moved(analysis, tmp_path):
    import shutil

    moved = tmp_path / "moved"
    shutil.copytree(analysis["path"].parent, moved)
    with ReactionPatternAnalysisResult(moved / analysis["path"].name) as reader:
        node_row = analysis["nodes"]["ho"]["node_row"]
        assert reader.compound_reaction_combinations(CCO, node_row)["combinations"]


def test_failed_write_leaves_no_completed_result(inputs, tmp_path, monkeypatch):
    from fredis.analysis.reaction_pattern import hdf5_storage

    def fail(self, name, columns):
        raise RuntimeError("disk full")

    monkeypatch.setattr(hdf5_storage.ResultWriter, "write_table", fail)
    with pytest.raises(RuntimeError, match="disk full"):
        _run(tmp_path / "failed", inputs)
    assert not (tmp_path / "failed" / "result.reactionpatternanalysis.pft").exists()


def test_parallel_workers_match_serial_results(analysis, inputs, tmp_path):
    parallel = _run(tmp_path / "parallel", inputs, num_workers=2)
    with ReactionPatternAnalysisResult(parallel) as reader:
        rows = {row["node_id"]: row for row in reader.node_summary()}
    for key, row in analysis["nodes"].items():
        for column in ("weighted_peak_coverage", "newly_assigned_peak_count", "overall_peak_coverage"):
            assert rows[key][column] == row[column]


# -- adding nodes to an existing result -------------------------------------------------------------
def test_extension_calculates_only_new_conditions(analysis, inputs, tmp_path, monkeypatch):
    import shutil

    output = tmp_path / "extended"
    shutil.copytree(analysis["path"].parent, output)
    calls = {}
    original = chemistry.run_task
    monkeypatch.setattr(chemistry, "run_task", lambda task: calls.__setitem__(
        task.compound.compound_id, [step.condition_id for step in task.steps]) or original(task))
    tree = _tree({"id": "hc", "patternSet": "C", "parent": "h"}, {"id": "con", "patternSet": "N", "parent": "co"},
                 {"id": "cn", "patternSet": "NO", "parent": "c"}, extra_sets=[_set("NO", "N", "O")])
    result = generate_reaction_pattern_analysis(output_dir=output, tree=tree, extend=True)
    with ReactionPatternAnalysisResult(result) as reader:
        rows = {row["node_id"]: row for row in reader.node_summary()}
        overall = reader.overall_summary()
    # Only the new condition H+C is calculated.  C→O→N and C→NO reach the same
    # condition C+N+O; N never matches, so for every compound it equals the
    # existing C+O condition and nothing is calculated for either node.
    assert calls == {CCO: [rows["hc"]["condition_id"]], CC: [rows["hc"]["condition_id"]]}
    assert rows["cn"]["condition_id"] == rows["con"]["condition_id"] != rows["co"]["condition_id"]
    assert rows["con"]["applicable_compound_count"] == 0
    assert rows["con"]["overall_peak_coverage"] == rows["co"]["overall_peak_coverage"]
    assert rows["cn"]["overall_peak_coverage"] == rows["co"]["overall_peak_coverage"]
    for key, row in analysis["nodes"].items():
        for column in ("weighted_peak_coverage", "newly_assigned_peak_count", "overall_peak_coverage"):
            assert rows[key][column] == row[column]
    assert overall["reusedConditionTrees"] > 0
    assert not (output / "compounds.new").exists() and not (output / "compounds.old").exists()


def test_extension_rejects_different_settings(analysis, tmp_path):
    import shutil

    output = tmp_path / "extended"
    shutil.copytree(analysis["path"].parent, output)
    with pytest.raises(ValueError, match="settings of the existing result"):
        generate_reaction_pattern_analysis(output_dir=output, extend=True, max_reaction_combinations=5)
    with ReactionPatternAnalysisResult(output / "result.reactionpatternanalysis.pft") as reader:
        assert len(reader.nodes()) == 6


# -- CLI / Workbench backend ------------------------------------------------------------------------
def test_cli_builds_the_tree_from_pattern_sets_and_nodes(monkeypatch, capsys, tmp_path, inputs):
    from fredis.cli.analysis.commands.reaction_pattern import ReactionPatternAnalysisCommand
    import argparse

    patterns = _patterns()
    (tmp_path / "h.pft").write_text(json.dumps(patterns[NAMES["H"]]))
    (tmp_path / "oc.pft").write_text(json.dumps({"reaction_pattern_set": {"name": "OC", "patterns": [
        patterns[NAMES["O"]], patterns[NAMES["C"]]]}}))
    chemistry = tmp_path / "chemistry.json"
    chemistry.write_text(json.dumps(_chemistry()))
    parser = argparse.ArgumentParser()
    command = ReactionPatternAnalysisCommand()
    command.configure(parser)
    args = parser.parse_args(["--source", str(inputs[0]), "--output-dir", str(tmp_path / "out"), "--params", str(chemistry),
                              "--pattern-set", "H=" + str(tmp_path / "h.pft"), "--pattern-set", str(tmp_path / "oc.pft"),
                              "--node", "r=H", "--node", "x=OC@r", "--overwrite", "1"])
    command.run(args)
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    with ReactionPatternAnalysisResult(printed["result"]) as reader:
        assert [row["condition_label"] for row in reader.nodes()] == ["H", "H + OC"]
    # Add a node with --extend; the existing tree, sources and settings are reused.
    args = parser.parse_args(["--output-dir", str(tmp_path / "out"), "--pattern-set", str(tmp_path / "h.pft"),
                              "--node", "y=OC", "--extend", "1"])
    command.run(args)
    with ReactionPatternAnalysisResult(tmp_path / "out" / "result.reactionpatternanalysis.pft") as reader:
        assert [row["node_id"] for row in reader.nodes()] == ["r", "x", "y"]
    # The same tree inline, without any Tree Configuration file.
    tree = {"patternSets": [{"id": "H", "name": "H", "patterns": [patterns[NAMES["H"]]]}],
            "nodes": [{"id": "r", "patternSet": "H", "parent": None}]}
    args = parser.parse_args(["--source", str(inputs[0]), "--output-dir", str(tmp_path / "inline"), "--params", str(chemistry),
                              "--tree-config-json", json.dumps(tree), "--overwrite", "1"])
    command.run(args)
    with ReactionPatternAnalysisResult(tmp_path / "inline" / "result.reactionpatternanalysis.pft") as reader:
        assert [row["condition_label"] for row in reader.nodes()] == ["H"]
    with pytest.raises(ValueError, match="either --tree-config or --tree-config-json"):
        command.run(parser.parse_args(["--output-dir", str(tmp_path / "x"), "--tree-config", "t.json",
                                       "--tree-config-json", "{}"]))
