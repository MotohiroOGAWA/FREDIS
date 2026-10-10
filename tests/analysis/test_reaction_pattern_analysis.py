"""Shared input helpers: CLI option parsing, source specs and the output directory.

The PatternSet-tree pipeline itself is covered by test_reaction_pattern_analysis_tree.py.
"""
import json
import time

import pytest

from fredis.analysis.reaction_pattern.sources import normalize_analysis_sources
from fredis.analysis.reaction_pattern.workspace import _activity, initialize_output_directory


def test_cli_pattern_sets_and_nodes_build_one_tree_configuration(tmp_path):
    from fredis.cli.analysis.commands.reaction_pattern import build_tree, parse_node_option, parse_pattern_set_option
    single = tmp_path / "one.reaction.pft"
    pattern_set = tmp_path / "many.reactionset.pft"
    single.write_text(json.dumps({"name": "a", "reactant_smarts": "C", "products": []}))
    pattern_set.write_text(json.dumps({"reaction_pattern_set": {"name": "many", "patterns": [
        {"name": "b", "reactant_smarts": "O", "products": []},
        {"name": "c", "reactant_smarts": "N", "products": []},
    ]}}))
    sets = [parse_pattern_set_option(str(single)), parse_pattern_set_option("custom=" + str(pattern_set))]
    # A single pattern becomes a one-pattern set named after the pattern.
    assert (sets[0].id, sets[0].name, len(sets[0].patterns)) == ("a", "a", 1)
    assert (sets[1].id, sets[1].name, [row["name"] for row in sets[1].patterns]) == ("custom", "custom", ["b", "c"])
    tree = build_tree(None, sets, [parse_node_option("r=a"), parse_node_option("x=custom@r"),
                                   parse_node_option("y=custom")])
    assert [(node.id, node.pattern_set, node.parent) for node in tree.nodes] == [
        ("r", "a", None), ("x", "custom", "r"), ("y", "custom", None)]
    with pytest.raises(ValueError, match="ID=PATTERN_SET"):
        parse_node_option("broken")


def test_source_specs_keep_columns_per_dataset(monkeypatch):
    import fredis.analysis.reaction_pattern.sources as module
    monkeypatch.setattr(module, "inspect_analysis_source", lambda path, kind="auto": {"kind": "dataset", "path": path})
    rows = normalize_analysis_sources(sources=[
        {"path": "one.msds", "columns": {"smiles": "CanonicalSMILES"}},
        {"path": "two.msp", "columns": {"adduct": "PRECURSORTYPE"}},
    ])
    assert rows[0]["columns"]["smiles"] == "CanonicalSMILES"
    assert rows[0]["columns"]["adduct"] == "AdductType"
    assert rows[1]["columns"]["smiles"] == "SMILES"
    assert rows[1]["columns"]["adduct"] == "PRECURSORTYPE"


def test_activity_reports_start_and_completion_without_progress_bar_interruptions(capsys):
    with _activity("Long database operation"):
        time.sleep(0.001)
    progress = capsys.readouterr().err
    assert "Long database operation: started" in progress
    assert "Long database operation: done" in progress
    assert ": working" not in progress


def test_cli_source_specs_accept_files(tmp_path):
    from fredis.cli.analysis.commands.reaction_pattern import parse_source_specs
    spec_file = tmp_path / "sources.json"
    spec_file.write_text(json.dumps([{"path": "one.msds"}, {"path": "old/result.reactionpatternanalysis.pft"}]))
    assert len(parse_source_specs(["@" + str(spec_file)])) == 2


def test_missing_overwrite_prompts_and_replaces_the_complete_directory(tmp_path, monkeypatch):
    import fredis.analysis.reaction_pattern.workspace as module
    output = tmp_path / "existing"
    stale = output / "nested" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    prompts = []
    monkeypatch.setattr(module.sys, "stdin", type("TTY", (), {"isatty": lambda self: True})())
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "yes")

    effective = initialize_output_directory(output, overwrite=False)

    assert effective is True
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert "entire directory" in prompts[0]


def test_declined_overwrite_keeps_the_complete_directory(tmp_path, monkeypatch):
    import fredis.analysis.reaction_pattern.workspace as module
    output = tmp_path / "existing"
    stale = output / "nested" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    monkeypatch.setattr(module.sys, "stdin", type("TTY", (), {"isatty": lambda self: True})())
    monkeypatch.setattr("builtins.input", lambda _: "n")

    with pytest.raises(FileExistsError, match="Cancelled"):
        initialize_output_directory(output, overwrite=False)

    assert stale.read_text() == "old"
