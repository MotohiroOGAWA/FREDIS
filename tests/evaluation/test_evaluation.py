from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from evaluation.chart import box_plot_svg
from evaluation.cli import PROGRESS_PREFIX, build_parser, execute, main
from evaluation.core import EvaluationRequest, inspect_dataset, summarize
from evaluation.grouped import compare, grouped_box_plot_svg


def _write_similarity(path, table):
    payload = table.to_parquet(index=False)
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "msentity.similarity"
        handle.attrs["schema_version"] = 1
        handle.create_dataset("table.parquet", data=np.frombuffer(payload, dtype=np.uint8))
        handle.create_dataset(
            "metadata.json", data=json.dumps({"row_count": len(table)}),
            dtype=h5py.string_dtype("utf-8"),
        )
    return path


def _fixtures(tmp_path):
    spec_ids = [f"S{index}" for index in range(6)]
    similarity = _write_similarity(
        tmp_path / "scores.mssim",
        pd.DataFrame({
            "SpecID": spec_ids, "index1": range(6), "index2": range(6),
            "cosine_similarity": [0.1, 0.2, 0.3, 0.4, 1.0, 0.5],
        }),
    )
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({
        "SpecID": spec_ids,
        "AdductType": ["[M+H]+"] * 5 + ["[M+Na]+"],
        "CollisionEnergy": [5, 7, 12, 15, 25, 25],
        "PrecursorMZ": [500.0] * 6,
        "SMILES": ["CCO", "CCO", "c1ccccc1", "c1ccccc1", "CC(=O)O", "CC(=O)O"],
    }).to_csv(metadata, index=False)
    return similarity, metadata


def test_default_spec_id_join_and_categorical_boxplot(tmp_path):
    similarity, source = _fixtures(tmp_path)
    inspected = inspect_dataset(str(similarity), str(source))
    names = {column["name"] for column in inspected["columns"]}
    assert {"SpecID", "cosine_similarity", "AdductType", "CollisionEnergy"} <= names

    result = summarize(EvaluationRequest(
        str(similarity), "AdductType", metadata=str(source), include=("[M+H]+",)
    ))
    assert len(result["rows"]) == 1
    assert [row["category"] for row in result["categoryRows"]] == ["[M+H]+", "[M+Na]+"]
    box = result["rows"][0]
    assert box["category"] == "[M+H]+"
    assert box["count"] == 5
    assert box["median"] == 0.3
    assert box["whiskerHigh"] == 0.4
    assert box["outliers"] == [1.0]
    assert len(box["density"]) == 65
    assert max(point[1] for point in box["density"]) == 1.0
    violin = box_plot_svg(result, plot_type="violin")
    assert 'class="violin"' in violin
    assert 'class="whisker"' not in violin
    empty = summarize(EvaluationRequest(
        str(similarity), "AdductType", metadata=str(source), include=()
    ))
    assert empty["rows"] == []
    assert len(empty["categoryRows"]) == 2


def test_numeric_ranges_are_grouped_and_ordered(tmp_path):
    similarity, source = _fixtures(tmp_path)
    result = summarize(EvaluationRequest(
        str(similarity), "CollisionEnergy", metadata=str(source), mode="numeric",
        bins=(0, 10, 20), order=("[20,~)", "[0,10)", "[10,20)"),
    ))
    assert [row["category"] for row in result["rows"]] == ["[20,~)", "[0,10)", "[10,20)"]
    assert [row["count"] for row in result["rows"]] == [2, 2, 2]


def test_cli_emits_machine_readable_progress(tmp_path, capsys):
    similarity, _ = _fixtures(tmp_path)
    main(["inspect", "--input", str(similarity)])
    captured = capsys.readouterr()
    updates = [
        json.loads(line.removeprefix(PROGRESS_PREFIX))
        for line in captured.err.splitlines() if line.startswith(PROGRESS_PREFIX)
    ]
    assert updates[0]["percent"] == 10
    assert updates[-1] == {"percent": 100, "message": "Columns loaded"}
    assert json.loads(captured.out)["ok"] is True


def test_collision_energy_parser_converts_metadata_before_grouping(tmp_path):
    similarity, source = _fixtures(tmp_path)
    metadata = pd.read_csv(source)
    metadata["CollisionEnergy"] = ["5 eV", "7 V", "12eV", "15", "25", "25%"]
    metadata.to_csv(source, index=False)
    result = summarize(EvaluationRequest(
        str(similarity), "CollisionEnergy", metadata=str(source), mode="numeric",
        bins=(0, 10, 20), transform="collision-energy",
        precursor_mz_column="PrecursorMZ",
    ))
    assert result["groupColumn"] == "CollisionEnergy (eV)"
    assert result["transform"] == "collision-energy"
    assert [(row["category"], row["count"]) for row in result["rows"]] == [
        ("[0,10)", 2), ("[10,20)", 2), ("[20,~)", 2)
    ]


def test_smiles_heavy_atom_count_is_available_as_numeric_group(tmp_path):
    similarity, source = _fixtures(tmp_path)
    result = summarize(EvaluationRequest(
        str(similarity), "SMILES", metadata=str(source), mode="numeric",
        bins=(0, 4, 6), transform="chemical", smiles_column="SMILES",
        chemical_descriptor="HeavyAtomCount",
    ))
    assert result["groupColumn"] == "HeavyAtomCount"
    assert [(row["category"], row["count"]) for row in result["rows"]] == [
        ("[0,4)", 2), ("[4,6)", 2), ("[6,~)", 2)
    ]


def test_smarts_queries_create_independent_overlapping_groups(tmp_path):
    similarity, source = _fixtures(tmp_path)
    result = summarize(EvaluationRequest(
        str(similarity), "SMILES", metadata=str(source), transform="smarts",
        smiles_column="SMILES",
        smarts_queries=(
            ("Contains oxygen", "[#8]"), ("Aromatic", "[c]"),
            ("Silicon", "[Si]"),
        ),
    ))
    assert result["groupColumn"] == "SMARTS match"
    assert result["groupMode"] == "categorical"
    assert result["smartsQueries"] == [
        {"label": "Contains oxygen", "smarts": "[#8]"},
        {"label": "Aromatic", "smarts": "[c]"},
        {"label": "Silicon", "smarts": "[Si]"},
    ]
    oxygen, aromatic, silicon = result["rows"]
    assert (oxygen["matchedRows"], oxygen["uniqueCompounds"], oxygen["count"]) == (4, 2, 4)
    assert (aromatic["matchedRows"], aromatic["uniqueCompounds"], aromatic["count"]) == (2, 1, 2)
    assert oxygen["median"] == pytest.approx(0.35)
    assert aromatic["median"] == pytest.approx(0.35)
    assert (silicon["matchedRows"], silicon["uniqueCompounds"], silicon["count"]) == (0, 0, 0)
    assert silicon["median"] is None
    assert result["structureSummary"] == {
        "rows": 6, "validSmilesRows": 6, "missingSmilesRows": 0,
        "invalidSmilesRows": 0, "uniqueValidCompounds": 3,
        "uniqueMatchedCompounds": 3,
    }


def test_smarts_grouping_validates_queries_and_cli_json(tmp_path):
    similarity, source = _fixtures(tmp_path)
    with pytest.raises(ValueError, match="Invalid or empty SMARTS"):
        summarize(EvaluationRequest(
            str(similarity), "SMILES", metadata=str(source), transform="smarts",
            smarts_queries=(("broken", "["),),
        ))
    args = build_parser().parse_args([
        "summarize", "--input", str(similarity), "--metadata", str(source),
        "--group-column", "SMILES", "--transform", "smarts",
        "--smarts-json", json.dumps([
            {"label": "Oxygen", "smarts": "[#8]"},
            {"label": "Aromatic", "smarts": "[c]"},
        ]),
    ])
    result = execute(args)
    assert [(row["category"], row["matchedRows"]) for row in result["rows"]] == [
        ("Oxygen", 4), ("Aromatic", 2),
    ]


def test_cli_writes_transparent_colored_boxplot(tmp_path):
    similarity, source = _fixtures(tmp_path)
    image = tmp_path / "evaluation.svg"
    args = build_parser().parse_args([
        "summarize", "--input", str(similarity), "--metadata", str(source),
        "--group-column", "AdductType", "--include", json.dumps(["[M+Na]+"]),
        "--width", "500", "--height", "300", "--color", "#ff00aa",
        "--graph-opacity", "0.4", "--x-label-size", "15",
        "--y-label-size", "16", "--x-axis-title-size", "18",
        "--y-axis-title-size", "19", "--title-size", "24",
        "--title", "Custom evaluation",
        "--output-image", str(image),
    ])
    progress = []
    result = execute(args, progress=lambda percent, message: progress.append((percent, message)))
    svg = image.read_text(encoding="utf-8")
    assert result["rows"][0]["category"] == "[M+Na]+"
    assert progress[-1] == (100, "Evaluation complete")
    assert any(message == "Rendering evaluation SVG…" for _, message in progress)
    assert 'width="500" height="300"' in svg
    assert 'fill="#ff00aa"' in svg
    assert '<rect width="500" height="300" fill="#ffffff"/>' not in svg
    assert '<g class="graph" opacity="0.4">' in svg
    assert '.x-label{font-size:15px}' in svg
    assert '.y-label{font-size:16px}' in svg
    assert '.x-axis-title{font-size:18px}' in svg
    assert '.y-axis-title{font-size:19px}' in svg
    assert '.title{font-size:24px' in svg
    assert '<text x="250.00" y="30.00" text-anchor="middle" class="title">Custom evaluation</text>' in svg
    assert box_plot_svg(result, transparent=False).count('fill="#ffffff"') == 1
    styled = box_plot_svg(
        result, graph_opacity=0.4, x_label_size=15,
        y_label_size=16, x_axis_title_size=18,
        y_axis_title_size=19, title_size=24,
    )
    assert '<g class="graph" opacity="0.4">' in styled
    assert '.x-label{font-size:15px}' in styled
    assert '.y-label{font-size:16px}' in styled
    assert '.x-axis-title{font-size:18px}' in styled
    assert '.y-axis-title{font-size:19px}' in styled
    assert '.title{font-size:24px' in styled
    large = box_plot_svg(
        result, width=240, height=200, x_label_size=96, y_label_size=96,
        x_axis_title_size=96, y_axis_title_size=96, title_size=96,
    )
    large_size = re.search(r'<svg[^>]+width="([^"]+)" height="([^"]+)"', large)
    assert float(large_size.group(1)) > 240
    assert float(large_size.group(2)) > 200
    assert 'rotate(90)' in large


def test_explicit_join_column_supports_msds_metadata(tmp_path):
    app_root = Path(__file__).parents[2]
    msds = app_root / "fredis/libs/msentity/tests/test_data/sample.msds"
    identifiers = ["MassSpecID0000001", "MassSpecID0000002"]
    similarity = _write_similarity(
        tmp_path / "scores.mssim",
        pd.DataFrame({
            "IDENTIFIER": identifiers, "index1": [0, 1], "index2": [0, 1],
            "cosine_similarity": [0.9, 0.5],
        }),
    )
    inspected = inspect_dataset(str(similarity), str(msds), "IDENTIFIER")
    names = {column["name"] for column in inspected["columns"]}
    assert {"ADDUCT", "CollisionEnergy", "cosine_similarity"} <= names
    result = summarize(EvaluationRequest(
        str(similarity), "CollisionEnergy", metadata=str(msds),
        join_column="IDENTIFIER", mode="numeric", bins=(0, 25, 35),
    ))
    assert [(row["category"], row["count"]) for row in result["rows"]] == [
        ("[0,25)", 1), ("[25,35)", 1)
    ]


def test_result_and_metadata_can_use_different_join_column_names(tmp_path):
    similarity = _write_similarity(
        tmp_path / "scores.mssim",
        pd.DataFrame({
            "SourceSpecID": ["S1", "S2"],
            # A result SpecID may also exist and must not collide with the
            # metadata SpecID used as the right-hand key.
            "SpecID": ["predicted-1", "predicted-2"],
            "index1": [0, 1], "index2": [0, 1],
            "cosine_similarity": [0.8, 0.4],
        }),
    )
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({
        "SpecID": ["S1", "S2"], "SMILES": ["CCO", "O"],
    }).to_csv(metadata, index=False)

    # Auto detection prefers SourceSpecID -> SpecID.
    inspected = inspect_dataset(str(similarity), str(metadata))
    names = {column["name"] for column in inspected["columns"]}
    assert {"SourceSpecID", "SpecID", "SMILES"} <= names
    result = summarize(EvaluationRequest(
        str(similarity), "SMILES", metadata=str(metadata), transform="smarts",
        smarts_queries=(("base", "CC"),),
    ))
    assert result["rows"][0]["matchedRows"] == 1
    assert result["rows"][0]["median"] == 0.8

    # The pair can also be selected explicitly from the CLI/API.
    explicit = summarize(EvaluationRequest(
        str(similarity), "SMILES", metadata=str(metadata),
        join_column="SourceSpecID", metadata_join_column="SpecID",
        transform="smarts", smarts_queries=(("base", "CC"),),
    ))
    assert explicit["rows"][0]["matchedRows"] == 1


def test_auto_join_prefers_source_spec_id_and_ignores_missing_metadata_keys(tmp_path):
    similarity = _write_similarity(
        tmp_path / "scores.mssim",
        pd.DataFrame({
            "SourceSpecID": ["S1", "S2"], "SpecID": ["S1", "S2"],
            "index1": [0, 1], "index2": [0, 1],
            "cosine_similarity": [0.8, 0.4],
        }),
    )
    metadata = tmp_path / "predicted.csv"
    pd.DataFrame({
        "SourceSpecID": ["S1", "S2", None, None],
        "SpecID": ["predicted-1", "predicted-2", "predicted-3", "predicted-4"],
        "SMILES": ["CCO", "O", "CCC", "CCCC"],
    }).to_csv(metadata, index=False)
    result = summarize(EvaluationRequest(
        str(similarity), "SMILES", metadata=str(metadata), transform="smarts",
        smarts_queries=(("base", "CC"),),
    ))
    assert result["structureSummary"]["missingSmilesRows"] == 0
    assert result["rows"][0]["matchedRows"] == 1


def test_grouped_boxplot_compares_series_and_preserves_order(tmp_path):
    fredis = _write_similarity(
        tmp_path / "fredis.mssim",
        pd.DataFrame({"index1": range(4), "index2": range(4),
                      "cosine_similarity": [0.2, 0.4, 0.6, 0.8]}),
    )
    fiora = _write_similarity(
        tmp_path / "fiora.mssim",
        pd.DataFrame({"index1": range(3), "index2": range(3),
                      "cosine_similarity": [0.5, 0.7, np.nan]}),
    )
    config = {
        "title": "Tool comparison",
        "groups": [{"id": "massbank", "name": "MassBank"}, {"id": "mona", "name": "MoNA"}],
        "series": [
            {"id": "fiora", "name": "FIORA", "color": "#ff00aa"},
            {"id": "fredis", "name": "FREDIS", "color": "#00aacc"},
        ],
        "entries": [
            {"groupId": "massbank", "seriesId": "fiora", "path": str(fiora), "noData": False},
            {"groupId": "massbank", "seriesId": "fredis", "path": str(fredis), "noData": False},
            {"groupId": "mona", "seriesId": "fiora", "noData": True},
            {"groupId": "mona", "seriesId": "fredis", "path": str(fredis), "noData": False},
        ],
    }
    result = compare(config)
    assert [item["name"] for item in result["groups"]] == ["MassBank", "MoNA"]
    assert [item["name"] for item in result["series"]] == ["FIORA", "FREDIS"]
    cells = {(cell["groupId"], cell["seriesId"]): cell for cell in result["cells"]}
    assert cells[("massbank", "fiora")]["count"] == 2
    assert cells[("massbank", "fiora")]["invalidCount"] == 1
    assert cells[("mona", "fiora")]["status"] == "no-data"
    svg = grouped_box_plot_svg(result, width=900, height=500)
    assert 'width="900" height="500"' in svg
    assert svg.count("#ff00aa") >= 2
    assert "No data" in svg
    assert '>Group</text>' not in svg
    assert '>MassBank</text>' in svg
    assert '>MoNA</text>' in svg
    assert 'class="separator"' in svg
    assert '<rect width="900" height="500"' not in svg
    assert '<rect x="62.52" y="48.00" width="14" height="10" fill="#ff00aa"/>' in svg
    assert '<text x="82.52" y="58.00" class="legend">FIORA</text>' in svg
    assert '<rect x="133.02" y="48.00" width="14" height="10" fill="#00aacc"/>' in svg
    violin = grouped_box_plot_svg(result, width=900, height=500, plot_type="violin")
    assert violin.count('class="violin"') == 3
    assert 'class="whisker"' not in violin
    wider = grouped_box_plot_svg(result, width=1400, height=500)
    box_pattern = r'<rect x="[^"]+" y="[^"]+" width="([^"]+)"[^>]+fill-opacity'
    assert float(re.search(box_pattern, wider).group(1)) > float(re.search(box_pattern, svg).group(1))
    fixed = grouped_box_plot_svg(
        result, width=1400, height=500, group_gap=90, series_gap=4, box_width=28,
        graph_opacity=0.35, x_label_size=17, y_label_size=16,
        x_axis_title_size=18, y_axis_title_size=19, title_size=25,
    )
    assert re.search(box_pattern, fixed).group(1) == "28.00"
    assert '<g class="graph" opacity="0.35">' in fixed
    assert '.x-label{font-size:17px' in fixed
    assert '.y-label{font-size:16px}' in fixed
    assert '.x-axis-title{font-size:18px}' in fixed
    assert '.y-axis-title{font-size:19px}' in fixed
    assert '.title{font-size:25px' in fixed
    assert '<text x="700.00" y="31.00" text-anchor="middle" class="title">' in fixed
    large = grouped_box_plot_svg(
        result, width=400, height=300, x_label_size=96, y_label_size=96,
        x_axis_title_size=96, y_axis_title_size=96, title_size=96,
    )
    large_size = re.search(r'<svg[^>]+width="([^"]+)" height="([^"]+)"', large)
    assert float(large_size.group(1)) > 400
    assert float(large_size.group(2)) > 300
    assert 'rotate(90)' in large


def test_grouped_cli_writes_opaque_svg(tmp_path):
    similarity, _ = _fixtures(tmp_path)
    image = tmp_path / "comparison.svg"
    config = {
        "graphOpacity": 0.45, "xLabelSize": 14,
        "yLabelSize": 15, "xAxisTitleSize": 18,
        "yAxisTitleSize": 19, "titleSize": 23,
        "groups": [{"id": "mona", "name": "MoNA"}],
        "series": [{"id": "tool", "name": "Tool", "color": "#123456"}],
        "entries": [{"groupId": "mona", "seriesId": "tool", "path": str(similarity)}],
    }
    args = build_parser().parse_args([
        "compare", "--request-json", json.dumps(config), "--width", "640",
        "--height", "400", "--opaque", "--background-color", "#abcdef",
        "--output-image", str(image),
    ])
    progress = []
    result = execute(args, progress=lambda percent, message: progress.append((percent, message)))
    assert result["cells"][0]["count"] == 6
    assert progress[-1] == (100, "Grouped evaluation complete")
    assert any(message == "Rendering grouped SVG…" for _, message in progress)
    cli_svg = image.read_text()
    assert '<rect width="640" height="400" fill="#abcdef"/>' in cli_svg
    assert '<g class="graph" opacity="0.45">' in cli_svg
    assert '.x-label{font-size:14px' in cli_svg
    assert '.y-label{font-size:15px}' in cli_svg
    assert '.x-axis-title{font-size:18px}' in cli_svg
    assert '.y-axis-title{font-size:19px}' in cli_svg
    assert '.title{font-size:23px' in cli_svg
    saved_settings = tmp_path / "grouped-settings.json"
    saved_settings.write_text(json.dumps({
        "schema": "fredis.evaluation.config", "schemaVersion": 1,
        "kind": "grouped", "config": config,
    }))
    loaded = execute(build_parser().parse_args(["compare", "--config", str(saved_settings)]))
    assert loaded["cells"][0]["count"] == 6
