import csv
import json

import pytest

from fredis.ml.training.fragment_tree_training.metric_store import (
    INDEX_NAME,
    chart_key,
    migrate_legacy,
    read_scalar_events,
    scalar_rows,
    write_series_rows,
)


def catalog(directory):
    return json.loads((directory / INDEX_NAME).read_text())


def read_series(directory, name):
    entry = next(item for item in catalog(directory)["series"] if item["name"] == name)
    with (directory / entry["file"]).open() as stream:
        return [
            row
            for row in csv.DictReader(stream, delimiter="\t")
            if row["series"] == name
        ]


def test_scalar_snapshot_append_distributions_and_resume(tmp_path):
    events = [
        {"event": "epoch_end", "global_step": 0, "epoch": 0, "validation_loss": 3.0},
        {"event": "train_step", "global_step": 10, "epoch": 1, "train_loss": 2.0},
        {"event": "epoch_end", "global_step": 10, "epoch": 1, "validation_loss": 1.0},
    ]
    write_series_rows(tmp_path, scalar_rows(events), replace=True)
    write_series_rows(
        tmp_path,
        [("distributions/validation/cosine_similarity@depth:1_mean", 10, 1, "", 0.5)],
    )
    write_series_rows(
        tmp_path,
        [("distributions/validation/cosine_similarity@depth:2_mean", 10, 1, "", 0.4)],
    )
    # Publishing an intermediate result must not replace regular train/validation data.
    write_series_rows(
        tmp_path,
        scalar_rows(
            [
                {
                    "event": "intermediate_validation",
                    "global_step": 10,
                    "epoch": 1,
                    "validation_loss": 1.5,
                },
            ]
        ),
        replace=True,
    )
    write_series_rows(tmp_path, scalar_rows(events), replace=True)
    assert read_scalar_events(tmp_path) == events
    assert (
        read_series(tmp_path, "metrics/intermediate_validation_loss")[0]["value"]
        == "1.5"
    )
    depth = [item for item in catalog(tmp_path)["series"] if "@depth:" in item["name"]]
    assert len({item["file"] for item in depth}) == 1
    assert len({item["file"] for item in catalog(tmp_path)["series"]}) == 3
    assert not (tmp_path / "metrics.tsv").exists()
    assert not (tmp_path / "metric_distributions.tsv").exists()


def test_migration_preserves_sources_and_normalizes_historical_metrics(tmp_path):
    scalar = "event\tepoch\tglobal_step\ttrain_samples\ttrain_batches\ttrain_intensity_loss\ttrain_spectrum_loss_with_precursor\ttrain_spectrum_loss_without_precursor\ttrain_transition_cross_entropy\ttrain_transition_cross_entropy_without_precursor\n"
    scalar += "epoch_end\t1\t10\t100\t5\t10\t6\t4\t2\t1\n"
    distribution = "global_step\tepoch\tsplit\tmetric\tvalue\n10\t1\tvalidation\tcosine_similarity@main_adduct:[M+H]+_mean\t0.5\n"
    (tmp_path / "metrics.tsv").write_text(scalar)
    (tmp_path / "metric_distributions.tsv").write_text(distribution)
    (tmp_path / "fragment-tree.training-config.pft").write_text(
        json.dumps({"intensityLossWeight": 2})
    )
    result = migrate_legacy(tmp_path)
    assert result["files"] > 1
    assert (tmp_path / "metrics.tsv").read_text() == scalar
    assert (tmp_path / "metric_distributions.tsv").read_text() == distribution
    assert read_series(tmp_path, "metrics/train_samples")[0]["value"] == "20.0"
    assert read_series(tmp_path, "metrics/train_total_samples")[0]["value"] == "100"
    assert (
        read_series(tmp_path, "metrics/train_cross_entropy_loss")[0]["value"] == "6.0"
    )
    assert read_series(tmp_path, "metrics/train_cosine_loss")[0]["value"] == "4.0"
    assert len(catalog(tmp_path)["legacySources"]) == 2
    with pytest.raises(ValueError, match="already exists"):
        migrate_legacy(tmp_path)


def test_paths_are_safe_and_precursor_chart_partition_matches_ui(tmp_path):
    assert (
        chart_key("metrics/train_spectrum_loss_without_precursor")
        == "spectrum_loss_without_precursor"
    )
    assert (
        chart_key(
            "distributions/validation/peak_precision_top5_without_precursor@depth:1_mean"
        )
        == "peak_precision_top05@depth"
    )
    names = [
        "distributions/validation/cosine_similarity@main_adduct:../../strange_mean",
        "distributions/validation/cosine_similarity_without_precursor@main_adduct:[M+H]+_q90",
    ]
    write_series_rows(tmp_path, [(name, 1, 1, "", 0.5) for name in names])
    entries = catalog(tmp_path)["series"]
    assert len({item["file"] for item in entries}) == 1
    assert all(
        (tmp_path / item["file"]).resolve().is_relative_to(tmp_path.resolve())
        for item in entries
    )


def test_tuning_reads_partitioned_validation_history(tmp_path):
    from fredis.ml.tuning.fragment_tree import validation_history

    (tmp_path / "metrics.tsv").write_text(
        "event\tepoch\tglobal_step\tvalidation_loss\n"
        "epoch_end\t1\t10\t2\n"
        "epoch_end\t2\t20\t1\n"
    )
    (tmp_path / "metric_distributions.tsv").write_text(
        "global_step\tepoch\tsplit\tmetric\tvalue\n"
        "10\t1\tvalidation\tcosine_similarity_mean\t0.4\n"
        "20\t2\tvalidation\tcosine_similarity_mean\t0.8\n"
        "15\t1\tintermediate_validation\tcosine_similarity_mean\t0.99\n"
    )
    expected = validation_history(tmp_path)
    migrate_legacy(tmp_path)
    assert validation_history(tmp_path) == expected


def test_appending_after_interruption_keeps_complete_rows(tmp_path):
    name = "distributions/validation/loss_mean"
    write_series_rows(tmp_path, [(name, 1, 1, "", 3.0)])
    entry = catalog(tmp_path)["series"][0]
    with (tmp_path / entry["file"]).open("a") as stream:
        stream.write(name + "\t2\t2\t\t1.")
    write_series_rows(tmp_path, [(name, 3, 3, "", 1.0)])
    assert [row["global_step"] for row in read_series(tmp_path, name)] == ["1", "3"]
