import numpy as np

from fredis.ml.training.fragment_tree_training.spectrum_validation import (
    QUANTILES,
    _compact_spectra,
)


def test_storage_keeps_only_cosine_quantile_examples_with_valid_indices():
    records = [
        {"cosine_similarity": i / 100, "generated_peaks": [{"mz": i}]}
        for i in range(101)
    ]
    representatives, spectra = _compact_spectra(records)
    assert len(records) == 101
    assert len(spectra) == 5
    assert set(representatives) == {"cosine_similarity"}
    for name, quantile in QUANTILES:
        item = representatives["cosine_similarity"][name]
        expected = int(quantile * 100)
        assert item["target"] == np.quantile(
            [r["cosine_similarity"] for r in records], quantile
        )
        assert item["source_spectrum_index"] == expected
        assert spectra[item["spectrum_index"]] is records[expected]


def test_storage_deduplicates_examples_when_quantiles_choose_same_spectrum():
    representatives, spectra = _compact_spectra([{"cosine_similarity": 0.5}] * 20)
    assert len(spectra) == 1
    assert all(
        item["spectrum_index"] == 0
        for item in representatives["cosine_similarity"].values()
    )


def test_each_metric_selects_from_its_own_full_validation_distribution():
    metrics = (
        "cosine_similarity",
        "cosine_similarity_without_precursor",
        "assignment_score",
        "assignment_score_without_precursor",
    )
    rng = np.random.default_rng(42)
    values = {metric: rng.random(101) for metric in metrics}
    records = [
        {metric: float(values[metric][i]) for metric in metrics} for i in range(101)
    ]
    representatives, spectra = _compact_spectra(records)
    assert set(representatives) == set(metrics)
    selected = set()
    for metric in metrics:
        for name, quantile in QUANTILES:
            target = float(np.quantile(values[metric], quantile))
            expected = int(np.argmin(abs(values[metric] - target)))
            item = representatives[metric][name]
            assert item["target"] == target
            assert item["source_spectrum_index"] == expected
            assert spectra[item["spectrum_index"]] is records[expected]
            selected.add(expected)
    assert len(spectra) == len(selected) <= 20
    assert len(selected) > 5
