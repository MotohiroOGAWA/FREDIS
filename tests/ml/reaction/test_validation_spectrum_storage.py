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


def test_msds_roundtrip_keeps_paired_peaks_scores_smiles_and_selection(tmp_path):
    import json
    from fredis.data import MSDataset
    from fredis.ml.training.fragment_tree_training.validation_storage import (
        save_validation_msdataset,
    )

    records = [
        {
            "smiles": "CCO",
            "piece_index": i,
            "sample_index": 0,
            "adduct": "[M+H]+",
            "main_adduct": "[M+H]+",
            "collision_energy": 20.0,
            "precursor_mz": 47.0,
            "assignment_score": 0.2 + i * 0.2,
            "assignment_score_without_precursor": 0.1 + i * 0.2,
            "cosine_similarity": 0.8 - i * 0.2,
            "cosine_similarity_without_precursor": 0.7 - i * 0.2,
            "generated_peaks": [
                {"mz": 47.0, "intensity": 1.0, "precursor": True, "depth": 0}
            ],
            "original_peaks": [
                {"mz": 20.0, "intensity": 0.5, "precursor": False, "depth": None}
            ],
        }
        for i in range(3)
    ]
    representatives, saved = _compact_spectra(records)
    target = save_validation_msdataset(
        saved, representatives, tmp_path / "epoch_2.msds", epoch=2, global_step=10
    )
    dataset = MSDataset.load(str(target))
    assert len(dataset) == 2 * len(saved)
    meta = dataset.metadata
    assert set(meta["SpectrumRole"]) == {"generated", "observed"}
    assert set(meta["SMILES"]) == {"CCO"}
    assert set(meta["Epoch"]) == {2}
    for index, record in enumerate(saved):
        pair = meta[meta["RepresentativeIndex"] == index]
        assert len(pair) == 2
        for field in (
            "assignment_score",
            "assignment_score_without_precursor",
            "cosine_similarity",
            "cosine_similarity_without_precursor",
        ):
            assert pair[field].tolist() == [record[field], record[field]]
        assert json.loads(pair.iloc[0]["SelectedBy"])
    assert not (tmp_path / ".epoch_2.msds.tmp").exists()
