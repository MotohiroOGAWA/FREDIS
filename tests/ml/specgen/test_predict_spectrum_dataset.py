from __future__ import annotations
from fredis.msdata import with_metadata

import pandas as pd
import pytest
import torch

import msentity as ms
from msentity_similarity import load as load_similarity
from fredis.ml.specgen import predict_spectrum
from fredis.ml.specgen.predict_spectrum import (
    direct_input_dataset,
    prediction_metadata,
    validate_prediction_input,
    validate_prediction_output,
)
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from tests.ml.reaction.TestReactionPipeline import config


def _dataset():
    dataset = direct_input_dataset(
        smiles_values=["CCO", "CCN"],
        collision_energy="20 eV",
        adduct_type="[M+H]+",
    )
    dataset["SpecID"] = ["source-1", "source-2"]
    dataset["DB"] = ["MoNA-source", "MoNA-source"]
    dataset["ExactMass"] = [46.0419, 45.0578]
    return dataset


def test_prediction_metadata_preserves_source_values_and_adds_provenance(tmp_path) -> None:
    dataset = _dataset()
    validate_prediction_input(
        dataset,
        smiles_column="SMILES",
        precursor_mz_column="PrecursorMZ",
        adduct_type_column="AdductType",
        collision_energy_column="CollisionEnergy",
        spec_id_column="SpecID",
        instrument_column=None,
    )
    original_precursor = dataset.to_pandas()["PrecursorMZ"].copy()
    original_exact_mass = dataset.to_pandas()["ExactMass"].copy()
    result = prediction_metadata(
        dataset.to_pandas(),
        db="MoNA",
        model_path="/models/fredis-model.pt",
        spec_id_column="SpecID",
        smiles_column="SMILES",
        adduct_type_column="AdductType",
        collision_energy_column="CollisionEnergy",
        precursor_mz_column="PrecursorMZ",
        instrument_column=None,
        timestamp="2026-01-02T03:04:05+00:00",
    )

    assert result["SpecID"].tolist() == ["FREDIS_source-1", "FREDIS_source-2"]
    assert result["SourceSpecID"].tolist() == ["source-1", "source-2"]
    assert result["DB"].tolist() == ["MoNA", "MoNA"]
    assert result["SourceDB"].tolist() == ["MoNA-source", "MoNA-source"]
    assert result["PredictionTool"].tolist() == ["FREDIS", "FREDIS"]
    assert result["PredictionModel"].tolist() == ["fredis-model.pt", "fredis-model.pt"]
    assert result["PredictionCollisionEnergy"].tolist() == [20.0, 20.0]
    assert result["PredictionCollisionEnergyUnit"].tolist() == ["eV", "eV"]
    assert result["PredictionCollisionEnergyEV"].tolist() == [20.0, 20.0]
    pd.testing.assert_series_equal(result["PrecursorMZ"], original_precursor, check_names=False)
    pd.testing.assert_series_equal(result["ExactMass"], original_exact_mass, check_names=False)
    assert result["PredictedPrecursorMZ"].notna().all()

    output = with_metadata(
        metadata=result,
        dataset=dataset.copy(),
        description="Predicted spectra",
        attributes={"prediction_tool": "FREDIS"},
        tags=["predicted", "FREDIS"],
    )
    validate_prediction_output(output)
    output_path = tmp_path / "predicted.msds"
    output.save(str(output_path))
    restored = ms.load(str(output_path))
    assert restored.to_pandas()["SourceSpecID"].tolist() == ["source-1", "source-2"]
    assert restored.to_pandas()["PredictionCollisionEnergyUnit"].tolist() == ["eV", "eV"]
    assert restored.attributes["prediction_tool"] == "FREDIS"
    assert restored.tags == ["predicted", "FREDIS"]


@pytest.mark.parametrize("embedded", [True, False])
def test_main_writes_msds_and_mssim_without_an_output_name_argument(tmp_path, monkeypatch, embedded) -> None:
    # No --output-name/--output: predicted.msds and predicted.mssim are fixed
    # names under --output-dir, and the .mssim is the full/embedded variant
    # (it carries its own matched peak data), per the msentity spec.
    model_config = config()
    generator = create_spectrum_generator(model_config).eval()
    model_path = tmp_path / "model.pt"
    torch.save({"model_config": model_config, "model_state_dict": generator.state_dict()}, model_path)

    dataset = direct_input_dataset(smiles_values=["CCCO", "CCCO"], collision_energy="20 eV", adduct_type="[M+H]+")
    dataset["SpecID"] = ["orig-1", "orig-2"]
    input_path = tmp_path / "input.msds"
    dataset.save(str(input_path))

    output_dir = tmp_path / "prediction-output"
    monkeypatch.setattr("sys.argv", [
        "predict_spectrum.py", "--input", str(input_path), "--output-dir", str(output_dir),
        "--model", str(model_path), "--device", "cpu", "--spec-id-column", "SpecID",
        "--similarity-mz-tolerance", "20ppm, 0.02da",
        "--intensity-exponent", "0.7",
        "--similarity-methods", "cosine,reverse_cosine,modified_cosine,bonanza",
        "--similarity-threads", "2",
        "--similarity-precursor-column", "PrecursorMZ",
        *([] if embedded else ["--similarity-no-matched-data"]),
    ])
    predict_spectrum.main()

    msds_path = output_dir / "predicted.msds"
    mssim_path = output_dir / "predicted.mssim"
    assert msds_path.exists()
    assert mssim_path.exists()

    predicted = ms.load(str(msds_path))
    assert predicted.to_pandas()["SpecID"].tolist() == ["FREDIS_orig-1", "FREDIS_orig-2"]
    assert predicted.to_pandas()["SourceSpecID"].tolist() == ["orig-1", "orig-2"]

    similarity = load_similarity(str(mssim_path))
    assert similarity.has_matched_data is embedded
    parameters = similarity.metadata["parameters"]
    assert parameters["intensity_exponent"] == 0.7
    assert parameters["methods"] == ["cosine", "reverse_cosine", "modified_cosine", "bonanza"]
    table = similarity.to_pandas()
    assert {"cosine", "reverse_cosine", "modified_cosine", "bonanza",
            "intensity_coverage1", "intensity_coverage2",
            "modified_intensity_coverage1", "modified_intensity_coverage2"} <= set(table.columns)
    assert table["modified_cosine"].notna().all()
    assert (parameters["mz_tolerance_da"], parameters["mz_tolerance_ppm"]) == (0.02, 20.0)

    # Mirrors training/data-prep's own .pft convention, so the
    # Workbench's "Load From Run" can restore this run's settings.
    import json
    pft = json.loads((output_dir / "config.batch-prediction.pft").read_text())
    assert pft["application"] == "spectrum-prediction"
    assert pft["input"] == str(input_path.resolve())
    assert pft["modelPath"] == str(model_path.resolve())
    assert pft["numWorkers"] == 1
    assert pft["chunkSize"] == 1
    assert pft["keepTemp"] is False
    assert pft["similarityMzTolerance"] == "0.02Da,20ppm"
    assert pft["intensityExponent"] == 0.7
    assert pft["similarityMethods"] == parameters["methods"]
    assert pft["similarityThreads"] == 2
    assert pft["similarityPrecursorColumn"] == "PrecursorMZ"
    assert pft["similarityIncludeMatchedData"] is embedded


def test_main_keeps_prepare_temp_files_under_output_dir_only_with_keep_temp(tmp_path, monkeypatch) -> None:
    # The parallel-prepare temp directory belongs under --output-dir (easy to
    # find while a run is in progress, and afterward with --keep-temp), not
    # the system temp directory; by default it leaves no trace once the run
    # finishes.
    model_config = config()
    generator = create_spectrum_generator(model_config).eval()
    model_path = tmp_path / "model.pt"
    torch.save({"model_config": model_config, "model_state_dict": generator.state_dict()}, model_path)

    dataset = direct_input_dataset(smiles_values=["CCCO", "CCN"], collision_energy="20 eV", adduct_type="[M+H]+")
    dataset["SpecID"] = ["orig-1", "orig-2"]
    input_path = tmp_path / "input.msds"
    dataset.save(str(input_path))

    default_output_dir = tmp_path / "default-output"
    monkeypatch.setattr("sys.argv", [
        "predict_spectrum.py", "--input", str(input_path), "--output-dir", str(default_output_dir),
        "--model", str(model_path), "--device", "cpu", "--spec-id-column", "SpecID",
        "--num-workers", "2",
    ])
    predict_spectrum.main()
    assert not (default_output_dir / "tmp").exists()

    kept_output_dir = tmp_path / "kept-output"
    monkeypatch.setattr("sys.argv", [
        "predict_spectrum.py", "--input", str(input_path), "--output-dir", str(kept_output_dir),
        "--model", str(model_path), "--device", "cpu", "--spec-id-column", "SpecID",
        "--num-workers", "2", "--keep-temp",
    ])
    predict_spectrum.main()
    kept_temp_dir = kept_output_dir / "tmp"
    assert kept_temp_dir.exists()
    assert any(kept_temp_dir.rglob("*.pkl"))


def test_main_writes_pft_json_before_prediction_even_if_it_later_fails(tmp_path, monkeypatch) -> None:
    # Settings should be visible/loadable (Workbench "Load From Run") even if
    # the run fails or is interrupted partway through, not only on success.
    model_config = config()
    generator = create_spectrum_generator(model_config).eval()
    model_path = tmp_path / "model.pt"
    torch.save({"model_config": model_config, "model_state_dict": generator.state_dict()}, model_path)

    dataset = direct_input_dataset(smiles_values=["CCCO"], collision_energy="20 eV", adduct_type="[M+H]+")
    dataset["SpecID"] = ["orig-1"]
    input_path = tmp_path / "input.msds"
    dataset.save(str(input_path))

    output_dir = tmp_path / "prediction-output"
    monkeypatch.setattr("sys.argv", [
        "predict_spectrum.py", "--input", str(input_path), "--output-dir", str(output_dir),
        "--model", str(model_path), "--device", "cpu", "--spec-id-column", "SpecID",
    ])

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated mid-run failure")
    monkeypatch.setattr(predict_spectrum, "predict_msdataset", _boom)

    with pytest.raises(RuntimeError, match="simulated mid-run failure"):
        predict_spectrum.main()

    import json
    pft = json.loads((output_dir / "config.batch-prediction.pft").read_text())
    assert pft["application"] == "spectrum-prediction"
    assert not (output_dir / "predicted.msds").exists()


def test_main_saves_successes_and_failure_report_for_unsupported_atoms(tmp_path, monkeypatch) -> None:
    model_config = config()
    generator = create_spectrum_generator(model_config).eval()
    model_path = tmp_path / "model.pt"
    torch.save({"model_config": model_config, "model_state_dict": generator.state_dict()}, model_path)

    dataset = direct_input_dataset(
        smiles_values=["CCCO", "CCN"], collision_energy="20 eV", adduct_type="[M+H]+"
    )
    dataset["SMILES"] = ["CCCO", "O=C(O)O[Na]"]
    dataset["SpecID"] = ["valid", "unsupported-sodium"]
    input_path = tmp_path / "input.msds"
    dataset.save(str(input_path))

    output_dir = tmp_path / "prediction-output"
    monkeypatch.setattr("sys.argv", [
        "predict_spectrum.py", "--input", str(input_path), "--output-dir", str(output_dir),
        "--model", str(model_path), "--device", "cpu", "--spec-id-column", "SpecID",
    ])

    with pytest.raises(SystemExit) as exit_info:
        predict_spectrum.main()
    assert exit_info.value.code == 3

    predicted = ms.load(str(output_dir / "predicted.msds"))
    assert predicted.to_pandas()["SourceSpecID"].tolist() == ["valid"]
    failures = pd.read_csv(output_dir / "run" / "failures.tsv", sep="\t")
    assert failures["index"].tolist() == [1]
    assert "Unsupported atom symbol(s): Na" in failures.iloc[0]["error"]

    import json
    run_info = json.loads((output_dir / "run" / "run.json").read_text())
    assert run_info["success_count"] == 1
    assert run_info["failure_count"] == 1
    assert (output_dir / "predicted.mssim").exists()


def test_prediction_input_rejects_duplicate_source_ids() -> None:
    dataset = _dataset()
    dataset["SpecID"] = ["duplicate", "duplicate"]
    with pytest.raises(ValueError, match="must be unique"):
        validate_prediction_input(
            dataset,
            smiles_column="SMILES",
            precursor_mz_column="PrecursorMZ",
            adduct_type_column="AdductType",
            collision_energy_column="CollisionEnergy",
            spec_id_column="SpecID",
            instrument_column=None,
        )


def test_load_gpu_tagged_checkpoint_on_cpu(tmp_path, monkeypatch):
    model_config = config()
    generator = create_spectrum_generator(model_config).eval()
    from fredis.ml.training.fragment_tree_training.model import ReactionFragmentTreeTrainingModel
    training_model = ReactionFragmentTreeTrainingModel(
        generator.feature_model, spectrum_model=generator.spectrum_model
    )
    checkpoint = tmp_path / "gpu-trained.pt"
    # Save genuine CUDA storage tags without requiring a GPU on the test host.
    with monkeypatch.context() as patch:
        patch.setattr(torch.serialization, "_package_registry", list(torch.serialization._package_registry))
        torch.serialization.register_package(0, lambda storage: "cuda:0", lambda storage, location: None)
        torch.save({"model_config": model_config, "model_state_dict": training_model.state_dict()}, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device"):
        torch.load(checkpoint, weights_only=False)
    restored = predict_spectrum.load_generator(model_path=str(checkpoint), params_path=None,
                                              device=torch.device("cpu"), strict=True)
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())
    for key, tensor in generator.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor)
    with pytest.raises(ValueError, match="--device cpu"):
        predict_spectrum.load_generator(model_path=str(checkpoint), params_path=None,
                                        device=torch.device("cuda"), strict=True)


def test_similarity_intensity_exponent_default_and_validation(monkeypatch):
    argv = ['predict_spectrum', '--input', 'input.msds', '--model', 'model.pt',
            '--output-dir', 'out']
    monkeypatch.setattr('sys.argv', argv)
    assert predict_spectrum.parse_args().intensity_exponent == 0.5
    for invalid in ['0', '-1', 'nan', 'inf']:
        monkeypatch.setattr('sys.argv', [*argv, '--intensity-exponent', invalid])
        with pytest.raises(SystemExit):
            predict_spectrum.parse_args()


@pytest.mark.parametrize("flags", [
    ["--similarity-methods", ""],
    ["--similarity-methods", "cosine,unknown"],
    ["--similarity-threads", "0"],
])
def test_similarity_options_reject_invalid_values(tmp_path, monkeypatch, flags):
    monkeypatch.setattr("sys.argv", ["predict", "--model", "unused.pt",
                                     "--output-dir", str(tmp_path / "out"), *flags])
    with pytest.raises(SystemExit) as error:
        predict_spectrum.parse_args()
    assert error.value.code == 2
