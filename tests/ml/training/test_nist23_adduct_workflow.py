"""Exercise the shipped NIST23 presets through preparation, training and inference.

Synthetic, assignable spectra test execution and gradients, not predictive accuracy.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import msentity as ms
import numpy as np
import pandas as pd
import pytest
import torch

from fredis.analysis.training_eligibility.native import extract
from fredis.domain.fragment import Fragmenter
from fredis.libs.mmkit.mmkit import Adduct, Compound
from fredis.ml.input.source_reaction_structure import SourceReactionStructure
from fredis.ml.specgen.predict_spectrum import load_generator
from fredis.ml.specgen.competitive_spectrum_model import merge_candidate_peaks
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.training.fragment_tree_training.model import ReactionFragmentTreeTrainingModel
from fredis.msdata import dataset_from_arrays, with_metadata
from tests.ml.reaction.TestReactionPipeline import config


ROOT = Path(__file__).resolve().parents[3]
CASES = {
    "plus": [("[M+H]+", "CCCO"), ("[M+H-H2O]+", "CCCO"),
             ("[M+H-2H2O]+", "OCCO"), ("[M+H-NH3]+", "CCCN"),
             ("[M+Na]+", "CCCO"), ("[2M+H]+", "CCCO"), ("[M+NH4]+", "CCCO")],
    "minus": [("[M-H]-", "CCCO"), ("[M-H-H2O]-", "CCCO"),
              ("[M-H-CO2]-", "CCC(=O)O"), ("[2M-H]-", "CCCO"),
              ("[M+Cl]-", "CCCO")],
}


def run_cli(directory, name, *args):
    result = subprocess.run(
        [sys.executable, *map(str, args)], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=180,
    )
    (directory / f"{name}.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (name, result.stdout[-3000:], result.stderr[-5000:])


def synthetic_dataset(fragmenter, cases):
    metadata, arrays, offsets = [], [], [0]
    for i, (label, smiles) in enumerate(cases):
        source, adduct = Compound.from_smiles(smiles), Adduct.parse(label)
        tree = fragmenter.build_fragment_ion_tree(source)
        assert fragmenter.resolve_precursor_reactions(tree, adduct), label
        precursor = adduct.apply_to_formula(source.formula).exact_mass
        group = tree.get_formula_candidate_group(adduct)
        mz = sorted({f.exact_mass for f in group.formulas
                     if f.exact_mass > 0 and f.exact_mass < precursor - 0.1
                     and all(v >= 0 for v in f.elements.values())})
        matches = fragmenter.annotate_peaks(tree, adduct, mz)
        mz = [m for m, annotations in zip(mz, matches) if annotations][:8]
        assert mz, f"No assignable non-precursor peaks: {label}"
        for energy in (20.0, 40.0):
            peaks = np.column_stack((mz, np.linspace(1.0, 0.2, len(mz))))
            arrays.append(peaks)
            offsets.append(offsets[-1] + len(peaks))
            metadata.append(dict(SpecID=f"case-{i}-{energy}", SMILES=smiles,
                                 AdductType=label, PrecursorMZ=precursor, CollisionEnergy=energy))
    return with_metadata(pd.DataFrame(metadata), dataset_from_arrays(np.concatenate(arrays), np.array(offsets)))


@pytest.mark.parametrize("polarity", ["plus", "minus"])
def test_nist23_all_adducts_preparation_training_and_prediction(tmp_path, polarity):
    torch.manual_seed(71)
    cfg = config()
    preset = ROOT / f"fredis/presets/fragment/adduct_rule_sets/nist23/NIST23_{polarity}.adductset.pft"
    rules = json.loads(preset.read_text())["fragment_ion_adduct_rule_set"]
    cfg["fragmenter_params"]["fragment_ion_tree_builder"]["fragment_ion_adduct_rule_set"] = rules
    cfg["adduct_type_strs"] = [label for label, _ in CASES[polarity]]
    cfg["reaction_model_params"]["reaction_transformer"] = dict(encoder_layers=1, decoder_layers=1)
    chemistry = dict(fragmenter_params=cfg["fragmenter_params"], symbols=cfg["mol_encoder_params"]["symbols"])
    fragmenter = Fragmenter.from_dict(cfg["fragmenter_params"])
    assert set(fragmenter.adduct_types) == {Adduct.parse(s) for s in cfg["adduct_type_strs"]}
    dataset = synthetic_dataset(fragmenter, CASES[polarity])
    input_path = tmp_path / "input.msds"
    dataset.save(str(input_path))
    validation_smiles = {"CCCO": "CCCCO", "OCCO": "OCCCO", "CCCN": "CCCCN", "CCC(=O)O": "CCCC(=O)O"}
    validation = synthetic_dataset(fragmenter, [(label, validation_smiles[s]) for label, s in CASES[polarity]])
    validation_path = tmp_path / "validation.msds"
    validation.save(str(validation_path))
    stats = extract(input_path, tmp_path / "trainable", chemistry, model_config=cfg,
                    workers=1, batch_size=2, max_reaction_combinations=1000)
    assert stats["trainable"] == len(dataset) and stats["excluded"] == 0, stats
    records = pd.read_csv(tmp_path / "trainable/records.tsv", sep="\t")
    assert records.GradientsFinite.all()
    params = tmp_path / "fragmenter.pft"
    params.write_text(json.dumps(cfg["fragmenter_params"]))
    prepared = tmp_path / "prepared"
    run_cli(tmp_path, "prepare", "-m", "fredis.cli", "train", "create-fragment-tree-data",
            "--input", input_path, "--validation-input", validation_path, "--params", params,
            "--output-dir", prepared, "--num-workers", "1", "--save-all-edges",
            "--max-reaction-combinations", "1000", "--overwrite", "1")
    generator = create_spectrum_generator(cfg)
    wrapper = ReactionFragmentTreeTrainingModel(generator.feature_model, spectrum_model=generator.spectrum_model)
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=0.001)
    seen = {"state_heads": set(), "fragment_heads": set()}
    for split in ("train_structures", "validation_structures"):
        files = list((prepared / split).rglob("*.preft.pt"))
        assert files
        for file in files:
            data = SourceReactionStructure.load(file)
            data.validate()
            if split != "train_structures":
                continue
            optimizer.zero_grad()
            output = wrapper(data)
            assert torch.isfinite(output.loss)
            output.loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None)
            changed = []
            for bank, reached in seen.items():
                for main in data.adducts.main_adducts:
                    index = generator.spectrum_model.head_index[main]
                    head = getattr(generator.spectrum_model, bank)[index]
                    active = [p for p in head.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
                    if active:
                        reached.add(main)
                        changed.append((active[0], active[0].detach().clone()))
            optimizer.step()
            assert all(not torch.equal(parameter, before) for parameter, before in changed)
    expected = {str(Adduct.parse(s)) for s in cfg["adduct_type_strs"]}
    assert all(reached == expected for reached in seen.values()), seen
    architecture = tmp_path / "architecture.pft"
    architecture.write_text(json.dumps(dict(model_config=cfg)))
    trained = tmp_path / "training"
    run_cli(tmp_path, "train", "-m", "fredis.cli", "train", "fragment-tree-detailed",
            "--dataset-dir", prepared, "--output-dir", trained, "--model-architecture", architecture,
            "--epochs", "1", "--batch-size", "3", "--device", "cpu", "--lr", "0.002",
            "--warmup-steps", "0", "--minimum-assignment-score", "0",
            "--minimum-assignment-score-without-precursor", "0", "--validate-before-training", "--overwrite")
    checkpoint = trained / "last.pt"
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["global_step"] > 0 and saved["optimizer_state_dict"]["state"]
    assert all(torch.isfinite(v).all() for v in saved["model_state_dict"].values() if v.is_floating_point())
    restored = load_generator(model_path=str(checkpoint), params_path=None, device=torch.device("cpu"), strict=True)
    sources = [Compound.from_smiles(s) for _, s in CASES[polarity]]
    adducts = [Adduct.parse(s) for s, _ in CASES[polarity]]
    chunks = list(restored.predict_batches(sources, adducts, [20.0] * len(sources), max_samples=3))
    found = {}
    for prediction in chunks:
        spectra = merge_candidate_peaks(prediction.spectra)
        for local, original in enumerate(prediction.sample_input_index):
            mask = spectra.sample_index == local
            found[original] = (spectra.formula_tensor[mask], spectra.intensity[mask])
    assert set(found) == set(range(len(sources)))
    for i, (source, adduct) in enumerate(zip(sources, adducts)):
        single = merge_candidate_peaks(restored.predict([source], [adduct], [20.0]).spectra)
        assert single.intensity.numel() and torch.isfinite(single.intensity).all()
        torch.testing.assert_close(found[i][0], single.formula_tensor)
        torch.testing.assert_close(found[i][1], single.intensity, rtol=1e-4, atol=1e-6)
    run_cli(tmp_path, "batch_prediction", "-m", "fredis.ml.specgen.predict_spectrum",
            "--model", checkpoint, "--input", input_path, "--output-dir", tmp_path / "prediction",
            "--device", "cpu", "--strict", "--max-samples", "3", "--num-workers", "2", "--chunk-size", "1",
            "--peak-intensity-threshold", "0")
    predicted = ms.load(str(tmp_path / "prediction/predicted.msds"))
    assert len(predicted) == len(dataset)
    assert predicted.to_pandas().SourceSpecID.tolist() == dataset.to_pandas().SpecID.tolist()
    for peaks in predicted:
        assert len(peaks.mz) and np.isfinite(peaks.mz).all() and np.isfinite(peaks.intensity).all()
    (tmp_path / "report.json").write_text(json.dumps(dict(
        polarity=polarity, adducts=cfg["adduct_type_strs"],
        trainable_records=stats["trainable"], validation_records=len(validation),
        nonzero_gradient_and_parameter_update={k: sorted(v) for k, v in seen.items()},
        training_steps=saved["global_step"], batch_prediction_records=len(predicted),
        single_batch_predictions_match=True, device="cpu",
        scope="Synthetic execution and numerical checks; not experimental accuracy",
    ), indent=2))
