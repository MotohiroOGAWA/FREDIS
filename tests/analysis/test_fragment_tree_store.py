"""A shared fragment-tree store connects Reaction Pattern Analysis and Training Data.

A tree saved by either one (every reaction history and its product) is reused
by the other when the compound, the Fragmenter settings, the precursor
adducts and the patterns match; a tree of a subset of the patterns (e.g. the
non-ring single-bond cleavages computed first) seeds a build that only adds
the remaining patterns.  Results must equal a build without the store.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fredis.analysis.prepared_result import load_prepared_structure
from fredis.analysis.reaction_pattern import ReactionPatternAnalysisResult, generate_reaction_pattern_analysis
from fredis.analysis.reaction_pattern.compounds import compound_id_for
from fredis.analysis.reaction_pattern.constants import assigned_column, count_column
from fredis.domain.fragment.tree_store import FragmentTreeStore
from fredis.ml.data_preparation.fragment_tree.create_training_data import create_reaction_training_data
from fredis.msdata import dataset_from_arrays, with_metadata
from tests.ml.reaction.TestReactionPipeline import chemistry

PATTERN_SET = Path("fredis/presets/fragment/reaction_pattern_sets/nonring_single_bond/nonring_single_bond.reactionset.pft")
NAMES = {"C": "nonring_c_single_bond", "O": "nonring_o_single_bond"}
# (SMILES, adduct, peaks); the [M+H-H2O]+ spectra prune their trees.
SPECTRA = [
    ("CCCO", "[M+H]+", [[61.065, 1.0], [43.054, 0.7], [31.018, 0.4], [29.039, 0.3]]),
    ("CCCO", "[M+H-H2O]+", [[43.054, 1.0], [41.039, 0.5], [29.039, 0.4], [27.023, 0.2]]),
    ("OCC(O)CO", "[M+H-H2O]+", [[75.044, 1.0], [57.034, 0.6], [45.034, 0.3], [31.018, 0.2]]),
    ("CCO", "[M+H]+", [[47.049, 1.0], [31.018, 0.6], [29.039, 0.3]]),
]


def _patterns(*keys: str) -> list[dict]:
    rows = {row["name"]: row for row in json.loads(PATTERN_SET.read_text())["reaction_pattern_set"]["patterns"]}
    return [rows[NAMES[key]] for key in keys]


def _config(*keys: str) -> dict:
    value = chemistry()
    builder = value["fragmenter_params"]["fragment_ion_tree_builder"]
    builder["max_reaction_count"] = 2
    builder["reaction_pattern_set"] = {"name": "".join(keys), "patterns": _patterns(*keys)}
    return value


def _dataset():
    metadata = pd.DataFrame({
        "SMILES": [row[0] for row in SPECTRA],
        "AdductType": [row[1] for row in SPECTRA],
        "CollisionEnergy": [20.0] * len(SPECTRA),
        "PrecursorMZ": [row[2][0][0] for row in SPECTRA],
    })
    peaks = [peak for _, _, values in SPECTRA for peak in values]
    offsets = np.cumsum([0] + [len(values) for _, _, values in SPECTRA])
    return with_metadata(metadata, dataset_from_arrays(np.asarray(peaks, dtype=float), np.asarray(offsets, dtype=np.int64)))


def _training(output: Path, keys, **store) -> dict:
    files = create_reaction_training_data(dataset=_dataset(), chemistry_config=_config(*keys), output_dir=output, **store)
    return {_tree_identity(path)[0]: path for path in files}


def _tree_identity(path: Path):
    """SMILES, every kept history (by reaction effect) and every peak's explanation count."""
    structure, metadata = load_prepared_structure(path)
    histories = frozenset(
        frozenset((r.reaction_pattern_id, r.reaction_rule_id, r.source_atom_maps, r.retained_atom_maps)
                  for r in sequence.reactions)
        for sequence, _ in metadata["reaction_sequence_compounds"])
    peaks = tuple(tuple(len(peak.get("matches", ())) for peak in annotation.get("peaks", ()))
                  for annotation in structure.sample_annotations)
    return metadata["smiles"], histories, peaks


def _same_training(left: dict, right: dict) -> None:
    assert sorted(left) == sorted(right)
    for smiles in left:
        assert _tree_identity(left[smiles]) == _tree_identity(right[smiles]), smiles


def _manifest(output: Path) -> pd.DataFrame:
    return pd.read_csv(output / "manifest.tsv", sep="\t")


def test_training_data_saves_and_replays_trees(tmp_path):
    store = tmp_path / "store"
    first = _training(tmp_path / "first", "CO", tree_store=store, save_trees=True)
    assert FragmentTreeStore(store).count() == 3  # one tree per compound
    again = _training(tmp_path / "again", "CO", tree_store=store, save_trees=False)
    _same_training(again, first)
    # Replayed trees run no RDKit reaction.
    assert (_manifest(tmp_path / "again")["num_rdkit_run_reactants"] == 0).all()
    assert (_manifest(tmp_path / "first")["num_rdkit_run_reactants"] > 0).any()


def test_a_tree_of_some_patterns_seeds_training_data_of_more_patterns(tmp_path):
    """Non-ring C–C cleavages first; training data with C–C and C–O cleavages reuses them."""
    store = tmp_path / "store"
    _training(tmp_path / "base", "C", tree_store=store, save_trees=True)
    reused = _training(tmp_path / "reused", "CO", tree_store=store)
    fresh = _training(tmp_path / "fresh", "CO")
    _same_training(reused, fresh)
    rdkit = lambda name: int(_manifest(tmp_path / name)["num_rdkit_run_reactants"].sum())
    assert rdkit("reused") < rdkit("fresh")


# -- Reaction Pattern Analysis ------------------------------------------------------------
def _set(key: str, *keys: str) -> dict:
    return {"id": key, "name": key, "patterns": _patterns(*(keys or (key,)))}


def _rpa(output: Path, tree: dict, dataset: Path, **store) -> Path:
    return generate_reaction_pattern_analysis(sources=[{"path": str(dataset)}], output_dir=output,
                                              chemistry_config=_config("C"), tree=tree, overwrite=True, **store)


@pytest.fixture
def dataset_file(tmp_path) -> Path:
    path = tmp_path / "spectra.msds"
    _dataset().save(str(path))
    return path


def _peaks(result: Path, node_id: str) -> dict:
    with ReactionPatternAnalysisResult(result) as reader:
        condition = {row["node_id"]: row for row in reader.nodes()}[node_id]["condition_id"]
    values = {}
    for smiles in sorted({row[0] for row in SPECTRA}):
        frame = pd.read_parquet(result.parent / "compounds" / f"{compound_id_for(_canonical(smiles))}.peaks.parquet")
        values[smiles] = (frame[assigned_column(condition)].tolist(), frame[count_column(condition)].tolist())
    return values


def _canonical(smiles: str) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


ROOT_CO = {"patternSets": [_set("CO", "C", "O")], "nodes": [{"id": "r", "patternSet": "CO"}]}


def test_reaction_pattern_analysis_reuses_trees_from_training_data(tmp_path, dataset_file):
    store = tmp_path / "store"
    _training(tmp_path / "training", "CO", tree_store=store, save_trees=True)
    reused = _rpa(tmp_path / "reused", ROOT_CO, dataset_file, tree_store=store)
    fresh = _rpa(tmp_path / "fresh", ROOT_CO, dataset_file)
    assert _peaks(reused, "r") == _peaks(fresh, "r")
    with ReactionPatternAnalysisResult(reused) as reader:
        overall = reader.overall_summary()
    assert overall["treeStoreReusedTrees"] == 3 and overall["treeStoreSavedTrees"] == 0


def test_training_data_reuses_a_non_ring_tree_from_reaction_pattern_analysis(tmp_path, dataset_file):
    """RPA computes the C–C cleavage root once and saves it; training data with
    C–C and C–O cleavages adds only the C–O reactions."""
    store = tmp_path / "store"
    tree = {"patternSets": [_set("C"), _set("O")],
            "nodes": [{"id": "c", "patternSet": "C"}, {"id": "co", "patternSet": "O", "parent": "c"}]}
    _rpa(tmp_path / "rpa", tree, dataset_file, tree_store=store, save_trees=True)
    assert FragmentTreeStore(store).count() >= 3
    reused = _training(tmp_path / "reused", "CO", tree_store=store)
    fresh = _training(tmp_path / "fresh", "CO")
    _same_training(reused, fresh)
    # An RPA run of the C + O condition as a root reuses the saved C → C+O edge trees.
    again = _rpa(tmp_path / "again", ROOT_CO, dataset_file, tree_store=store)
    assert _peaks(again, "r") == _peaks(_rpa(tmp_path / "fresh_rpa", ROOT_CO, dataset_file), "r")


def test_trees_are_not_reused_across_different_settings(tmp_path):
    store = tmp_path / "store"
    _training(tmp_path / "base", "CO", tree_store=store, save_trees=True)
    config = _config("C", "O")
    config["fragmenter_params"]["fragment_ion_tree_builder"]["max_reaction_count"] = 1
    create_reaction_training_data(dataset=_dataset(), chemistry_config=config, output_dir=tmp_path / "other",
                                  tree_store=store)
    assert (_manifest(tmp_path / "other")["num_rdkit_run_reactants"] > 0).any()


def test_parallel_workers_share_one_store(tmp_path, dataset_file):
    store = tmp_path / "store"
    parallel = _training(tmp_path / "parallel", "CO", tree_store=store, save_trees=True, num_workers=2, chunk_size=1)
    assert FragmentTreeStore(store).count() == 3
    _same_training(parallel, _training(tmp_path / "serial", "CO"))
    rpa = generate_reaction_pattern_analysis(sources=[{"path": str(dataset_file)}], output_dir=tmp_path / "rpa",
                                             chemistry_config=_config("C"), tree=ROOT_CO, overwrite=True,
                                             tree_store=store, num_workers=2)
    with ReactionPatternAnalysisResult(rpa) as reader:
        assert reader.overall_summary()["treeStoreReusedTrees"] == 3


def test_multi_dataset_training_data_is_a_reaction_pattern_analysis_source(tmp_path):
    """Training data prepared from several datasets (each with its own columns)
    is an RPA source: its spectra come from the original datasets' rows and its
    stored trees are replayed."""
    import contextlib
    import io

    from fredis.ml.data_preparation.fragment_tree.create_training_data import main

    frames = []
    for index, rows in enumerate((SPECTRA[:2], SPECTRA[2:])):
        rows = list(rows)
        metadata = pd.DataFrame({
            "Structure": [row[0] for row in rows], "Precursor": [row[1] for row in rows],
            "Energy": [20.0] * len(rows), "Mz": [row[2][0][0] for row in rows]})
        peaks = [peak for _, _, values in rows for peak in values]
        offsets = np.cumsum([0] + [len(values) for _, _, values in rows])
        path = tmp_path / f"input-{index}.msds"
        with_metadata(metadata, dataset_from_arrays(np.asarray(peaks, dtype=float),
                                                    np.asarray(offsets, dtype=np.int64))).save(str(path))
        frames.append(dict(id=f"dataset-{index}", path=str(path), split="train",
                           mapping=dict(smilesColumn="Structure", adductTypeColumn="Precursor",
                                        collisionEnergyColumn="Energy", precursorMzColumn="Mz")))
    output, store = tmp_path / "prepared", tmp_path / "store"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        main(["--datasets-json", json.dumps(frames), "--dataset-mode", "assigned",
              "--params-json", json.dumps(_config("C", "O")), "--output-dir", str(output),
              "--num-workers", "1", "--tree-store", str(store), "--save-trees", "1"])
    assert FragmentTreeStore(store).count() == 3
    result = generate_reaction_pattern_analysis(
        sources=[{"path": str(output / "train_structures")}], output_dir=tmp_path / "rpa",
        chemistry_config=_config("C"), tree=ROOT_CO, overwrite=True, tree_store=store)
    with ReactionPatternAnalysisResult(result) as reader:
        overall = reader.overall_summary()
    assert overall["treeStoreReusedTrees"] == 3
    fresh = _rpa(tmp_path / "fresh", ROOT_CO, _dataset_file(tmp_path))
    assert _peaks(result, "r") == _peaks(fresh, "r")


def _dataset_file(tmp_path) -> Path:
    path = tmp_path / "all.msds"
    _dataset().save(str(path))
    return path
