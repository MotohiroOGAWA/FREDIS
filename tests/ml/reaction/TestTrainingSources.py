import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from fredis.libs.mmkit.mmkit import Adduct, Compound, Formula
from fredis.ml.data_preparation.fragment_tree.context import create_preparation_context
from fredis.ml.input.reaction_batching import select_samples
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.training.fragment_tree_training.sources import (
    dataset_chemistry_config,
    dataset_sources,
    inherit_model_config,
)
from fredis.ml.training.fragment_tree_training import training
from .TestReactionPipeline import chemistry


class TestTrainingSources(unittest.TestCase):
    def test_training_progress_bar_has_fixed_layout(self):
        with patch.object(training, "tqdm") as tqdm:
            training.progress_bar(
                total=12, description="Iteration", position=1, leave=False
            )
        kwargs = tqdm.call_args.kwargs
        self.assertEqual(kwargs["ncols"], 100)
        self.assertFalse(kwargs["dynamic_ncols"])
        self.assertEqual(kwargs["position"], 1)
        self.assertIn("{bar:36}", kwargs["bar_format"])

    def test_resumed_progress_starts_at_completed_epoch(self):
        with patch.object(training, "tqdm") as tqdm:
            training.progress_bar(
                total=34, initial=4, description="Epoch", position=0, leave=True
            )
        self.assertEqual(tqdm.call_args.kwargs["initial"], 4)
        self.assertEqual(tqdm.call_args.kwargs["total"], 34)

    def test_overwrite_removes_previous_run_leftovers(self):
        # metric_distributions.tsv is appended to across a run, and
        # spectrum_validation/*.json from a longer previous run is never
        # cleaned up on its own -- --overwrite must not leave those behind.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            (output / "spectrum_validation").mkdir(parents=True)
            (output / "spectrum_validation" / "epoch_9.json").write_text("{}")
            (output / "metric_distributions.tsv").write_text("stale\trow\n")
            (output / "last.pt").write_text("stale checkpoint")
            training.confirm_output_overwrite(output, overwrite=True, resume=None)
            self.assertFalse(output.exists())

    def test_interactive_yes_also_clears_the_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            (output / "stale.json").write_text("{}")
            with (
                patch("sys.stdin.isatty", return_value=True),
                patch("builtins.input", return_value="y"),
            ):
                training.confirm_output_overwrite(output, overwrite=False, resume=None)
            self.assertFalse(output.exists())

    def test_declining_interactively_raises_without_deleting_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            (output / "stale.json").write_text("{}")
            with (
                patch("sys.stdin.isatty", return_value=True),
                patch("builtins.input", return_value="n"),
            ):
                with self.assertRaises(SystemExit):
                    training.confirm_output_overwrite(
                        output, overwrite=False, resume=None
                    )
            self.assertTrue((output / "stale.json").exists())

    def test_resume_never_clears_even_with_overwrite_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            (output / "last.pt").write_text("checkpoint")
            training.confirm_output_overwrite(output, overwrite=True, resume="last.pt")
            self.assertTrue((output / "last.pt").exists())

    def test_empty_or_missing_output_directory_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            training.confirm_output_overwrite(output, overwrite=True, resume=None)
            self.assertTrue(output.exists())  # untouched, not recreated
            missing = Path(directory) / "missing"
            training.confirm_output_overwrite(missing, overwrite=True, resume=None)
            self.assertFalse(missing.exists())

    def test_overwrite_refuses_to_delete_a_protected_input_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            train_dir = output / "train_structures"
            train_dir.mkdir(parents=True)
            (output / "stale.json").write_text("{}")
            with self.assertRaises(ValueError):
                training.confirm_output_overwrite(
                    output,
                    overwrite=True,
                    resume=None,
                    protected_paths=(str(train_dir),),
                )
            self.assertTrue(train_dir.exists())

    def _dataset(self, root: Path, name: str, symbols: list[str]) -> Path:
        directory = root / name
        directory.mkdir()
        (directory / "reaction_statistics.json").write_text(
            json.dumps(
                {
                    "chemistry_config": {
                        "fragmenter_params": {"test": True},
                        "adduct_type_strs": ["[M+H]+"],
                        "symbols": symbols,
                    }
                }
            )
        )
        return directory

    def test_symbol_order_is_canonicalized_like_atom_feature_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = self._dataset(root, "train", ["C", "N", "O", "Cl", "Br"])
            validation = self._dataset(root, "validation", ["Br", "C", "Cl", "N", "O"])
            sources = dataset_sources(train, validation)
            self.assertEqual(sources["symbols"], ("Br", "C", "Cl", "N", "O"))
            inherited = inherit_model_config(
                {"mol_encoder_params": {"node_dim": 16}}, train, validation
            )
            self.assertEqual(
                tuple(inherited["mol_encoder_params"]["symbols"]),
                ("Br", "C", "Cl", "N", "O"),
            )

    def test_inheritance_has_no_role_capacity_setting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = self._dataset(root, "train", ["C", "O"])
            validation = self._dataset(root, "validation", ["C", "O"])
            inherited = inherit_model_config(
                {"mol_encoder_params": {"node_dim": 16}}, train, validation
            )
            self.assertNotIn("max_roles", inherited.get("reaction_model_params", {}))

    def test_fresh_encoder_uses_configured_architecture_and_dataset_symbols(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = self._dataset(root, "train", ["C", "O"])
            validation = self._dataset(root, "validation", ["O", "C"])
            inherited = inherit_model_config(
                {
                    "mol_encoder_params": {
                        "symbols": ["Xe"],
                        "node_dim": 48,
                        "graph_dim": 192,
                        "num_layers": 3,
                        "num_heads": 6,
                    }
                },
                train,
                validation,
            )
            self.assertEqual(inherited["mol_encoder_params"]["symbols"], ["C", "O"])
            self.assertEqual(inherited["mol_encoder_params"]["node_dim"], 48)
            self.assertEqual(inherited["mol_encoder_params"]["graph_dim"], 192)

    def test_different_symbol_sets_remain_incompatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = self._dataset(root, "train", ["C", "N", "O"])
            validation = self._dataset(root, "validation", ["C", "N", "S"])
            with self.assertRaisesRegex(ValueError, "different elements"):
                dataset_sources(train, validation)

    def test_metadata_priority_and_stale_file_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "train"
            directory.mkdir()
            fragmenter = {"mass_tolerance": "0.01Da"}
            chemistry = {
                "fragmenter_params": fragmenter,
                "symbols": ["C", "O"],
                "adduct_type_strs": ["[M+H]+"],
            }
            (directory / "preparation_config.json").write_text(
                json.dumps({"chemistry_config": chemistry})
            )
            (directory / "fragment-tree.training-data.pft").write_text(
                json.dumps({"fragmenterParams": fragmenter, "symbols": ["O", "C"]})
            )
            (directory / "reaction_statistics.json").write_text(
                json.dumps({"chemistry_config": chemistry})
            )
            self.assertEqual(
                dataset_chemistry_config(directory),
                {**chemistry, "reaction_models": []},
            )
            stale = {
                "fragmenter_params": {"mass_tolerance": "0.5Da"},
                "symbols": ["C", "N"],
                "adduct_type_strs": ["[M+Na]+"],
            }
            (directory / "reaction_statistics.json").write_text(
                json.dumps({"chemistry_config": stale})
            )
            with self.assertRaisesRegex(ValueError, "Regenerate this dataset"):
                dataset_chemistry_config(directory)

    def test_molecular_encoder_symbols_must_match_prepared_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = self._dataset(root, "train", ["C", "N", "O"])
            validation = self._dataset(root, "validation", ["O", "N", "C"])
            sources = dataset_sources(train, validation)
            saved = {
                "fragmenter_params": sources["fragmenter_params"],
                "adduct_type_strs": sources["adduct_type_strs"],
                "mol_encoder_params": {"symbols": ("C", "N", "S")},
            }
            with self.assertRaisesRegex(ValueError, "molecular encoder checkpoint"):
                inherit_model_config({}, train, validation, saved_model=saved)

    def test_sample_meets_assignment_score_treats_missing_scores_as_passing(self):
        self.assertTrue(
            training._sample_meets_assignment_score(
                {"assignmentScore": None, "assignmentScoreWithoutPrecursor": None},
                0.9,
                0.9,
            )
        )
        self.assertTrue(
            training._sample_meets_assignment_score(
                {"assignmentScore": 0.9, "assignmentScoreWithoutPrecursor": None},
                0.5,
                0.9,
            )
        )
        self.assertFalse(
            training._sample_meets_assignment_score(
                {"assignmentScore": 0.4, "assignmentScoreWithoutPrecursor": None},
                0.5,
                0,
            )
        )
        self.assertFalse(
            training._sample_meets_assignment_score(
                {"assignmentScore": 1.0, "assignmentScoreWithoutPrecursor": 0.2}, 0, 0.5
            )
        )

    def test_assignment_score_filter_drops_low_scoring_samples_via_select_samples(self):
        # Data preparation always keeps every sample; training-time loading is
        # where a below-threshold sample now gets dropped, via select_samples.
        model = chemistry()
        model["fragmenter_params"]["fragment_ion_tree_builder"][
            "max_reaction_count"
        ] = 1
        builder = ReactionStructureBuilder(create_preparation_context(model))
        source = Compound.from_smiles("CCO")
        adduct = Adduct.parse("[M+H]+")
        precursor_mz = Formula.parse("C2H7O+").exact_mass
        # Sample 0 is a lone matched precursor peak (assignmentScore 1.0). Sample 1 adds
        # a large unmatched peak alongside the matched precursor, dragging its score down.
        data, kept = builder.build(
            source,
            [adduct, adduct],
            [20.0, 20.0],
            [[precursor_mz], [precursor_mz, 1000.0]],
            [[1.0], [1.0, 100.0]],
        )
        self.assertEqual(len(data.sample_annotations), 2)
        full_scores = [sample["assignmentScore"] for sample in data.sample_annotations]
        self.assertAlmostEqual(full_scores[0], 1.0)
        self.assertLess(full_scores[1], 0.5)
        surviving = [
            index
            for index, sample in enumerate(data.sample_annotations)
            if training._sample_meets_assignment_score(sample, 0.5, 0)
        ]
        self.assertEqual(surviving, [0])
        filtered = select_samples(data, surviving)
        self.assertEqual(filtered.num_samples, 1)
        self.assertAlmostEqual(filtered.sample_annotations[0]["assignmentScore"], 1.0)

    def test_select_samples_keeps_long_dtype_when_no_peak_paths_or_transitions_survive(
        self,
    ):
        # A lone matched precursor peak has no fragmentation steps, so
        # select_samples rebuilds teacher_peak_branch_group_index and
        # state_transition_next_state_index as empty lists. torch.tensor([])
        # silently defaults to float32, which used to crash a *second*
        # select_samples call (e.g. chunking by max_samples after assignment
        # score filtering) with "tensors used as indices must be long, int,
        # byte or bool tensors".
        model = chemistry()
        model["fragmenter_params"]["fragment_ion_tree_builder"][
            "max_reaction_count"
        ] = 1
        builder = ReactionStructureBuilder(create_preparation_context(model))
        source = Compound.from_smiles("CCO")
        adduct = Adduct.parse("[M+H]+")
        precursor_mz = Formula.parse("C2H7O+").exact_mass
        data, kept = builder.build(source, [adduct], [20.0], [[precursor_mz]], [[1.0]])
        filtered = select_samples(data, [0])
        self.assertEqual(filtered.conditions.branch_group_index.dtype, torch.long)
        self.assertEqual(filtered.reaction_contexts.edge_index.dtype, torch.long)
        select_samples(filtered, [0])  # must not raise


if __name__ == "__main__":
    unittest.main()
