"""Real dataset inspection and train/validation preparation regressions."""

from fredis.ml.input.adduct_data import AdductData
import contextlib
import importlib.util
import io
import json
import torch
import tempfile
import unittest
import unittest.mock
from pathlib import Path
import numpy as np
import pandas as pd
from fredis.data import MSDataset
from fredis.data.core.PeakSeries import PeakSeries
from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.ml.data_preparation.fragment_tree.datasets import (
    load_spectrum_dataset,
    split_by_smiles,
)
from fredis.ml.data_preparation.fragment_tree.create_training_data import (
    main,
    _confirm_existing_output,
)
from fredis.ml.data_preparation.fragment_tree.context import create_preparation_context
from fredis.ml.data_preparation.fragment_tree.record_validation import inspect_records
from .TestReactionPipeline import chemistry, config

class TestPreparationWorkflow(unittest.TestCase):
    def dataset(self):
        smiles = ["CCO", "CCO", "CCN", "CCC", "COC"]
        mzs = [
            Compound.from_smiles(value).formula.exact_mass + 1.007276466
            for value in smiles
        ]
        metadata = pd.DataFrame(
            {
                "SMILES": smiles,
                "AdductType": ["[M+H]+"] * 5,
                "CollisionEnergy": [20] * 5,
                "PrecursorMZ": mzs,
            }
        )
        peaks = np.array([[mz, 1] for mz in mzs], dtype=float)
        return MSDataset(metadata, PeakSeries(peaks, np.arange(6, dtype=np.int64)))

    def test_existing_output_is_confirmed_before_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            args = type("Args", (), {"output_dir": directory, "overwrite": 0})()
            with (
                unittest.mock.patch("sys.stdin.isatty", return_value=True),
                unittest.mock.patch("builtins.input", return_value="yes") as prompt,
            ):
                _confirm_existing_output(args)
            prompt.assert_called_once()
            self.assertEqual(args.overwrite, 1)
            args.overwrite = 0
            with (
                unittest.mock.patch("sys.stdin.isatty", return_value=True),
                unittest.mock.patch("builtins.input", return_value="no"),
            ):
                with self.assertRaisesRegex(FileExistsError, "Cancelled"):
                    _confirm_existing_output(args)

    def test_split_is_deterministic_and_keeps_source_groups(self):
        data = self.dataset()
        train, validation = split_by_smiles(data, "SMILES", 0.5, 7)
        train2, validation2 = split_by_smiles(data, "SMILES", 0.5, 7)
        self.assertFalse(set(train["SMILES"]) & set(validation["SMILES"]))
        self.assertEqual(len(train) + len(validation), len(data))
        self.assertEqual(validation["SMILES"].tolist(), validation2["SMILES"].tolist())

    def test_csv_loader_uses_official_spectrum_table_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "spectra.csv"
            pd.DataFrame({"SMILES": ["CCO"], "Peak": ["100,0.5;150,1"]}).to_csv(
                file, index=False
            )
            data = load_spectrum_dataset(file)
            self.assertEqual(data[0].peaks.mz.tolist(), [100, 150])
            self.assertEqual(data[0].peaks.intensity.tolist(), [0.5, 1])



    def test_three_reaction_limit_keeps_exact_two_reaction_annotation(self):
        from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
        from fredis.ml.input.structure_builder import ReactionStructureBuilder

        cfg = config()
        cfg["fragmenter_params"]["fragment_ion_tree_builder"]["max_reaction_count"] = 3
        generator = create_spectrum_generator(cfg)
        source = Compound.from_smiles("CCCCCO")
        reference = generator.fragmenter.build_fragment_ion_tree(
            source, max_reaction_count=2
        )
        limit = reference.search_stats["num_raw_combinations"]
        builder = ReactionStructureBuilder(generator, max_reaction_combinations=limit)
        data, kept = builder.build(
            source, [Adduct.parse("[M+H]+")], [20.0], [[103.0759]], [[1.0]]
        )
        self.assertEqual(tuple(kept), (0,))
        self.assertEqual(builder.last_search_stats["completed_max_reaction_count"], 2)
        self.assertEqual(builder.last_search_stats["discarded_from_reaction_count"], 3)
        self.assertEqual(builder.last_search_stats["num_raw_combinations"], limit)
        self.assertEqual(
            set(builder.last_reaction_sequence_compounds),
            set(reference.reaction_sequence_compounds),
        )
        self.assertTrue(
            all(
                len(sequence.reactions) <= 2
                for sequence in builder.last_reaction_sequence_compounds
            )
        )
        self.assertTrue(data.sample_annotations)

    def test_tree_limits_save_previous_complete_depth_serial_and_parallel(self):
        from fredis.ml.data_preparation.fragment_tree.create_training_data import (
            create_reaction_training_data,
        )

        metadata = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCO", "C"],
                "AdductType": ["[M+H]+"] * 3,
                "CollisionEnergy": [20] * 3,
                "PrecursorMZ": [47, 47, 17],
            }
        )
        data = MSDataset(
            metadata,
            PeakSeries(
                np.array([[47.0, 1.0], [47.0, 1.0], [17.0, 1.0]]),
                np.arange(4, dtype=np.int64),
            ),
        )
        model = chemistry()
        model["fragmenter_params"]["fragment_ion_tree_builder"][
            "max_reaction_count"
        ] = 1
        with tempfile.TemporaryDirectory() as directory:
            for workers in (1, 2):
                for limit in (
                    {"max_unique_fragment_smiles": 1},
                    {"max_reaction_combinations": 1},
                ):
                    output = Path(directory) / f"{workers}-{next(iter(limit))}"
                    files = create_reaction_training_data(
                        dataset=data,
                        chemistry_config=model,
                        output_dir=output,
                        num_workers=workers,
                        chunk_size=3,
                        **limit,
                    )
                    self.assertEqual(len(files), 2)
                    manifest = pd.read_csv(output / "manifest.tsv", sep="\t")
                    retained = manifest[manifest.smiles == "CCO"].iloc[0]
                    self.assertEqual(retained.status, "completed")
                    self.assertEqual(retained.num_valid_samples, 2)
                    self.assertEqual(retained.completed_max_reaction_count, 0)
                    self.assertTrue(retained.search_truncated)
                    self.assertEqual(retained.discarded_from_reaction_count, 1)
                    payload = torch.load(
                        output / "data" / retained.file, weights_only=False
                    )
                    self.assertEqual(
                        payload["metadata"]["search_stats"][
                            "completed_max_reaction_count"
                        ],
                        0,
                    )
                    self.assertEqual(
                        payload["metadata"]["search_stats"]["limit_attempts"][0][
                            "limit_name"
                        ],
                        next(iter(limit)),
                    )

    def test_explicit_hydrogen_preparation_preserves_graph_and_reaction_alignment(self):
        from fredis.ml.input.structure_builder import ReactionStructureBuilder
        from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
        from fredis.libs.mmkit.mmkit import Adduct
        import torch

        model = config()
        model["fragmenter_params"]["fragment_ion_tree_builder"][
            "max_reaction_count"
        ] = 1
        generator = create_spectrum_generator(model).eval()
        source = Compound.from_smiles("C([2H])O")
        from fredis.ml.input._reaction_preparation import prepare_source_reactions

        reactions = (
            generator.fragmenter.fragment_ion_tree_builder.create_reaction_instances(
                source
            )
        )
        data = prepare_source_reactions(
            source=source,
            reactions=reactions,
            graph_builder=generator.mol_encoder.graph_builder,
            collision_energy_eV=torch.tensor([20.0]),
            main_adduct_index=torch.tensor([0]),
            adducts=AdductData.from_fragmenter(generator.fragmenter),
            max_reaction_count=1,
        )
        self.assertEqual(data.source_graph.num_nodes, 2)
        self.assertEqual(data.source_atom_capacity, 2)
        self.assertTrue(torch.all(data.reactions.reaction_source_atom_index < 2))
        self.assertEqual(
            int(data.reactions.reaction_source_atom_ptr[-1]),
            data.reactions.reaction_source_atom_index.numel(),
        )
        batch, indices = generator.mol_encoder.encode_batch(
            [source, Compound.from_smiles("[H]")]
        )
        self.assertEqual(indices.tolist(), [0, 1])
        self.assertEqual(batch.shape[0], 2)

    def test_shared_subprocess_runner_reports_completion_order(self):
        import sys
        from fredis.utils.parallel_subprocess import run_parallel_subprocesses

        commands = [
            [sys.executable, "-c", "import time; time.sleep(0.3)"],
            [sys.executable, "-c", "pass"],
        ]
        completed = []
        with contextlib.redirect_stderr(io.StringIO()):
            run_parallel_subprocesses(
                commands, max_workers=2, on_complete=completed.append
            )
        self.assertEqual(completed, [commands[1], commands[0]])

    def test_overwrite_removes_all_previous_outputs_and_preserves_complete_root_config(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_file = root / "train.msds"
            validation_file = root / "validation.msds"
            data = self.dataset()
            data[[0, 1, 2]].save(str(input_file))
            data[[3, 4]].save(str(validation_file))
            output = root / "out"
            (output / "train_structures/stale").mkdir(parents=True)
            (output / "train_structures/stale/old.bin").write_text("stale")
            (output / "orphan.txt").write_text("old")
            (output / "fragment-tree.training-data.pft").write_text(
                json.dumps({"validationRatio": 0.25})
            )
            model = chemistry()
            model["fragmenter_params"]["fragment_ion_tree_builder"][
                "max_reaction_count"
            ] = 1
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                main(
                    [
                        "--input",
                        str(input_file),
                        "--validation-input",
                        str(validation_file),
                        "--output-dir",
                        str(output),
                        "--params-json",
                        json.dumps(model),
                        "--max-unique-fragment-smiles",
                        "100",
                        "--max-reaction-combinations",
                        "200",
                        "--overwrite",
                        "1",
                        "--minimum-relative-intensity",
                        "0.2",
                        "--validation-seed",
                        "7",
                    ]
                )
            self.assertFalse((output / "orphan.txt").exists())
            self.assertFalse((output / "train_structures/stale").exists())
            saved = json.loads(
                (
                    output / "train_structures/fragment-tree.training-data.pft"
                ).read_text()
            )
            self.assertEqual(
                (output / "train_structures" / saved["validationInput"]).resolve(),
                validation_file.resolve(),
            )
            self.assertEqual(saved["validationRatio"], 0.25)
            self.assertEqual(saved["validationSeed"], 7)
            self.assertEqual(saved["maxUniqueFragmentSmiles"], 100)
            self.assertEqual(saved["maxReactionCombinations"], 200)
            prepared = json.loads((output / "fragment-tree.training-data-config.pft").read_text())
            self.assertFalse((output / "preparation_config.json").exists())
            self.assertFalse((output / "fragment-tree.training-data.pft").exists())
            self.assertEqual(
                json.loads((output / "fragment-tree.fragmenter.pft").read_text()),
                prepared["chemistry_config"]["fragmenter_params"],
            )
            for split in ("train_structures", "validation_structures"):
                for filename in ("fragment-tree.training-data.pft", "invalid_records.json",
                                 "manifest.tsv", "assignment_scores.tsv",
                                 "reaction_statistics.json", "skipped_sources.json"):
                    self.assertTrue((output / split / filename).is_file(), (split, filename))
            self.assertEqual(prepared["max_unique_fragment_smiles"], 100)
            self.assertEqual(prepared["max_reaction_combinations"], 200)
            self.assertEqual(saved["minimumRelativeIntensity"], 0.2)
            self.assertFalse(list(output.glob("*/preparation_config.json")))

    def test_assignment_scores_handle_precursor_only_and_zero_intensity(self):
        from fredis.libs.mmkit.mmkit import Adduct, Formula
        from fredis.ml.data_preparation.fragment_tree.context import (
            create_preparation_context,
        )
        from fredis.ml.input.structure_builder import ReactionStructureBuilder
        from fredis.ml.input.source_reaction_structure import SourceReactionStructure

        model = chemistry()
        model["fragmenter_params"]["fragment_ion_tree_builder"][
            "max_reaction_count"
        ] = 1
        builder = ReactionStructureBuilder(create_preparation_context(model))
        source = Compound.from_smiles("CCO")
        adduct = Adduct.parse("[M+H]+")
        mz = Formula.parse("C2H7O+").exact_mass
        data, kept = builder.build(
            source, [adduct, adduct], [20.0, 20.0], [[mz], [mz]], [[1.0], [0.0]]
        )
        first, second = data.sample_annotations
        self.assertEqual(first["assignmentScore"], 1.0)
        self.assertIsNone(first["assignmentScoreWithoutPrecursor"])
        self.assertIsNone(second["assignmentScore"])
        self.assertIsNone(second["assignmentScoreWithoutPrecursor"])
        self.assertGreater(len(first["peaks"][0]["matches"]), 0)
        collated = SourceReactionStructure.from_structures([data, data])
        self.assertEqual(len(collated.sample_annotations), 4)
        offset = len(data.spectrum_batch.decoded.compounds)
        original = first["peaks"][0]["matches"][0]["nodeIds"]
        shifted = collated.sample_annotations[2]["peaks"][0]["matches"][0]["nodeIds"]
        self.assertEqual(shifted, original)


    def test_split_cli_creates_both_structure_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "dataset.msds"
            self.dataset().save(str(file))
            model = chemistry()
            model["fragmenter_params"]["fragment_ion_tree_builder"][
                "max_reaction_count"
            ] = 1
            with contextlib.redirect_stdout(io.StringIO()) as log:
                main(
                    [
                        "--input",
                        str(file),
                        "--output-dir",
                        str(root / "out"),
                        "--validation-ratio",
                        "0.5",
                        "--params-json",
                        json.dumps(model),
                        "--minimum-relative-intensity",
                        "0.1",
                        "--overwrite",
                        "0",
                    ]
                )
            self.assertEqual(
                len(list((root / "out/train_structures").rglob("*.preft.pt"))), 2
            )
            self.assertEqual(
                len(list((root / "out/validation_structures").rglob("*.preft.pt"))), 2
            )
            self.assertTrue((root / "out/fragment-tree.training-data-config.pft").is_file())
            self.assertFalse(
                (root / "out/train_structures/preparation_config.json").exists()
            )
            self.assertFalse(
                (root / "out/validation_structures/preparation_config.json").exists()
            )
            self.assertFalse((root / "out/fragment-tree.pft").exists())
            self.assertTrue(
                (root / "out/train_structures/fragment-tree.training-data.pft").exists()
            )
            self.assertTrue(
                (
                    root / "out/validation_structures/fragment-tree.training-data.pft"
                ).exists()
            )
            for split in ("train", "validation"):
                folder = root / "out" / f"{split}_structures"
                self.assertEqual(len(list((folder / "data").glob("*.preft.pt"))), 2)
                rows = pd.read_csv(folder / "manifest.tsv", sep="\t")
                self.assertEqual(len(rows), 2)
                self.assertTrue(
                    all((folder / "data" / file).exists() for file in rows["file"])
                )
            # No per-source "progress"/"source_skipped" events on stdout: a
            # plain print there breaks a live tqdm bar's in-place redraw on
            # a large dataset (thousands of sources). tqdm's own postfix
            # already shows prepared/skipped live, and skipped_sources.json
            # / manifest.tsv already report the same data once the run ends.
            self.assertNotIn('"event": "progress"', log.getvalue())
            self.assertNotIn('"event": "source_skipped"', log.getvalue())
            with (
                self.assertRaises(FileExistsError),
                contextlib.redirect_stdout(io.StringIO()),
                unittest.mock.patch(
                    "fredis.ml.data_preparation.fragment_tree.create_training_data.load_spectrum_dataset",
                    side_effect=AssertionError(
                        "existing output must be rejected before loading the dataset"
                    ),
                ),
            ):
                main(
                    [
                        "--input",
                        str(file),
                        "--output-dir",
                        str(root / "out"),
                        "--validation-ratio",
                        "0.5",
                        "--params-json",
                        json.dumps(model),
                        "--overwrite",
                        "0",
                    ]
                )



    def test_preparation_context_needs_only_chemistry(self):
        from fredis.ml.data_preparation.fragment_tree.context import (
            create_preparation_context,
        )

        model = chemistry()
        context = create_preparation_context(model)
        self.assertEqual(
            context.adduct_type_strs,
            tuple(str(adduct) for adduct in context.fragmenter.adduct_types),
        )
        self.assertEqual(set(context.mol_encoder.symbols), set(model["symbols"]))
        self.assertFalse(hasattr(context, "feature_model"))

    def test_preparation_cli_has_no_model_options(self):
        from fredis.ml.data_preparation.fragment_tree.create_training_data import (
            build_arg_parser,
        )

        parser = build_arg_parser()
        base = ["--input", "missing.msds", "--output-dir", "unused"]
        self.assertFalse(parser.parse_args(base).save_all_edges)
        self.assertTrue(parser.parse_args(base + ["--save-all-edges"]).save_all_edges)
        options = parser._option_string_actions
        for flag in (
            "--node-dim",
            "--max-fragment-nodes",
            "--mol-encoder-params-json",
            "--reaction-model-params-json",
        ):
            self.assertNotIn(flag, options)
        with self.assertRaisesRegex(ValueError, "Specify the Fragmenter"):
            main(["--input", "missing.msds", "--output-dir", "unused"])


    def test_parallel_preparation_matches_serial_outputs(self):
        from fredis.ml.data_preparation.fragment_tree.create_training_data import (
            create_reaction_training_data,
            build_arg_parser,
        )
        from fredis.ml.input.source_reaction_structure import SourceReactionStructure
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = chemistry()
            model["fragmenter_params"]["fragment_ion_tree_builder"][
                "max_reaction_count"
            ] = 1
            serial = create_reaction_training_data(
                dataset=self.dataset(),
                chemistry_config=model,
                output_dir=root / "serial",
            )
            parallel = create_reaction_training_data(
                dataset=self.dataset(),
                chemistry_config=model,
                output_dir=root / "parallel",
                num_workers=2,
                chunk_size=2,
            )
            self.assertEqual(
                [file.name for file in serial], [file.name for file in parallel]
            )
            for left, right in zip(serial, parallel):
                first = SourceReactionStructure.load(left)
                second = SourceReactionStructure.load(right)
                self.assertTrue(
                    torch.equal(first.collision_energy_eV, second.collision_energy_eV)
                )
                self.assertEqual(first.collision_energy_eV.dtype, torch.float32)
                self.assertEqual(
                    first.branch_groups.main_adduct_index.dtype, torch.int64
                )
                self.assertTrue(
                    torch.equal(
                        first.branch_groups.main_adduct_index,
                        second.branch_groups.main_adduct_index,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        first.reaction_contexts.edge_label,
                        second.reaction_contexts.edge_label,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        first.spectrum_batch.target_intensity,
                        second.spectrum_batch.target_intensity,
                    )
                )
            stats = json.loads((root / "parallel/reaction_statistics.json").read_text())
            self.assertEqual(stats["num_workers"], 2)

    def test_parallel_preparation_bounds_temp_files_to_num_workers_at_a_time(self):
        # Every chunk's task file used to be written to --output-dir upfront,
        # for the whole dataset, before a single subprocess started -- on a
        # NIST-scale run this can duplicate close to the full dataset's size
        # on disk before any of it is freed. Dispatch now happens in waves
        # of at most num_workers chunks, each wave's files deleted once
        # consumed, so no single run_parallel_subprocesses call (and thus no
        # single on-disk moment) ever holds more than num_workers chunks.
        import fredis.ml.data_preparation.fragment_tree.create_training_data as create_training_data_module
        from fredis.ml.data_preparation.fragment_tree.create_training_data import (
            create_reaction_training_data,
        )

        max_seen = 0
        real_run_parallel_subprocesses = (
            create_training_data_module.run_parallel_subprocesses
        )

        def spy(commands_list, *args, **kwargs):
            nonlocal max_seen
            max_seen = max(max_seen, len(commands_list))
            return real_run_parallel_subprocesses(commands_list, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = chemistry()
            model["fragmenter_params"]["fragment_ion_tree_builder"][
                "max_reaction_count"
            ] = 1
            with unittest.mock.patch.object(
                create_training_data_module, "run_parallel_subprocesses", spy
            ):
                create_reaction_training_data(
                    dataset=self.dataset(),
                    chemistry_config=model,
                    output_dir=root / "parallel",
                    num_workers=2,
                    chunk_size=1,
                )
        self.assertGreater(max_seen, 0)
        self.assertLessEqual(max_seen, 2)

    def test_unexpected_error_in_one_source_is_skipped_not_fatal(self):
        import fredis.ml.data_preparation.fragment_tree.create_training_data as create_training_data_module
        from fredis.ml.data_preparation.fragment_tree.create_training_data import (
            create_reaction_training_data,
        )

        real_prepare_group = create_training_data_module._prepare_group

        def failing_prepare_group(task, builder, options):
            if task[1] == "CCN":
                raise ValueError("Unsupported atom symbol: 'As'")
            return real_prepare_group(task, builder, options)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = chemistry()
            model["fragmenter_params"]["fragment_ion_tree_builder"][
                "max_reaction_count"
            ] = 1
            with unittest.mock.patch.object(
                create_training_data_module, "_prepare_group", failing_prepare_group
            ):
                files = create_reaction_training_data(
                    dataset=self.dataset(),
                    chemistry_config=model,
                    output_dir=root / "serial",
                )
            skipped = json.loads((root / "serial/skipped_sources.json").read_text())
        self.assertEqual(len(files), 3)
        self.assertEqual([source["smiles"] for source in skipped], ["CCN"])
        self.assertIn("ValueError", skipped[0]["reason"])
        self.assertIn("Traceback", skipped[0]["error"])

    def test_crashed_worker_only_skips_the_source_that_crashes(self):
        # A worker process dying outright (native crash, OOM kill) used to
        # abort the whole run. Its chunk is now re-run one source per
        # process, so only the source that still crashes on its own is lost.
        import pickle, sys
        import fredis.ml.data_preparation.fragment_tree.create_training_data as create_training_data_module
        from fredis.ml.data_preparation.fragment_tree.create_training_data import (
            create_reaction_training_data,
        )

        real_run_parallel_subprocesses = (
            create_training_data_module.run_parallel_subprocesses
        )
        crash = [sys.executable, "-c", "import sys; sys.exit(3)"]

        def spy(commands_list, *args, **kwargs):
            def sources(command):
                with open(command[command.index("--task") + 1], "rb") as stream:
                    return [task[1] for task in pickle.load(stream)["tasks"]]

            patched = [
                crash + command[3:] if "CCN" in sources(command) else command
                for command in commands_list
            ]
            return real_run_parallel_subprocesses(patched, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = chemistry()
            model["fragmenter_params"]["fragment_ion_tree_builder"][
                "max_reaction_count"
            ] = 1
            with (
                unittest.mock.patch.object(
                    create_training_data_module, "run_parallel_subprocesses", spy
                ),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                files = create_reaction_training_data(
                    dataset=self.dataset(),
                    chemistry_config=model,
                    output_dir=root / "parallel",
                    num_workers=2,
                    chunk_size=2,
                )
            skipped = json.loads((root / "parallel/skipped_sources.json").read_text())
            leftovers = [
                path.name for path in (root / "parallel").glob("fredis-preparation-*/*")
            ]
        self.assertEqual(len(files), 3)
        self.assertEqual([source["smiles"] for source in skipped], ["CCN"])
        self.assertIn("status 3", skipped[0]["reason"])
        self.assertEqual(skipped[0]["category"], "worker_crash")
        self.assertEqual(leftovers, [])

    def test_manifest_and_reaction_statistics_record_search_statistics(self):
        from fredis.ml.data_preparation.fragment_tree.create_training_data import (
            create_reaction_training_data,
            SEARCH_STAT_FIELDS,
        )
        from fredis.ml.data_preparation.fragment_tree.context import (
            create_preparation_context,
        )

        model = chemistry()
        model["fragmenter_params"]["fragment_ion_tree_builder"][
            "max_reaction_count"
        ] = 2
        fragmenter = create_preparation_context(model).fragmenter
        expected = {
            smiles: fragmenter.build_fragment_ion_tree(
                Compound.from_smiles(smiles)
            ).search_stats
            for smiles in ("CCO", "CCN", "CCC", "COC")
        }
        with tempfile.TemporaryDirectory() as directory:
            for workers in (1, 2):
                with self.subTest(workers=workers):
                    output = Path(directory) / str(workers)
                    with contextlib.redirect_stderr(io.StringIO()):
                        create_reaction_training_data(
                            dataset=self.dataset(),
                            chemistry_config=model,
                            output_dir=output,
                            num_workers=workers,
                            chunk_size=1,
                        )
                    manifest = pd.read_csv(output / "manifest.tsv", sep="\t").set_index(
                        "smiles"
                    )
                    self.assertTrue(set(SEARCH_STAT_FIELDS) <= {*manifest.columns})
                    self.assertIn("skip_category", manifest.columns)
                    for smiles, stats in expected.items():
                        for field in SEARCH_STAT_FIELDS[:6]:
                            self.assertEqual(
                                manifest.loc[smiles, field],
                                stats[field],
                                (smiles, field),
                            )
                    summary = json.loads(
                        (output / "reaction_statistics.json").read_text()
                    )
                    for field, key in (
                        ("num_primitive_reactions", "max_primitive_reactions"),
                        ("num_raw_combinations", "max_raw_combinations"),
                        ("num_compiled_sequences", "max_compiled_sequences"),
                        ("num_rdkit_run_reactants", "max_rdkit_run_reactants"),
                        ("num_generated_fragments", "max_generated_fragments"),
                        (
                            "num_unique_fragment_smiles",
                            "max_observed_unique_fragment_smiles",
                        ),
                    ):
                        best = max(stats[field] for stats in expected.values())
                        self.assertEqual(summary[key], best)
                        # Ties resolve to the smallest SMILES, independent of worker order.
                        self.assertEqual(
                            summary[key + "_smiles"],
                            min(
                                smiles
                                for smiles, stats in expected.items()
                                if stats[field] == best
                            ),
                        )
                    self.assertEqual(
                        summary["search_limits"],
                        {
                            "max_unique_fragment_smiles": -1,
                            "max_reaction_combinations": -1,
                        },
                    )
                    self.assertEqual(summary["num_skipped_sources_by_category"], {})
                    self.assertEqual(summary["num_limit_skipped_sources"], {})

    def test_other_skip_reasons_are_distinguished_from_limits(self):
        import fredis.ml.data_preparation.fragment_tree.create_training_data as create_training_data_module
        from fredis.ml.data_preparation.fragment_tree.create_training_data import (
            create_reaction_training_data,
        )
        from fredis.ml.input.structure_builder import ReactionStructureBuilder

        real_build = ReactionStructureBuilder.build

        def build(builder, source, *args, **kwargs):
            if source.smiles == "CCN":
                real_build(builder, Compound.from_smiles("CCN"), *args, **kwargs)
                raise RuntimeError("after the tree search")
            return real_build(builder, source, *args, **kwargs)

        model = chemistry()
        model["fragmenter_params"]["fragment_ion_tree_builder"][
            "max_reaction_count"
        ] = 1
        with (
            tempfile.TemporaryDirectory() as directory,
            unittest.mock.patch.object(ReactionStructureBuilder, "build", build),
        ):
            output = Path(directory)
            with contextlib.redirect_stderr(io.StringIO()):
                create_reaction_training_data(
                    dataset=self.dataset(),
                    chemistry_config=model,
                    output_dir=output,
                    max_reaction_combinations=1,
                )
            skipped = {
                source["smiles"]: source
                for source in json.loads((output / "skipped_sources.json").read_text())
            }
            manifest = pd.read_csv(output / "manifest.tsv", sep="\t").set_index(
                "smiles"
            )
            summary = json.loads((output / "reaction_statistics.json").read_text())
        for smiles in ("CCO", "CCC", "COC"):
            self.assertEqual(manifest.loc[smiles, "status"], "completed")
            self.assertEqual(manifest.loc[smiles, "completed_max_reaction_count"], 0)
        self.assertEqual(skipped["CCN"]["category"], "error")
        self.assertEqual(summary["num_limit_skipped_sources"], {})
        with (
            tempfile.TemporaryDirectory() as directory,
            unittest.mock.patch.object(ReactionStructureBuilder, "build", build),
        ):
            output = Path(directory)
            with contextlib.redirect_stderr(io.StringIO()):
                create_reaction_training_data(
                    dataset=self.dataset(), chemistry_config=model, output_dir=output
                )
            skipped = {
                source["smiles"]: source
                for source in json.loads((output / "skipped_sources.json").read_text())
            }
            manifest = pd.read_csv(output / "manifest.tsv", sep="\t").set_index(
                "smiles"
            )
            summary = json.loads((output / "reaction_statistics.json").read_text())
        self.assertEqual(list(skipped), ["CCN"])
        self.assertEqual(skipped["CCN"]["category"], "error")
        self.assertEqual(manifest.loc["CCN", "skip_category"], "error")
        self.assertNotIn("limit", skipped["CCN"])
        # Statistics of a tree search that finished before a later error are still reported.
        from fredis.ml.data_preparation.fragment_tree.context import (
            create_preparation_context,
        )

        expected = (
            create_preparation_context(model)
            .fragmenter.build_fragment_ion_tree(Compound.from_smiles("CCN"))
            .search_stats
        )
        for field, value in expected.items():
            self.assertEqual(skipped["CCN"]["search_stats"][field], value)
        self.assertEqual(
            manifest.loc["CCN", "num_primitive_reactions"],
            expected["num_primitive_reactions"],
        )
        self.assertEqual(summary["num_skipped_sources_by_category"], {"error": 1})
        self.assertEqual(summary["num_limit_skipped_sources"], {})


    def test_full_graph_storage_through_parallel_preparation(self):
        from fredis.ml.data_preparation.fragment_tree.create_training_data import create_reaction_training_data
        from fredis.ml.input.source_reaction_structure import SourceReactionStructure

        model = chemistry()
        model["fragmenter_params"]["fragment_ion_tree_builder"]["max_reaction_count"] = 1
        with tempfile.TemporaryDirectory() as directory:
            for save_all in (False, True):
                files = create_reaction_training_data(
                    dataset=self.dataset(), chemistry_config=model,
                    output_dir=Path(directory) / str(save_all),
                    save_all_edges=save_all, num_workers=2, chunk_size=2,
                )
                self.assertEqual(len(files), 4)
                for path in files:
                    data = SourceReactionStructure.load(path)
                    # This dataset observes only precursor peaks: no positive branches.
                    # Root continuations are weak negatives even without positives.
                    self.assertGreater(data.compound_graph.reaction_index.numel(), 0)
                    self.assertTrue((data.reaction_contexts.edge_label == 0).all())
