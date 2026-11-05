"""MSDataset adapter for Source/reaction inference, with no full-tree precompute."""

from __future__ import annotations
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
import pandas as pd
import torch
from tqdm.auto import tqdm
from fredis.libs.mmkit.mmkit import Adduct, Compound
from fredis.domain.mass.parse_ce import parse_ce_to_ev
from fredis.utils.parallel_subprocess import run_parallel_subprocesses
from .prepare_worker import prepare_compound_cache
from .source_anchored_spectrum_predictor import SourceAnchoredFragmentSpectrumGenerator
from .spectrum_generator import (
    GeneratedMassSpectrum,
    GeneratedSpectrumPeak,
    FragmentSpectrumGeneratorOutput,
    fragment_spectrum_output_to_msdataset,
)


def build_prediction_cache(
    groups: dict[str, list[str]],
    prep_config: dict,
    *,
    num_workers: int = 1,
    chunk_size: int = 1,
    desc: str = "Preparing molecules",
    keep_temp: bool = False,
    temp_dir: str | None = None,
) -> dict[str, dict]:
    """groups: {smiles: [adduct_str, ...]} for every unique compound in the
    batch. Returns {smiles: result} (see prepare_worker.prepare_chunk) for
    every compound RDKit could process -- a compound left out (or marked
    ok=False) is resolved inline instead, by
    SourceAnchoredFragmentSpectrumGenerator.prepare()'s own fallback path.

    num_workers==1 runs in-process, with per-compound progress; num_workers>1
    fans out via fredis.utils.parallel_subprocess (real OS processes -- RDKit
    does not parallelize well across Python threads). The per-chunk task and
    result files those subprocesses exchange live under a temporary
    directory that is removed once every chunk has completed, unless
    keep_temp is set (for inspecting a chunk's inputs/outputs directly)."""
    if type(num_workers) is not int or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not groups:
        return {}
    from fredis.ml.data_preparation.fragment_tree.context import (
        create_preparation_context,
    )

    if num_workers == 1 or len(groups) < 2:
        fragmenter = create_preparation_context(prep_config).fragmenter
        results = {}
        for smiles, adduct_strs in tqdm(groups.items(), total=len(groups), desc=desc):
            try:
                results[smiles] = {
                    "ok": True,
                    **prepare_compound_cache(smiles, adduct_strs, fragmenter),
                }
            except Exception as error:
                results[smiles] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        return results
    items = list(groups.items())
    effective_chunk_size = min(chunk_size, max(1, len(items) // num_workers))
    results = {}
    directory = tempfile.mkdtemp(prefix="fredis-predict-prepare-", dir=temp_dir)
    try:
        commands = []
        for offset in range(0, len(items), effective_chunk_size):
            chunk = dict(items[offset : offset + effective_chunk_size])
            task_file = Path(directory) / f"task-{len(commands)}.pkl"
            result_file = task_file.with_suffix(".result.pkl")
            with task_file.open("wb") as stream:
                pickle.dump(
                    {"groups": chunk, "prep_config": prep_config},
                    stream,
                    pickle.HIGHEST_PROTOCOL,
                )
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "fredis.ml.specgen.prepare_worker",
                    "--task",
                    str(task_file),
                    "--result",
                    str(result_file),
                ]
            )

        def on_complete(command):
            with Path(command[-1]).open("rb") as stream:
                results.update(pickle.load(stream))

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        root = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
        run_parallel_subprocesses(
            commands,
            max_workers=min(num_workers, len(commands)),
            print_output=False,
            env=env,
            desc=desc,
            unit="chunk",
            on_complete=on_complete,
        )
    finally:
        if keep_temp:
            print(f"kept prepare task/result files: {directory}", file=sys.stderr)
        else:
            shutil.rmtree(directory, ignore_errors=True)
    return results


def _predict_packs(generator, sources, adducts, energies, precomputed,
                   valid_indices, smiles_values, include_formula_annotation,
                   intensity_threshold, packs, progress=None):
    """Predict packs atomically, bisecting failed packs down to single records."""
    spectra_by_position = {}
    failures = []
    def split_rows(rows: list[int]) -> tuple[list[int], list[int]]:
        """Prefer keeping same-compound rows together while isolating errors."""
        grouped: dict[str, list[int]] = {}
        for position in rows:
            grouped.setdefault(sources[position].smiles, []).append(position)
        groups = list(grouped.values())
        if len(groups) > 1:
            cumulative = 0
            target = len(rows) / 2
            split_at = 1
            best = float("inf")
            for index, group in enumerate(groups[:-1], start=1):
                cumulative += len(group)
                distance = abs(cumulative - target)
                if distance < best:
                    best = distance
                    split_at = index
            return [item for group in groups[:split_at] for item in group], [
                item for group in groups[split_at:] for item in group
            ]
        midpoint = len(rows) // 2
        return rows[:midpoint], rows[midpoint:]


    def predict_rows(rows: list[int]) -> None:
        local_spectra = {
            position: GeneratedMassSpectrum(position, []) for position in rows
        }
        error_name = error_message = None
        try:
            for output in generator.predict_batches(
                [sources[position] for position in rows],
                [adducts[position] for position in rows],
                [energies[position] for position in rows],
                precomputed=precomputed,
            ):
                prediction = output.spectra
                from .competitive_spectrum_model import merge_candidate_peaks

                prediction = merge_candidate_peaks(prediction)
                for sample, formula, mz, intensity, confidence in zip(
                    prediction.sample_index.detach().cpu().tolist(),
                    prediction.formula_tensor.detach().cpu(),
                    prediction.mz.detach().cpu().tolist(),
                    prediction.intensity.detach().cpu().tolist(),
                    prediction.confidence.detach().cpu().tolist(),
                ):
                    if intensity <= intensity_threshold:
                        continue
                    position = rows[output.sample_input_index[sample]]
                    local_spectra[position].peaks.append(
                        GeneratedSpectrumPeak(
                            mz=mz,
                            intensity=max(float(intensity), 0.0),
                            sample_id=position,
                            formula=(
                                str(
                                    generator.tensorizer.tensor_to_formula(
                                        formula
                                    )
                                )
                                if include_formula_annotation
                                else None
                            ),
                        )
                    )
            for spectrum in local_spectra.values():
                if spectrum.peaks:
                    maximum = max(peak.intensity for peak in spectrum.peaks)
                    for peak in spectrum.peaks:
                        peak.intensity = (
                            peak.intensity / maximum if maximum > 0 else 0.0
                        )
                    spectrum.peaks.sort(key=lambda peak: peak.mz)
        except Exception as error:
            error_name = type(error).__name__
            error_message = str(error)
            output = None
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        if error_name is None:
            spectra_by_position.update(local_spectra)
            if progress is not None:
                progress.update(len(rows))
            return
        if len(rows) == 1:
            position = rows[0]
            input_index = valid_indices[position]
            failures.append(
                {
                    "index": input_index,
                    "smiles": smiles_values[input_index],
                    "error": f"{error_name}: {error_message}",
                }
            )
            if progress is not None:
                progress.update(1)
            return
        left, right = split_rows(rows)
        predict_rows(left)
        predict_rows(right)


    for pack in packs:
        predict_rows(pack)
    return spectra_by_position, failures


_CPU_PREDICTION_CONTEXT = None


def _init_cpu_prediction(context):
    global _CPU_PREDICTION_CONTEXT
    # Avoid multiplying PyTorch's intra-op thread pool by the process count.
    torch.set_num_threads(1)
    _CPU_PREDICTION_CONTEXT = context


def _predict_cpu_pack(pack):
    spectra, failures = _predict_packs(*_CPU_PREDICTION_CONTEXT, [pack])
    return spectra, failures, len(pack)


def predict_source_msdataset(
    dataset: object,
    generator: SourceAnchoredFragmentSpectrumGenerator,
    *,
    smiles_column: str,
    adduct_type_column: str,
    collision_energy_column: str,
    precursor_mz_column: str,
    instrument_column: str | None,
    include_formula_annotation: bool = True,
    peak_intensity_threshold: float = 0.0,
    num_workers: int = 1,
    chunk_size: int = 1,
    keep_temp: bool = False,
    temp_dir: str | None = None,
    validate_desc: str = "Validating records",
    prepare_desc: str = "Preparing molecules",
    predict_desc: str = "Predicting spectra",
) -> tuple[object, pd.DataFrame]:
    if not 0 <= peak_intensity_threshold <= 1:
        raise ValueError("Peak intensity threshold must be between 0 and 1.")
    total = len(dataset)
    smiles_values = list(dataset[smiles_column])
    adduct_values = list(dataset[adduct_type_column])
    ce_values = list(dataset[collision_energy_column])
    mz_values = list(dataset[precursor_mz_column])
    instrument_values = (
        None if instrument_column is None else list(dataset[instrument_column])
    )

    # Phase 1: cheap, RDKit-free per-row validation only (adduct syntax, this
    # model's known adduct vocabulary, collision energy). The RDKit-heavy
    # per-compound work -- SMILES parsing and fragmentation -- happens once
    # per UNIQUE compound below, not once per row, since a real dataset
    # typically repeats the same compound across many adducts/energies.
    candidates, failures = [], []
    for i in tqdm(range(total), total=total, desc=validate_desc):
        try:
            adduct = Adduct.parse(str(adduct_values[i]))
            if str(adduct) not in generator.adduct_type_strs:
                # A syntactically valid adduct this particular model was
                # never trained on (unlike training's validation split, real
                # input data isn't limited to the model's known adduct
                # vocabulary). Left unchecked, this only fails deep inside
                # generator.prepare()'s conditions tensor, after grouping,
                # crashing every other sample sharing that compound.
                raise ValueError(
                    f"Adduct {adduct} is not one of this model's supported adduct types: {generator.adduct_type_strs}"
                )
            mz = float(mz_values[i])
            instrument = None if instrument_values is None else instrument_values[i]
            ev = parse_ce_to_ev(ce_values[i], mz, instrument)
            if ev is None:
                raise ValueError(
                    f"Cannot convert collision energy {ce_values[i]!r} to eV"
                )
        except Exception as error:
            # Broad on purpose: a malformed record can fail in ways this
            # function cannot enumerate up front (a bad adduct string or a
            # bug surfaced deep in a third-party parser). Any of those should
            # be skipped and reported, never abort every other spectrum.
            failures.append(
                {
                    "index": i,
                    "smiles": smiles_values[i],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        candidates.append((i, str(smiles_values[i]), adduct, float(ev)))
    print(
        f"{validate_desc}: {len(candidates)}/{total} records valid, {len(failures)} failed",
        file=sys.stderr,
    )

    # Phase 2: parse each unique compound once and reject structures the
    # checkpoint's molecular encoder cannot represent.  Fragmentation itself
    # can accept atoms (for example Na or Si) that the learned atom vocabulary
    # cannot encode, so leaving this check until generator.prepare() would let
    # one such compound abort the entire prediction batch.
    compound_cache: dict[str, Compound] = {}
    compound_errors: dict[str, str] = {}
    compatible_candidates = []
    graph_builder = generator.mol_encoder.graph_builder
    supported_symbols = set(graph_builder.symbols)
    for i, smiles, adduct, ev in candidates:
        if smiles not in compound_cache and smiles not in compound_errors:
            try:
                compound = Compound.from_smiles(smiles)
                unsupported = sorted(
                    {
                        atom.GetSymbol()
                        for atom in graph_builder.graph_atoms(compound)
                        if atom.GetSymbol() not in supported_symbols
                    }
                )
                if unsupported:
                    names = ", ".join(unsupported)
                    raise ValueError(
                        f"Unsupported atom symbol(s): {names}. "
                        f"Supported symbols are: {graph_builder.symbols}"
                    )
                compound_cache[smiles] = compound
            except Exception as error:
                compound_errors[smiles] = f"{type(error).__name__}: {error}"
        if smiles in compound_errors:
            failures.append(
                {"index": i, "smiles": smiles, "error": compound_errors[smiles]}
            )
            continue
        compatible_candidates.append((i, smiles, adduct, ev))

    # Phase 3: group by unique compound and precompute reaction relations, reactions plus
    # candidate precursor sequences once per compound, optionally in
    # parallel. This is the RDKit-bound step generator.prepare() would
    # otherwise run synchronously, per compound, inside predict_batches().
    groups: dict[str, list[str]] = {}
    for _, smiles, adduct, _ in compatible_candidates:
        requested = groups.setdefault(smiles, [])
        if str(adduct) not in requested:
            requested.append(str(adduct))
    prep_config = {
        "fragmenter_params": generator.fragmenter.to_dict(),
        "symbols": list(generator.mol_encoder.symbols),
        "adduct_type_strs": list(generator.adduct_type_strs),
    }
    cache = build_prediction_cache(
        groups,
        prep_config,
        num_workers=num_workers,
        chunk_size=chunk_size,
        desc=prepare_desc,
        keep_temp=keep_temp,
        temp_dir=temp_dir,
    )

    # Phase 4: a compound RDKit could not fragment, or an adduct with no
    # valid precursor reaction sequence for this source (an empty tuple --
    # source_reaction_structure.prepare_source_reactions raises
    # UnresolvedPrecursorError on that, deep inside the batched prediction
    # loop, taking every other sample sharing this compound down with it),
    # fails only its own rows instead of crashing
    # generator.predict_batches() for the rest of the dataset.
    # Parent-side Compound objects validated above are reused here; worker
    # processes return only picklable reactions and precursor sequences.
    sources, adducts, energies, valid_indices, precomputed = [], [], [], [], {}
    for i, smiles, adduct, ev in compatible_candidates:
        result = cache.get(smiles)
        if result is None or not result.get("ok"):
            error = (result or {}).get(
                "error", "RDKit fragmentation produced no result for this compound"
            )
            failures.append({"index": i, "smiles": smiles, "error": error})
            continue
        if not result["precursor_sequences"].get(str(adduct)):
            failures.append(
                {
                    "index": i,
                    "smiles": smiles,
                    "error": f"No valid precursor reaction sequence found for adduct {adduct} on this compound",
                }
            )
            continue
        if smiles not in precomputed:
            precomputed[smiles] = {
                "reactions": result["reactions"],
                "relations": result["relations"],
                "precursor_sequences": result["precursor_sequences"],
            }
        sources.append(compound_cache[smiles])
        adducts.append(adduct)
        energies.append(ev)
        valid_indices.append(i)

    # Phase 5: predict in the same source-grouped packs used by
    # generator.predict_batches(), but isolate a failing pack recursively.
    # Successful outputs are staged locally and committed only after the whole
    # pack finishes, so an exception after one or more yielded chunks cannot
    # duplicate peaks when that pack is retried.  A singleton failure is a
    # normal record-level failure and does not abort the remaining dataset.
    source_groups: dict[str, list[int]] = {}
    for position, source in enumerate(sources):
        source_groups.setdefault(source.smiles, []).append(position)
    cpu_parallel = num_workers > 1 and next(generator.parameters()).device.type == "cpu"
    limit = generator.max_samples
    if cpu_parallel:
        # Distribute even datasets that fit within a single model batch.
        limit = min(limit, max(1, (len(sources) + num_workers - 1) // num_workers))
    packs: list[list[int]] = []
    current: list[int] = []
    for rows in source_groups.values():
        if current and len(current) + len(rows) > limit:
            packs.append(current)
            current = []
        current.extend(rows)
        if len(current) >= limit:
            packs.append(current)
            current = []
    if current:
        packs.append(current)

    context = (generator, sources, adducts, energies, precomputed,
               valid_indices, smiles_values, include_formula_annotation,
               peak_intensity_threshold)
    spectra_by_position = {}
    if sources:
        with tqdm(total=len(sources), desc=predict_desc) as progress:
            if cpu_parallel and len(packs) > 1:
                # Spawn avoids inheriting active PyTorch/OpenMP threads.
                with ProcessPoolExecutor(
                    max_workers=min(num_workers, len(packs)),
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_cpu_prediction,
                    initargs=(context,),
                ) as executor:
                    for spectra, errors, count in executor.map(_predict_cpu_pack, packs):
                        spectra_by_position.update(spectra)
                        failures.extend(errors)
                            progress.update(count)
            else:
                spectra_by_position, errors = _predict_packs(
                    *context, packs, progress=progress
                )
                failures.extend(errors)

    successful_positions = sorted(spectra_by_position)
    spectra = []
    for sample_id, position in enumerate(successful_positions):
        spectrum = spectra_by_position[position]
        spectrum.sample_id = sample_id
        for peak in spectrum.peaks:
            peak.sample_id = sample_id
        spectra.append(spectrum)
    successful_indices = [valid_indices[position] for position in successful_positions]
    metadata = dataset.to_pandas().iloc[successful_indices].reset_index(drop=True)
    # Values actually fed to the model, not re-derived afterward from the raw
    # columns: the resolved main adduct (the base ion rule the model used,
    # e.g. [M+H-H2O]+ -> [M+H]+) and the numeric collision energy in eV.
    metadata["MainAdduct"] = [
        str(generator.fragmenter._resolve_main_adduct_type(adducts[position]))
        for position in successful_positions
    ]
    metadata["ModelCollisionEnergyEV"] = [
        energies[position] for position in successful_positions
    ]
    predicted = fragment_spectrum_output_to_msdataset(
        FragmentSpectrumGeneratorOutput(spectra), metadata=metadata
    )
    failure_frame = pd.DataFrame(failures, columns=["index", "smiles", "error"])
    if len(failure_frame):
        failure_frame.sort_values(
            "index", kind="stable", inplace=True, ignore_index=True
        )
    return predicted, failure_frame
