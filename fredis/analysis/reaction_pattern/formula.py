"""Peak formula assignment alongside Reaction Pattern Analysis.

Spectra are grouped by precursor formula as in Assign Formula, so the
subformula candidates of one formula are generated once and a worker chunk
always holds whole formula groups (compounds sharing a precursor formula).
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import math
import multiprocessing
import sys

from tqdm import tqdm

from fredis.domain.formula._assignment import (
    PrecursorMZMismatch,
    precursor_grouping_formula,
)
from fredis.domain.formula._candidate import estimate_subformula_candidate_count
from fredis.domain.formula.utils import (
    assign_formulas_to_peaks,
    get_possible_sub_formulas,
)
from fredis.domain.mass.tolerance import (
    MassTolerance,
    format_mass_tolerance,
    parse_mass_tolerance,
)

from .compounds import CompoundGroup

ASSIGNED = "assigned"


@dataclass(frozen=True)
class FormulaAssignmentOptions:
    """Assign Formula settings; tolerance and columns come from the analysis."""

    max_formula_candidates: int | None = None
    timeout_sec: float | None = None
    hydrogen_delta: int = 1
    enable_neutral_loss: bool = False
    # Default intensity coverage of the result; the other mode is stored too.
    intensity_coverage_formula_only: bool = True

    @classmethod
    def from_dict(cls, value: dict) -> "FormulaAssignmentOptions":
        """Read the Workbench settings keys (maxFormulaCandidates, timeoutSec, ...)."""
        unknown = set(value) - {
            "maxFormulaCandidates",
            "timeoutSec",
            "hydrogenDelta",
            "enableNeutralLoss",
            "intensityCoverageFormulaOnly",
        }
        if unknown:
            raise ValueError(
                f"Unknown formula assignment settings: {', '.join(sorted(unknown))}"
            )
        maximum = value.get("maxFormulaCandidates")
        timeout = value.get("timeoutSec")
        hydrogen = value.get("hydrogenDelta", 1)
        neutral_loss = value.get("enableNeutralLoss", False)
        if maximum not in (None, "") and (
            isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1
        ):
            raise ValueError("maxFormulaCandidates must be a positive integer.")
        if timeout not in (None, "") and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not timeout > 0
        ):
            raise ValueError("timeoutSec must be positive.")
        if isinstance(hydrogen, bool) or not isinstance(hydrogen, int) or hydrogen < 0:
            raise ValueError("hydrogenDelta must be a non-negative integer.")
        if not isinstance(neutral_loss, bool):
            raise ValueError("enableNeutralLoss must be true or false.")
        formula_only = value.get("intensityCoverageFormulaOnly", True)
        if not isinstance(formula_only, bool):
            raise ValueError("intensityCoverageFormulaOnly must be true or false.")
        return cls(
            max_formula_candidates=None if maximum in (None, "") else int(maximum),
            timeout_sec=None if timeout in (None, "") else float(timeout),
            hydrogen_delta=hydrogen,
            enable_neutral_loss=neutral_loss,
            intensity_coverage_formula_only=formula_only,
        )

    def to_dict(self) -> dict:
        return {
            "maxFormulaCandidates": self.max_formula_candidates,
            "timeoutSec": self.timeout_sec,
            "hydrogenDelta": self.hydrogen_delta,
            "enableNeutralLoss": self.enable_neutral_loss,
            # Always explicit: results that predate this key used all peaks.
            "intensityCoverageFormulaOnly": self.intensity_coverage_formula_only,
        }


@dataclass
class SampleFormula:
    status: str
    precursor_formula: str = ""
    formulas: list[str] | None = None  # comma-joined matches per peak
    message: str = ""


@dataclass
class _FormulaTask:
    precursor_formula: str
    samples: list[tuple[int, str, list[float]]]  # (sample_id, smiles, mz)


def _failed(task: _FormulaTask, status: str, message: str) -> dict[int, dict]:
    return {
        sample_id: asdict(SampleFormula(status, task.precursor_formula, None, message))
        for sample_id, _, _ in task.samples
    }


def _run_task(
    task: _FormulaTask, tolerance_text: str, options: FormulaAssignmentOptions
) -> dict[int, dict]:
    from fredis.libs.mmkit.mmkit import Compound, Formula

    tolerance = parse_mass_tolerance(tolerance_text)
    maximum = options.max_formula_candidates
    try:
        precursor = Formula.parse(task.precursor_formula)
        estimated = estimate_subformula_candidate_count(
            precursor, hydrogen_delta=options.hydrogen_delta
        )
        if maximum is not None and estimated > maximum:
            return _failed(
                task,
                "TooManyFormulaCandidates",
                f"estimated={estimated}, max={maximum}",
            )
        candidates = get_possible_sub_formulas(
            precursor,
            hydrogen_delta=options.hydrogen_delta,
            timeout=options.timeout_sec if options.timeout_sec else math.inf,
        )
    except TimeoutError:
        return _failed(
            task, "TimeoutGeneratingFormulas", f"timeout_sec={options.timeout_sec}"
        )
    except Exception as error:
        return _failed(task, "ErrorGeneratingFormulas", str(error))
    if maximum is not None and len(candidates) > maximum:
        return _failed(
            task,
            "TooManyGeneratedFormulaCandidates",
            f"generated={len(candidates)}, max={maximum}",
        )
    results = {}
    for sample_id, smiles, mz in task.samples:
        try:
            original = (
                Compound.from_smiles(smiles).formula.plain
                if options.enable_neutral_loss
                else None
            )
            matches = assign_formulas_to_peaks(
                peaks_mz=mz,
                formula_candidates=candidates,
                mass_tolerance=tolerance,
                original_formula=original,
            )
            results[sample_id] = asdict(
                SampleFormula(
                    ASSIGNED,
                    task.precursor_formula,
                    [",".join(match["matched_formulas"]) for match in matches],
                )
            )
        except Exception as error:
            results[sample_id] = asdict(
                SampleFormula("ErrorAssigningFormula", task.precursor_formula, None, str(error))
            )
    return results


def _run_chunk(
    chunk: list[_FormulaTask], tolerance_text: str, options: FormulaAssignmentOptions
) -> dict[int, dict]:
    results = {}
    for task in chunk:
        results.update(_run_task(task, tolerance_text, options))
    return results


def assign_sample_formulas(
    groups: list[CompoundGroup],
    mass_tolerance: MassTolerance,
    options: FormulaAssignmentOptions,
    *,
    num_workers: int = 1,
    chunk_size: int = 1,
) -> dict[int, SampleFormula]:
    """``{sample_id: SampleFormula}`` for every sample of ``groups``.

    ``chunk_size`` counts precursor formula groups per worker chunk.
    """
    results: dict[int, SampleFormula] = {}
    by_formula: dict[str, list[tuple[int, str, list[float]]]] = defaultdict(list)
    for group in groups:
        for sample in group.samples:
            try:
                label = precursor_grouping_formula(
                    sample.smiles, sample.adduct, sample.precursor_mz, mass_tolerance
                )
            except PrecursorMZMismatch as error:
                results[sample.sample_id] = SampleFormula(
                    "PrecursorMZMismatch", message=str(error)
                )
                continue
            except Exception as error:
                results[sample.sample_id] = SampleFormula(
                    "ErrorCalculatingPrecursorFormula", message=str(error)
                )
                continue
            by_formula[label].append(
                (sample.sample_id, sample.smiles, [float(mz) for mz in sample.mz])
            )
    tasks = [_FormulaTask(label, by_formula[label]) for label in sorted(by_formula)]
    chunks = [
        tasks[start : start + max(1, chunk_size)]
        for start in range(0, len(tasks), max(1, chunk_size))
    ]
    tolerance_text = format_mass_tolerance(mass_tolerance)

    def collect(values: dict[int, dict]) -> None:
        for sample_id, value in values.items():
            results[int(sample_id)] = SampleFormula(**value)

    with tqdm(
        total=len(tasks), desc="Peak formulas", unit="formula", file=sys.stderr
    ) as progress:
        if num_workers <= 1 or len(chunks) < 2:
            for chunk in chunks:
                collect(_run_chunk(chunk, tolerance_text, options))
                progress.update(len(chunk))
        else:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = {
                    pool.submit(_run_chunk, chunk, tolerance_text, options): chunk
                    for chunk in chunks
                }
                # Completion order: one slow precursor formula must not freeze the bar.
                for future in as_completed(futures):
                    collect(future.result())
                    progress.update(len(futures[future]))
    return results
