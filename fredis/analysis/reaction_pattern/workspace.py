"""Output-directory and pattern-selection helpers for a new analysis run."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable


@contextmanager
def _activity(label: str, *, announce: bool = True):
    """Report one start/end pair without interrupting item progress bars."""
    started = time.monotonic()
    if announce:
        print(f"{label}: started", file=sys.stderr, flush=True)
    try:
        yield
    except BaseException:
        if announce:
            print(
                f"{label}: failed ({time.monotonic() - started:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
        raise
    else:
        if announce:
            print(
                f"{label}: done ({time.monotonic() - started:.1f}s)",
                file=sys.stderr,
                flush=True,
            )


def initialize_output_directory(
    output: Path, *, overwrite: bool, protected_paths: Iterable[str | Path] = ()
) -> bool:
    """Confirm an implicit overwrite, then replace the complete directory."""
    output = output.resolve()
    if output.exists() and not output.is_dir():
        raise FileExistsError(
            f"Output path already exists and is not a directory: {output}"
        )
    has_content = output.exists() and any(output.iterdir())
    if has_content and not overwrite:
        message = f"Output directory already exists and is not empty: {output}"
        if not sys.stdin.isatty():
            raise FileExistsError(
                message + ". Re-run with --overwrite 1 or choose another directory."
            )
        answer = (
            input(message + "\nDelete the entire directory and overwrite it? [y/N] ")
            .strip()
            .lower()
        )
        if answer not in ("y", "yes"):
            raise FileExistsError(
                "Cancelled because the output directory already exists."
            )
        overwrite = True
    if overwrite and output.exists():
        for protected in [
            Path(__file__).resolve(),
            *(Path(value).resolve() for value in protected_paths),
        ]:
            if protected == output or protected.is_relative_to(output):
                raise ValueError(
                    "Output directory contains an input or application file."
                )
        with _activity("Remove previous output directory"):
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return overwrite
