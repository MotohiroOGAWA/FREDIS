from __future__ import annotations

import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Generator, List, Optional

from tqdm import tqdm


def generate_task_arguments(
    input_files: List[str],
    output_files: List[str],
    input_flag: str = "-i",
    output_flag: str = "-o",
) -> Generator[List[str], None, None]:
    """Generate arguments for subprocess tasks.

    Parameters
    ----------
    input_files:
        Input file paths.

    output_files:
        Output file paths.

    input_flag:
        Command-line flag for input.

    output_flag:
        Command-line flag for output.

    Yields
    ------
    List[str]
        Argument list for one subprocess.
    """

    for in_file, out_file in zip(input_files, output_files):
        yield [input_flag, in_file, output_flag, out_file]


def to_module_name(path: str) -> str:
    """Convert file path to module name if needed."""

    if path.endswith(".py"):
        path = os.path.splitext(path)[0]
        path = path.replace(os.sep, ".")

    return path


def run_in_subprocess(
    commands: List[str],
    print_output: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> None:
    """Run a single task in a subprocess.

    Parameters
    ----------
    commands:
        Full command list to execute.

    print_output:
        If True, show subprocess stdout/stderr.
        If False, suppress stdout/stderr.
    """

    if print_output:
        subprocess.run(
            commands,
            check=True,
            env=env,
        )
    else:
        result = subprocess.run(
            commands,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(
                "Subprocess failed: " + " ".join(commands),
                file=sys.stderr,
            )
            if result.stdout:
                print(result.stdout, file=sys.stderr, end="")
            if result.stderr:
                print(result.stderr, file=sys.stderr, end="")
            raise subprocess.CalledProcessError(
                result.returncode,
                commands,
                output=result.stdout,
                stderr=result.stderr,
            )


def run_parallel_subprocesses(
    commands_list: List[List[str]],
    max_workers: int = 4,
    print_output: bool = False,
    env: Optional[Dict[str, str]] = None,
    desc: str = "Parallel tasks",
    unit: str = "it",
    on_complete: Optional[Callable[[List[str]], None]] = None,
    progress: Optional[tqdm] = None,
    on_error: Optional[Callable[[List[str], subprocess.CalledProcessError], None]] = None,
) -> None:
    """Run multiple subprocesses in parallel.

    Parameters
    ----------
    commands_list:
        List of full subprocess commands.

    max_workers:
        Number of parallel workers.

    print_output:
        If True, show subprocess stdout/stderr.
        If False, suppress stdout/stderr.
    on_complete:
        Optional callback receiving the command after each successful subprocess,
        invoked in the calling thread in completion order.
    on_error:
        Optional callback receiving the command and its CalledProcessError
        when a subprocess exits non-zero, invoked in the calling thread.
        When given, a failed subprocess no longer aborts the remaining
        ones; when omitted, the first failure is raised as before.
    progress:
        Optional externally-owned tqdm bar to advance (one update per
        completed command) instead of creating a new one. Lets a caller that
        invokes this function repeatedly, in bounded waves of commands_list,
        show one continuous bar across every wave rather than a bar that
        resets to 0 each call.
    """


    # Optional parent-side notification preserves subprocess isolation.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_in_subprocess,
                commands,
                print_output,
                env,
            ): commands
            for commands in commands_list
        }

        completed = as_completed(futures) if progress is not None else tqdm(
            as_completed(futures),
            total=len(futures),
            desc=desc,
            unit=unit,
        )
        for future in completed:
            try:
                future.result()
            except subprocess.CalledProcessError as error:
                if on_error is None:
                    raise
                if progress is not None:
                    progress.update(1)
                on_error(futures[future], error)
                continue
            if progress is not None:
                progress.update(1)
            if on_complete is not None:
                on_complete(futures[future])
