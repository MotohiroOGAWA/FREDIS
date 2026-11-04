from __future__ import annotations

import sys
import subprocess
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from typing import Callable, Dict, List, Optional

from tqdm import tqdm


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
    on_error: Optional[
        Callable[[List[str], subprocess.CalledProcessError], None]
    ] = None,
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

        completed = (
            as_completed(futures)
            if progress is not None
            else tqdm(
                as_completed(futures),
                total=len(futures),
                desc=desc,
                unit=unit,
            )
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


def run_rolling_subprocesses(
    next_command: Callable[[], Optional[List[str]]],
    max_workers: int = 4,
    print_output: bool = False,
    env: Optional[Dict[str, str]] = None,
    on_complete: Optional[Callable[[List[str]], None]] = None,
    on_error: Optional[
        Callable[[List[str], subprocess.CalledProcessError], None]
    ] = None,
) -> None:
    """Keep up to ``max_workers`` subprocesses running until ``next_command`` is exhausted.

    ``next_command()`` is called only when a worker slot is free, so a caller
    that writes each command's input file lazily never has more than
    ``max_workers`` of them at once, and a slot is refilled as soon as any
    subprocess finishes (no waiting for a whole batch).  Callbacks run in the
    calling thread in completion order; ``next_command`` is asked again after
    every callback, so a callback may queue more work (e.g. retries).  Without
    ``on_error`` the first failure is raised after the running ones finish.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        running = {}

        def fill():
            while len(running) < max_workers:
                command = next_command()
                if command is None:
                    return
                running[executor.submit(run_in_subprocess, command, print_output, env)] = command

        fill()
        while running:
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for future in done:
                command = running.pop(future)
                try:
                    future.result()
                except subprocess.CalledProcessError as error:
                    if on_error is None:
                        raise
                    on_error(command, error)
                else:
                    if on_complete is not None:
                        on_complete(command)
            fill()
