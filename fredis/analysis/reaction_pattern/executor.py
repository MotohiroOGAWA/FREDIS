"""Run chemistry task queues serially or in a process pool."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import multiprocessing
import sys
from typing import Callable, Iterable

from tqdm import tqdm

from . import chemistry
from .constants import FAILED, SKIP_WORKER_CRASH


def _crash_result(task: chemistry.ChemistryTask, message: str) -> dict:
    return {"compound_id": task.compound.compound_id, "steps": [
        dict(compound_id=task.compound.compound_id, condition_id=step.condition_id,
             parent_condition_id=step.parent_condition_id, status=FAILED, skip_category=SKIP_WORKER_CRASH,
             reason=message, error="", kept=[], counts=[], file="", incremental=False, search_stats=None,
             seconds=0.0) for step in task.steps]}


def _chunks(tasks: list, size: int) -> list[list]:
    return [tasks[start:start + size] for start in range(0, len(tasks), size)]


def execute(tasks: Iterable[chemistry.ChemistryTask], options: chemistry.ChemistryOptions, *,
            num_workers: int = 1, chunk_size: int = 1, description: str = "Chemistry",
            on_result: Callable[[dict], None]) -> None:
    """Run every task once; ``on_result`` receives results in completion order.

    A Python exception is already classified inside ``chemistry.run_task``.
    If a worker process itself dies (native crash or OOM), the tasks of the
    affected chunks are retried one per fresh process so that only the
    offending task is recorded as a worker crash.
    """
    tasks = list(tasks)
    if not tasks:
        return
    with tqdm(total=len(tasks), desc=description, unit="task", file=sys.stderr) as progress:
        def consume(result: dict) -> None:
            on_result(result)
            progress.update(1)

        if num_workers <= 1 or len(tasks) < 2:
            chemistry.initialize(options)
            for task in tasks:
                consume(chemistry.run_task(task))
            return
        context = multiprocessing.get_context("spawn")
        chunk_size = max(1, min(chunk_size, len(tasks) // num_workers or 1))
        pending_chunks = _chunks(tasks, chunk_size)
        crashed: list[list] = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=context,
                                 initializer=chemistry.initialize, initargs=(options,)) as pool:
            in_flight = {}
            window = num_workers * 2
            chunk_iter = iter(pending_chunks)
            broken = False

            def submit_next() -> bool:
                chunk = next(chunk_iter, None)
                if chunk is None:
                    return False
                try:
                    in_flight[pool.submit(chemistry.run_tasks, chunk)] = chunk
                except BrokenProcessPool:
                    crashed.append(chunk)
                    crashed.extend(chunk_iter)
                    return False
                return True

            for _ in range(window):
                if not submit_next():
                    break
            while in_flight:
                done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for future in done:
                    chunk = in_flight.pop(future)
                    try:
                        results = future.result()
                    except BrokenProcessPool:
                        broken = True
                        crashed.append(chunk)
                        continue
                    for result in results:
                        consume(result)
                    if not broken:
                        submit_next()
            if broken:
                crashed.extend(chunk_iter)
        for task in (task for chunk in crashed for task in chunk):
            try:
                with ProcessPoolExecutor(max_workers=1, mp_context=context,
                                         initializer=chemistry.initialize, initargs=(options,)) as single:
                    consume(single.submit(chemistry.run_task, task).result())
            except BrokenProcessPool:
                consume(_crash_result(task, "Worker process exited unexpectedly."))
