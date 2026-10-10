from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from tqdm import tqdm


TQDM_NCOLS = 108
TQDM_BAR_FORMAT = (
    "{desc:<18.18} {percentage:3.0f}%|{bar:30}| "
    "{n_fmt:>6}/{total_fmt:<6} [{elapsed}<{remaining}, {rate_fmt}]"
)

_edge_bar: ContextVar[Optional[tqdm]] = ContextVar("fragment_edge_bar", default=None)
_edge_bar_base_desc: ContextVar[Optional[str]] = ContextVar("fragment_edge_bar_desc", default=None)


def fixed_tqdm(*args, position: int, leave: bool, **kwargs) -> tqdm:
    """Construct a fixed-width bar that never follows terminal resizing."""
    kwargs.setdefault("ncols", TQDM_NCOLS)
    kwargs.setdefault("dynamic_ncols", False)
    kwargs.setdefault("position", int(position))
    kwargs.setdefault("leave", bool(leave))
    kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)
    kwargs.setdefault("mininterval", 0.1)
    return tqdm(*args, **kwargs)


@contextmanager
def iteration_edge_progress(total: int, *, desc: str = "edges") -> Iterator[None]:
    bar = fixed_tqdm(
        total=max(int(total), 0),
        desc=desc,
        position=2,
        leave=False,
    )
    token = _edge_bar.set(bar)
    desc_token = _edge_bar_base_desc.set(desc)
    try:
        yield
    finally:
        _edge_bar_base_desc.reset(desc_token)
        _edge_bar.reset(token)
        bar.close()


def advance_edge_progress(count: int = 1) -> None:
    bar = _edge_bar.get()
    if bar is not None:
        remaining = max(int(bar.total or 0) - int(bar.n), 0)
        bar.update(min(max(int(count), 0), remaining))
        # The surrounding optimization step continues with candidate heads,
        # losses, backward, and optimizer work. Close this bar as soon as its
        # actual edge-attention phase is complete so that those later phases
        # do not look like a stalled edge calculation.
        if int(bar.n) >= int(bar.total or 0):
            bar.close()


def set_edge_progress_total(total: int) -> None:
    """Replace a provisional edge total after the bounded set is selected."""
    bar = _edge_bar.get()
    if bar is not None:
        bar.total = max(int(total), 0)
        bar.refresh()


def set_edge_progress_phase(label: str) -> None:
    """Update the current edge bar's description with a coarse phase label.

    Cheap (a string update plus tqdm's own refresh); does not touch the
    bar's total/count. Used to make otherwise-silent work (device transfer,
    the frozen MolEncoder pass, base-logit computation, training-edge
    sampling, tree-importance pre-scoring) visible before the per-edge
    attention loop starts ticking.
    """
    bar = _edge_bar.get()
    if bar is not None:
        base = _edge_bar_base_desc.get() or "edges"
        bar.set_description_str(f"{base} [{label}]")
        bar.refresh()
