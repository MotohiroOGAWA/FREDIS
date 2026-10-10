from __future__ import annotations

from collections import deque
from typing import Deque, Optional


class EarlyStopping:
    """Windowed early stopping helper for minimizing or maximizing a score."""

    def __init__(
        self,
        patience: Optional[int] = None,
        window_size: int = 1,
        min_delta: float = 0.0,
        reset_step: float = 1.0,
        mode: str = "min",
        verbose: bool = False,
    ) -> None:
        self._enable = patience is not None and patience > 0
        if not self._enable:
            patience = float("inf")

        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")

        self._patience = patience
        self._window_size = int(window_size)
        self._verbose = bool(verbose)
        self._counter = 0
        self._min_delta = float(min_delta)
        self._reset_step = max(1, int(float(patience) * float(reset_step))) if self._enable else 0
        self._mode = mode
        self._history: Deque[float] = deque(maxlen=self._window_size)

    def __call__(self, score: float) -> None:
        score = float(score)
        if not self._enable:
            self._history.append(score)
            return

        if len(self._history) < self._history.maxlen:
            self._history.append(score)
            return

        best_score = min(self._history) if self._mode == "min" else max(self._history)
        improved = (
            score < best_score - self._min_delta
            if self._mode == "min"
            else score > best_score + self._min_delta
        )
        if improved:
            self._counter = max(0, self._counter - self._reset_step)
        else:
            self._counter += 1
            if self._verbose:
                print(f"EarlyStopping counter: {self._counter} out of {self._patience}")

        self._history.append(score)

    @property
    def enabled(self) -> bool:
        return self._enable

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def patience(self) -> int | float:
        return self._patience

    @property
    def early_stop(self) -> bool:
        return self._counter >= self._patience

    @staticmethod
    def from_params(params: dict) -> "EarlyStopping":
        return EarlyStopping(**params)
