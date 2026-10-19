"""Track the best validation loss and consecutive non-improving epochs."""

from dataclasses import dataclass
import math


@dataclass
class EarlyStopping:
    """Minimize validation loss while retaining resumable progress.

    An improvement must exceed ``min_delta`` relative to the global best loss.
    ``patience=0`` disables stopping, but continues to update the tracked state.
    Restore ``best`` and ``bad_epochs`` together when resuming training.
    """

    patience: int = 10
    min_delta: float = 1e-6
    best: float = math.inf
    bad_epochs: int = 0

    def __post_init__(self):
        if type(self.patience) is not int or self.patience < 0:
            raise ValueError("patience must be a non-negative integer")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")
        if type(self.bad_epochs) is not int or self.bad_epochs < 0:
            raise ValueError("bad_epochs must be a non-negative integer")

    def update(self, loss: float) -> bool:
        """Record an epoch; return whether it establishes a new best loss."""
        improved = loss < self.best - self.min_delta
        if improved:
            self.best = float(loss)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        """A patience of zero disables stopping, while still tracking the best."""
        return self.patience > 0 and self.bad_epochs >= self.patience
