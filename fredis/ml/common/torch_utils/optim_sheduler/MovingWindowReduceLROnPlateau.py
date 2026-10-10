from collections import deque
import torch

class MovingWindowReduceLROnPlateau(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, mode="min", window_size=5, factor=0.1, patience=10, min_lr=1e-4, min_delta=0.0, reset_step=0.0, verbose=False):
        """
        Custom LR scheduler that reduces learning rate if no improvement
        is observed in the last `window_size` metrics.

        Args:
            optimizer: PyTorch optimizer
            mode: "min" (for loss) or "max" (for accuracy/score)
            window_size: number of past metrics to consider
            factor: LR reduction factor
            patience: number of bad epochs before reducing LR
            min_lr: minimum allowed LR
            min_delta: minimum change in the monitored quantity to qualify as an improvement
            verbose: whether to print logs
        """
        self.mode = mode
        self.window_size = window_size
        self.factor = factor
        self._patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.reset_step = int(reset_step * patience)
        self.verbose = verbose

        # keep track of recent metrics
        self.history = deque(maxlen=window_size)
        # count consecutive bad epochs
        self.num_bad_epochs = 0

        super().__init__(optimizer)

    def step(self, metric=None, epoch=None):
        """
        Update scheduler with the latest metric.
        If no improvement compared to the recent window, reduce LR.
        """
        if metric is None:
            # fallback (e.g., manual stepping like StepLR)
            # return super().step(epoch)
            return


        # wait until we have enough history
        if len(self.history) + 1 < self.history.maxlen:
            self.history.append(metric)
            return

        # reference value: best in the window
        if self.mode == "min":
            ref = max(self.history)  # exclude current
            improve = (metric < ref - self.min_delta)
        else:
            ref = min(self.history)  # exclude current
            improve = (metric > ref + self.min_delta)

        if improve:
            # reset patience counter on improvement
            self.num_bad_epochs = max(0, self.num_bad_epochs - self.reset_step)
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self._patience:
                # reduce LR
                for i, group in enumerate(self.optimizer.param_groups):
                    old_lr = group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    if old_lr - new_lr > 1e-8 and self.verbose:
                        print(f"Reduce LR: {old_lr:.2e} → {new_lr:.2e}")
                    group['lr'] = new_lr
                # reset patience counter
                self.num_bad_epochs = 0
            
        self.history.append(metric)

    @property
    def counter(self):
        return self.num_bad_epochs
    
    @property
    def patience(self):
        return self._patience