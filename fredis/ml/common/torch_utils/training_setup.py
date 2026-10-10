import os
from typing import Dict, Tuple, Union, Optional
import torch
from .optim_sheduler.MovingWindowReduceLROnPlateau import MovingWindowReduceLROnPlateau

def get_optimizer(model: torch.nn.Module, optimizer_info: dict, is_return_scheduler: bool = False) -> Union[torch.optim.Optimizer, Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]]:
    """
    Create and return a PyTorch optimizer based on the given configuration.

    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        optimizer_info (dict): Dictionary containing optimizer configuration.
            Must include:
                - "name" (str): Name of the optimizer. Supported options are:
                    ["sgd", "adagrad", "rmsprop", "adadelta", "adam", "adamw"].
                - "lr" (float): Learning rate.
            May include (depending on optimizer):
                - "momentum" (float): Momentum factor (used in SGD, RMSprop).
                - "eps" (float): Term added to denominator for numerical stability.
        if is_return_scheduler:
            scheduler = get_scheduler(optimizer, optimizer_info.get('scheduler', None))
            return optimizer, scheduler

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Raises:
        SystemExit: If the optimizer name is not recognized.

    Example:
        >>> model = MyModel()
        >>> optimizer_info = {
        ...     "name": "adam",
        ...     "lr": 1e-3,
        ...     "eps": 1e-8
        ... }
        >>> optimizer = get_optimizer(model, optimizer_info)
    """
    optimizer_info = optimizer_info.copy()
    name = optimizer_info.pop('name')
    scheduler_info = optimizer_info.pop('scheduler', None)
    if name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_info['lr'], momentum=optimizer_info['momentum'])
    elif name.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=optimizer_info['lr'], eps=optimizer_info['eps'])
    elif name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=optimizer_info['lr'], momentum=optimizer_info['momentum'], eps=optimizer_info['eps'])
    elif name.lower() == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=optimizer_info['lr'], eps=optimizer_info['eps'])
    elif name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_info)
    elif name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_info)
    else:
        print("Error: optimizer not recognized")
        quit()

    if not is_return_scheduler:
        return optimizer
    else:
        scheduler = get_scheduler(optimizer, scheduler_info)
        return optimizer, scheduler

def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_info: dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create and return a PyTorch learning rate scheduler based on the given configuration.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be scheduled.
        scheduler_info (dict): Dictionary containing scheduler configuration.
            Must include:
                - "name" (str): Name of the scheduler. Supported options are:
                    ["steplr", "multisteplr", "exponentiallr", "cosineannealinglr", 
                     "reducelronplateau", "onecyclelr"]
            Additional keys depend on the scheduler type:
                - StepLR: step_size (int), gamma (float)
                - MultiStepLR: milestones (list of int), gamma (float)
                - ExponentialLR: gamma (float)
                - CosineAnnealingLR: T_max (int), eta_min (float, optional)
                - ReduceLROnPlateau: mode (str), factor (float), patience (int)
                - OneCycleLR: max_lr (float), epochs (int), steps_per_epoch (int)

    Returns:
        torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau

    Raises:
        ValueError: If the scheduler name is not recognized.
    """
    if scheduler_info is None:
        return None
    if scheduler_info['name'] is None:
        return None

    scheduler_info = scheduler_info.copy()
    name = scheduler_info.pop('name').lower()
    scheduler_info['optimizer'] = optimizer

    if name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_info.get('step_size', 10),
            gamma=scheduler_info.get('gamma', 0.1)
        )
    elif name == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_info.get('milestones', [30, 80]),
            gamma=scheduler_info.get('gamma', 0.1)
        )
    elif name == 'exponentiallr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_info.get('gamma', 0.95)
        )
    elif name == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_info.get('T_max', 50),
            eta_min=scheduler_info.get('eta_min', 0.0)
        )
    elif name == 'reducelronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_info.get('mode', 'min'),
            factor=scheduler_info.get('factor', 0.1),
            patience=scheduler_info.get('patience', 10)
        )
    elif name == 'onecyclelr':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_info['max_lr'],
            epochs=scheduler_info['epochs'],
            steps_per_epoch=scheduler_info['steps_per_epoch']
        )
    elif name == 'movingwindowreducelronplateau':
        scheduler = MovingWindowReduceLROnPlateau(**scheduler_info)
    else:
        raise ValueError(f"Error: scheduler '{name}' not recognized")

    return scheduler


def get_criterion(name: str):
    name = name.lower()
    if name == 'crossentropy': # CrossEntropy
        return torch.nn.CrossEntropyLoss()
    elif name == 'bce': # Binary Cross Entropy
        return torch.nn.BCELoss()
    elif name == 'bcewithlogits': # Binary Cross Entropy with Logits
        return torch.nn.BCEWithLogitsLoss()
    elif name == 'mse': # Mean Squared Error
        return torch.nn.MSELoss()
    elif name == 'l1': # L1 Loss
        return torch.nn.L1Loss()
    elif name == 'smoothl1': # Smooth L1 Loss
        return torch.nn.SmoothL1Loss()
    elif name == 'hinge': # Hinge Loss
        return torch.nn.HingeEmbeddingLoss()
    elif name == 'kldiv': # KL Divergence
        return torch.nn.KLDivLoss()
    elif name == 'nll': # Negative Log Likelihood
        return torch.nn.NLLLoss()
    elif name == 'poissonnll': # Poisson Negative Log Likelihood
        return torch.nn.PoissonNLLLoss()
    elif name == 'cosineembedding': # Cosine Embedding
        return torch.nn.CosineEmbeddingLoss()
    elif name == 'huber': # Huber Loss
        return torch.nn.HuberLoss()
    elif name == 'multilabelmargin': # Multi Label Margin
        return torch.nn.MultiLabelMarginLoss()
    elif name == 'multilabelsoftmargin': # Multi Label Soft Margin
        return torch.nn.MultiLabelSoftMarginLoss()
    elif name == 'multimargin': # Multi Margin
        return torch.nn.MultiMarginLoss()
    elif name == 'marginranking': # Margin Ranking
        return torch.nn.MarginRankingLoss()
    elif name == 'ctc': # Connectionist Temporal Classification
        return torch.nn.CTCLoss()
    else:
        print("Error: loss function not recognized")
        quit()