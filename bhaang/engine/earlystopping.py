import os
import torch
from torch.nn import Module

class EarlyStopping:
    """
    Early stops training if a monitored metric doesn't improve after a given patience.

    Args:
        mode (str): One of {'min', 'max'}. 'min' for metrics to decrease (e.g., loss), 'max' for metrics to increase (e.g., accuracy).
        patience (int): How many epochs to wait after last improvement before stopping.
        verbose (bool): If True, prints messages on improvement and counter.
        delta (float): Minimum change to qualify as improvement.
        path (str): File path for saving checkpoints.
        trace_func (callable): Logging function (default: print).
    """
    def __init__(
        self,
        mode: str = 'min',
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0.0005,
        path: str = 'checkpoint.pt',
        trace_func=print
    ):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        self.counter = 0
        self.early_stop = False
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.last_score = None

        # ensure directory exists
        dirpath = os.path.dirname(self.path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

    def __call__(
        self,
        metric: float,
        model: Module,
        optimizer: torch.optim.Optimizer = None,
        epoch: int = None,
        **kwargs
    ) -> None:
        """
        Invoke after each validation. Optionally pass optimizer and epoch for checkpointing.

        Args:
            metric (float): Current metric value.
            model (Module): Model to save on improvement.
            optimizer (Optimizer, optional): Optimizer to save state.
            epoch (int, optional): Current epoch number.
            **kwargs: Any other state to include in the checkpoint (e.g., lr_scheduler state).
        """
        improved = (
            (self.mode == 'min' and metric < self.best_score - self.delta) or
            (self.mode == 'max' and metric > self.best_score + self.delta)
        )

        if improved:
            if self.verbose:
                if self.last_score is None:
                    self.trace_func(f"First save: {metric:.6f}. Saving checkpoint to {self.path}...")
                else:
                    change = 'decreased' if self.mode == 'min' else 'increased'
                    self.trace_func(
                        f"Metric {change} ({self.best_score:.6f} -> {metric:.6f}). "
                        f"Saving checkpoint to {self.path}..."
                    )
            self.best_score = metric
            self.save_checkpoint(model, optimizer, epoch, **kwargs)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        self.last_score = metric

    def save_checkpoint(
        self,
        model: Module,
        optimizer: torch.optim.Optimizer = None,
        epoch: int = None,
        **kwargs
    ) -> None:
        """
        Saves model, optimizer, and additional state dict to file.
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_score': self.best_score,
            'counter': self.counter
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        # include any other state
        for k, v in kwargs.items():
            checkpoint[k] = v

        torch.save(checkpoint, self.path)

    @staticmethod
    def load_checkpoint(
        path: str,
        model: Module,
        optimizer: torch.optim.Optimizer = None,
        map_location=None
    ) -> dict:
        """
        Loads checkpoint from file.

        Args:
            path (str): Path to checkpoint file.
            model (Module): Model instance to load state into.
            optimizer (Optimizer, optional): Optimizer to load state into.
            map_location: torch.load map_location parameter.

        Returns:
            dict: The loaded checkpoint dictionary.
        """
        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

    def reset(self) -> None:
        """
        Resets the early stopping state for a fresh training run.
        """
        self.counter = 0
        self.early_stop = False
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.last_score = None
