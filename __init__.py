"""
traincore_dl
------------
A lightweight modular deep learning training core
for unified CNN, GNN, and hybrid model pipelines.

Modules:
- Trainer          : Core training controller (loop, early stop, metrics)
- Data utilities   : set_seed, normalize_data, inspect_dataset
- Scheduler        : build_scheduler for step decay or warmup
- Logger utilities : TrainingLogger (summary, loss, model save, plot)
"""

from .trainer_core import Trainer
from .data_utils import set_seed, normalize_data, inspect_dataset
from .scheduler_utils import build_scheduler
from .logger_utils import TrainingLogger

__all__ = [
    "Trainer",
    "set_seed",
    "normalize_data",
    "inspect_dataset",
    "build_scheduler",
    "TrainingLogger",
]
