# traincore_dl/__init__.py
"""
traincore_dl
------------
A lightweight modular deep learning training core
for unified CNN, GNN, and hybrid model pipelines.
"""

from .trainer_core import Trainer
from .data_utils import set_seed, normalize_data, inspect_dataset
from .scheduler_utils import build_scheduler
from .logger_utils import log_summary, plot_loss