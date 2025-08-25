"""
Command line interface module for toxicity detection system.
Contains main analyzer and training interfaces.
"""

from .main import main as analyzer_main
from .train import main as train_main

__all__ = [
    "analyzer_main",
    "train_main",
]
