"""
Core module for toxicity detection system.
Contains training and model management functionality.
"""

from .training import evaluate_model, load_training_data, train_toxicity_model

__all__ = [
    "train_toxicity_model",
    "load_training_data",
    "evaluate_model",
]
