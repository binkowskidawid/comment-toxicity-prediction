"""
Toxicity Detection System

A machine learning-based system for detecting toxicity in online comments.
Provides training, testing, and interactive analysis capabilities.
"""

from .config import LABELS, MODEL_FILE, VECTORIZER_FILE
from .utils.model_utils import load_model_and_vectorizer, models_exist
from .utils.text_processing import format_comment_results, get_comment_rating

__version__ = "0.1.0"
__author__ = "ML Course Project"

__all__ = [
    "LABELS",
    "MODEL_FILE",
    "VECTORIZER_FILE",
    "load_model_and_vectorizer",
    "models_exist",
    "get_comment_rating",
    "format_comment_results",
]
