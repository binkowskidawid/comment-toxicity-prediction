"""
Model utilities for toxicity detection system.
Handles saving, loading, and checking existence of trained models.
"""

import os.path
import joblib
from typing import Tuple, Any
from config import MODEL_FILE, VECTORIZER_FILE


def models_exist() -> bool:
    """
    Check if saved model files exist on disk.
    
    Returns:
        bool: True if both model and vectorizer files exist, False otherwise
    """
    model_exists = os.path.exists(MODEL_FILE)
    vectorizer_exists = os.path.exists(VECTORIZER_FILE)
    return model_exists and vectorizer_exists


def save_model_and_vectorizer(model: Any, vectorizer: Any) -> None:
    """
    Save trained model and vectorizer to .joblib files.
    
    Args:
        model: Trained machine learning model
        vectorizer: Trained TF-IDF vectorizer
    """
    print(f"\n💾 Saving model and vectorizer to disk...")
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    
    # Save model and vectorizer using joblib (optimized for scikit-learn objects)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    
    print(f"✅ Model saved to: {MODEL_FILE}")
    print(f"✅ Vectorizer saved to: {VECTORIZER_FILE}")
    print("\nNext runs will be significantly faster! ⚡")


def load_model_and_vectorizer() -> Tuple[Any, Any]:
    """
    Load saved model and vectorizer from .joblib files.
    
    Returns:
        tuple: (model, vectorizer) - Loaded model and vectorizer objects
        
    Raises:
        FileNotFoundError: If model files don't exist
    """
    if not models_exist():
        raise FileNotFoundError(
            f"Model files not found. Expected files:\n"
            f"- {MODEL_FILE}\n"
            f"- {VECTORIZER_FILE}\n"
            f"Please train a model first using train_model.py"
        )
    
    print(f"\n📦 Loading saved model from disk...")
    
    # Load model and vectorizer using joblib
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    
    print("✅ Model and vectorizer loaded successfully!")
    print("⚡ Skipped training - using pre-trained model!")
    
    return model, vectorizer


def get_model_info() -> dict:
    """
    Get information about saved models.
    
    Returns:
        dict: Information about model files including existence and sizes
    """
    info = {
        'model_exists': os.path.exists(MODEL_FILE),
        'vectorizer_exists': os.path.exists(VECTORIZER_FILE),
        'model_file': MODEL_FILE,
        'vectorizer_file': VECTORIZER_FILE
    }
    
    if info['model_exists']:
        info['model_size_mb'] = round(os.path.getsize(MODEL_FILE) / (1024 * 1024), 2)
    
    if info['vectorizer_exists']:
        info['vectorizer_size_mb'] = round(os.path.getsize(VECTORIZER_FILE) / (1024 * 1024), 2)
    
    return info