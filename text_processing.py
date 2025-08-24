"""
Text processing utilities for toxicity detection system.
Handles TF-IDF vectorization and comment rating analysis.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any, List
from config import MAX_FEATURES, LABELS


def create_vectorizer() -> TfidfVectorizer:
    """
    Create and configure TF-IDF vectorizer for text processing.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) converts text to numerical vectors:
    - TF: How often a word appears in a document
    - IDF: How rare a word is across all documents
    - Effect: Important words get higher values, common words get lower values
    
    Returns:
        TfidfVectorizer: Configured vectorizer ready for training
    """
    return TfidfVectorizer(max_features=MAX_FEATURES)


def get_comment_rating(comment: str, model: Any, vectorizer: Any) -> np.ndarray:
    """
    Analyze single comment and return predicted toxicity levels.
    
    Args:
        comment (str): Text comment to analyze
        model: Trained machine learning model
        vectorizer: Trained TF-IDF vectorizer
        
    Returns:
        np.ndarray: Array of toxicity scores for each label (7 values)
                   Values range from 0 (not toxic) to 1 (highly toxic)
    """
    # Transform comment to TF-IDF vector using trained vectorizer
    comment_tfidf = vectorizer.transform([comment])
    
    # Predict toxicity levels using trained model
    # Return first (and only) result - array with 7 values for 7 labels
    return model.predict(comment_tfidf)[0]


def format_comment_results(comment: str, results: np.ndarray) -> str:
    """
    Format comment analysis results for display.
    
    Args:
        comment (str): Original comment text
        results (np.ndarray): Toxicity prediction results
        
    Returns:
        str: Formatted string with results breakdown
    """
    output = []
    output.append(f"Comment: '{comment}'")
    output.append(f"Overall toxicity: {results[0]:.3f}")
    output.append("\nDetailed breakdown:")
    
    for i, label in enumerate(LABELS):
        toxicity_level = "HIGH" if results[i] > 0.5 else "MEDIUM" if results[i] > 0.2 else "LOW"
        output.append(f"  {label:>16}: {results[i]:.3f} ({toxicity_level})")
    
    return "\n".join(output)


def batch_analyze_comments(comments: List[str], model: Any, vectorizer: Any) -> List[np.ndarray]:
    """
    Analyze multiple comments at once for efficiency.
    
    Args:
        comments (List[str]): List of comments to analyze
        model: Trained machine learning model
        vectorizer: Trained TF-IDF vectorizer
        
    Returns:
        List[np.ndarray]: List of toxicity prediction arrays, one per comment
    """
    if not comments:
        return []
    
    # Transform all comments to TF-IDF vectors at once
    comments_tfidf = vectorizer.transform(comments)
    
    # Predict toxicity for all comments
    return model.predict(comments_tfidf)


def get_labels_info() -> dict:
    """
    Get information about toxicity labels.
    
    Returns:
        dict: Information about each toxicity label
    """
    labels_info = {
        "toxicity": "General toxicity level",
        "severe_toxicity": "Severe toxicity level", 
        "obscene": "Obscene language",
        "threat": "Threats and intimidation",
        "insult": "Insults and personal attacks",
        "identity_attack": "Identity-based attacks",
        "sexual_explicit": "Sexually explicit content"
    }
    
    return {
        "labels": LABELS,
        "descriptions": labels_info,
        "count": len(LABELS)
    }