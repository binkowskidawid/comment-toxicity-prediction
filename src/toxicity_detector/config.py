"""
Configuration file for toxicity detection system.
Contains all constants, file paths, and model parameters.
"""

# Model file paths
MODEL_FILE = "models/model.joblib"
VECTORIZER_FILE = "models/vectorizer.joblib"

# Toxicity labels analyzed by the model
LABELS = [
    "toxicity",  # General toxicity level
    "severe_toxicity",  # Severe toxicity level
    "obscene",  # Obscene language
    "threat",  # Threats and intimidation
    "insult",  # Insults and personal attacks
    "identity_attack",  # Identity-based attacks
    "sexual_explicit",  # Sexually explicit content
]

# Training parameters
MAX_FEATURES = 5000  # Maximum number of TF-IDF features
TEST_SIZE = 0.2  # Proportion of data for testing (20%)
RANDOM_STATE = 42  # Random seed for reproducible results

# Dataset configuration
DATASET_NAME = "google/civil_comments"
DATASET_SPLIT = "train"

# Model parameters
MODEL_TYPE = "LinearRegression"  # Type of model to use

# Display configuration
SEPARATOR_LENGTH = 60
TEST_SEPARATOR_LENGTH = 40
