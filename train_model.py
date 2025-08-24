"""
Standalone model training script for toxicity detection system.
Loads data, trains model, and saves it for future use.
"""

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from config import (
    DATASET_NAME, DATASET_SPLIT, LABELS, TEST_SIZE, 
    RANDOM_STATE, SEPARATOR_LENGTH
)
from model_utils import save_model_and_vectorizer, models_exist
from text_processing import create_vectorizer


def load_training_data():
    """
    Load and prepare training data from civil_comments dataset.
    
    Returns:
        tuple: (X, y) where X is text data and y is toxicity labels
    """
    print("1️⃣ Loading data from internet...")
    print(f"Dataset: {DATASET_NAME}")
    
    # Load civil_comments dataset from Google - contains real internet comments with expert ratings
    dataset = load_dataset(DATASET_NAME)
    
    # Convert training split to pandas DataFrame for easier manipulation
    df = dataset[DATASET_SPLIT].to_pandas()
    
    print(f"✅ Loaded {len(df)} comments for analysis")
    
    # X = input features (comment texts)
    X = df["text"]
    # y = output targets (toxicity ratings for each category)
    y = df[LABELS]
    
    return X, y


def split_training_data(X, y):
    """
    Split data into training and testing sets.
    
    Args:
        X: Input features (comment texts)
        y: Target labels (toxicity ratings)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("2️⃣ Splitting data into training and testing sets...")
    
    # Split data: 80% for training, 20% for testing
    # random_state ensures reproducible results
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    print(f"✅ Training data: {len(X_train)} comments")
    print(f"✅ Testing data: {len(X_test)} comments")
    
    return X_train, X_test, y_train, y_test


def create_tfidf_vectors(X_train, X_test):
    """
    Convert text to TF-IDF numerical vectors.
    
    Args:
        X_train: Training text data
        X_test: Testing text data
        
    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, vectorizer)
    """
    print("3️⃣ Creating TF-IDF vectorizer...")
    
    # Create and configure TF-IDF vectorizer
    vectorizer = create_vectorizer()
    
    # Fit vectorizer on training data and transform to numerical vectors
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Transform test data using already fitted vectorizer
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("✅ TF-IDF vectorizer trained and data transformed!")
    print(f"✅ Feature dimensions: {X_train_tfidf.shape[1]} features")
    
    return X_train_tfidf, X_test_tfidf, vectorizer


def train_linear_regression_model(X_train_tfidf, y_train):
    """
    Train linear regression model on TF-IDF vectors.
    
    Args:
        X_train_tfidf: Training data as TF-IDF vectors
        y_train: Training toxicity labels
        
    Returns:
        LinearRegression: Trained model
    """
    print("4️⃣ Training linear regression model...")
    
    # Create linear regression model
    # Linear regression finds linear relationship between words and toxicity
    model = LinearRegression()
    
    # Train model on TF-IDF vectors and toxicity labels
    model.fit(X_train_tfidf, y_train)
    
    print("✅ Linear regression model trained successfully!")
    
    return model


def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained machine learning model
        X_test_tfidf: Test data as TF-IDF vectors
        y_test: True toxicity labels for test data
    """
    print("5️⃣ Evaluating model performance...")
    
    # Make predictions on test data
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 Model Performance Metrics:")
    print(f"   Mean Squared Error: {mse:.4f}")
    print(f"   R² Score: {r2:.4f}")
    
    # Interpret R² score
    if r2 > 0.7:
        print("   🎉 Excellent model performance!")
    elif r2 > 0.5:
        print("   👍 Good model performance!")
    elif r2 > 0.3:
        print("   ⚠️  Moderate model performance")
    else:
        print("   ⚠️  Model performance needs improvement")


def train_toxicity_model():
    """
    Complete training pipeline - loads data, trains model, and saves it.
    
    Returns:
        tuple: (model, vectorizer) - Trained model and vectorizer
    """
    print("=" * SEPARATOR_LENGTH)
    print("🧠 TOXICITY DETECTION MODEL TRAINING")
    print("=" * SEPARATOR_LENGTH)
    print("This may take 5-10 minutes - but will make future runs instant!")
    
    # Step 1: Load and prepare data
    X, y = load_training_data()
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_training_data(X, y)
    
    # Step 3: Create TF-IDF vectors
    X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_vectors(X_train, X_test)
    
    # Step 4: Train model
    model = train_linear_regression_model(X_train_tfidf, y_train)
    
    # Step 5: Evaluate model
    evaluate_model(model, X_test_tfidf, y_test)
    
    # Step 6: Save model and vectorizer
    save_model_and_vectorizer(model, vectorizer)
    
    print("\n" + "=" * SEPARATOR_LENGTH)
    print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * SEPARATOR_LENGTH)
    
    return model, vectorizer


def main():
    """
    Main training script entry point.
    Checks if model exists and offers to retrain if needed.
    """
    print("🤖 TOXICITY DETECTION - MODEL TRAINING")
    print("-" * 50)
    
    if models_exist():
        print("⚠️  Trained model already exists!")
        response = input("Do you want to retrain the model? (y/N): ").lower().strip()
        
        if response not in ['y', 'yes']:
            print("✅ Using existing model. No training needed.")
            return
        
        print("🔄 Retraining model as requested...")
    else:
        print("🆕 No trained model found. Starting training...")
    
    try:
        train_toxicity_model()
        print("\n✅ Training completed! You can now use main.py for predictions.")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()