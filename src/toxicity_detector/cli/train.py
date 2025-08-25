"""
Training command line interface for toxicity detection system.
Entry point for model training process.
"""

from ..core.training import train_toxicity_model
from ..utils.model_utils import models_exist


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

        if response not in ["y", "yes"]:
            print("✅ Using existing model. No training needed.")
            return

        print("🔄 Retraining model as requested...")
    else:
        print("🆕 No trained model found. Starting training...")

    try:
        train_toxicity_model()
        print("\n✅ Training completed! You can now use the main analyzer.")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()
