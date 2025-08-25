"""
Interactive toxicity detection analyzer.
Main interface for real-time comment analysis using pre-trained model.
"""

import sys

from ..config import SEPARATOR_LENGTH
from ..utils.model_utils import get_model_info, load_model_and_vectorizer, models_exist
from ..utils.text_processing import (
    format_comment_results,
    get_comment_rating,
    get_labels_info,
)


def display_welcome_message():
    """
    Display welcome message and system information.
    """
    print("=" * SEPARATOR_LENGTH)
    print("🤖 INTERACTIVE TOXICITY DETECTION SYSTEM")
    print("=" * SEPARATOR_LENGTH)
    print("Analyze comments for toxicity in real-time using AI")
    print("Type 'help' for commands or 'quit' to exit")


def display_help():
    """
    Display available commands and usage information.
    """
    print("\n📖 AVAILABLE COMMANDS:")
    print("  help     - Show this help message")
    print("  info     - Display model and system information")
    print("  labels   - Show toxicity labels information")
    print("  quit     - Exit the application")
    print("\n💡 USAGE:")
    print("  Simply type any comment and press Enter to analyze it")
    print("  The system will show toxicity scores for 7 categories")
    print("\n📊 SCORE INTERPRETATION:")
    print("  0.0-0.1  = VERY LOW toxicity")
    print("  0.1-0.3  = LOW toxicity")
    print("  0.3-0.5  = MEDIUM toxicity")
    print("  0.5-0.7  = HIGH toxicity")
    print("  0.7-1.0  = VERY HIGH toxicity")


def display_model_info():
    """
    Display detailed model and system information.
    """
    model_info = get_model_info()
    labels_info = get_labels_info()

    print("\n🔍 SYSTEM INFORMATION:")
    print(
        f"Model Status: {'✅ Loaded' if model_info['model_exists'] else '❌ Not Found'}"
    )
    print(
        f"Vectorizer Status: {'✅ Loaded' if model_info['vectorizer_exists'] else '❌ Not Found'}"
    )

    if model_info["model_exists"]:
        print(f"Model File: {model_info['model_file']}")
        if "model_size_mb" in model_info:
            print(f"Model Size: {model_info['model_size_mb']} MB")

    if model_info["vectorizer_exists"]:
        print(f"Vectorizer File: {model_info['vectorizer_file']}")
        if "vectorizer_size_mb" in model_info:
            print(f"Vectorizer Size: {model_info['vectorizer_size_mb']} MB")

    print(f"Analysis Categories: {labels_info['count']}")
    print("Model Type: Linear Regression")


def display_labels_info():
    """
    Display information about toxicity analysis categories.
    """
    labels_info = get_labels_info()

    print("\n🏷️  TOXICITY ANALYSIS CATEGORIES:")
    for i, label in enumerate(labels_info["labels"], 1):
        description = labels_info["descriptions"][label]
        print(f"  {i}. {label}: {description}")


def analyze_comment(comment: str, model, vectorizer) -> bool:
    """
    Analyze a single comment and display results.

    Args:
        comment (str): Comment text to analyze
        model: Trained machine learning model
        vectorizer: Trained TF-IDF vectorizer

    Returns:
        bool: True if analysis successful, False otherwise
    """
    try:
        # Get toxicity predictions
        results = get_comment_rating(comment, model, vectorizer)

        # Format and display results
        print("\n" + "=" * 50)
        print("📊 TOXICITY ANALYSIS RESULTS")
        print("=" * 50)

        formatted_results = format_comment_results(comment, results)
        print(formatted_results)

        # Add interpretation
        main_toxicity = results[0]
        if main_toxicity >= 0.7:
            interpretation = "⚠️  WARNING: Very high toxicity detected"
        elif main_toxicity >= 0.5:
            interpretation = "⚠️  High toxicity detected"
        elif main_toxicity >= 0.3:
            interpretation = "⚠️  Moderate toxicity detected"
        elif main_toxicity >= 0.1:
            interpretation = "ℹ️  Low toxicity detected"
        else:
            interpretation = "✅ Very low toxicity - comment appears safe"

        print(f"\n🎯 INTERPRETATION: {interpretation}")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False


def interactive_comment_analyzer():
    """
    Main interactive interface for comment analysis.
    Handles user input and provides real-time toxicity analysis.
    """
    # Check if model exists
    if not models_exist():
        print("❌ ERROR: No trained model found!")
        print("\n🔧 SOLUTION:")
        print("1. Run 'python -m toxicity_detector.cli.train' to train a new model")
        print("2. This will take 5-10 minutes but only needs to be done once")
        print("3. Then run this program again")
        return False

    try:
        # Load trained model and vectorizer
        print("\n📦 Loading AI model...")
        model, vectorizer = load_model_and_vectorizer()

        display_welcome_message()

        # Main interaction loop
        while True:
            print("\n" + "-" * 60)
            user_input = input("💬 Enter comment to analyze (or command): ").strip()

            # Handle empty input
            if not user_input:
                print("ℹ️  Please enter a comment or command.")
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thank you for using Toxicity Detection System!")
                print("Stay safe online! 🛡️")
                break

            elif user_input.lower() == "help":
                display_help()
                continue

            elif user_input.lower() == "info":
                display_model_info()
                continue

            elif user_input.lower() == "labels":
                display_labels_info()
                continue

            # Analyze the comment
            success = analyze_comment(user_input, model, vectorizer)

            if success:
                # Ask if user wants to continue
                print(
                    "\n💡 TIP: Try different types of comments to see how the AI responds"
                )
            else:
                print("❌ Analysis failed. Please try again.")

        return True

    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
        return True

    except Exception as e:
        print(f"❌ System error: {e}")
        print("Please check your model files and try again.")
        return False


def main():
    """
    Main application entry point.
    Initializes system and starts interactive analyzer.
    """
    try:
        success = interactive_comment_analyzer()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
