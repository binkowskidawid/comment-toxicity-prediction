"""
Testing framework for toxicity detection system.
Provides comprehensive testing of trained models with predefined and custom comments.
"""

from typing import List
from config import LABELS, SEPARATOR_LENGTH, TEST_SEPARATOR_LENGTH
from model_utils import load_model_and_vectorizer, models_exist, get_model_info
from text_processing import get_comment_rating, format_comment_results, get_labels_info


# Predefined test comments with expected behavior
TEST_COMMENTS = [
    {
        "comment": "This is a terrible comment.",
        "description": "Negative comment",
        "expected": "Medium toxicity - contains negative language",
        "emoji": "🔴"
    },
    {
        "comment": "This is a very nice comment. Thank you!",
        "description": "Positive comment", 
        "expected": "Low toxicity - contains positive language",
        "emoji": "🟢"
    },
    {
        "comment": "I want to harm you!",
        "description": "Threatening comment",
        "expected": "High toxicity - contains explicit threat",
        "emoji": "🔴"
    },
    {
        "comment": "Hello everyone, have a great day!",
        "description": "Friendly greeting",
        "expected": "Very low toxicity - polite and friendly",
        "emoji": "🟢"
    },
    {
        "comment": "You are stupid and ugly!",
        "description": "Personal insult",
        "expected": "High toxicity - direct personal attack",
        "emoji": "🔴"
    }
]


def display_labels_info():
    """
    Display information about toxicity labels analyzed by the model.
    """
    print(f"\n🏷️  Toxicity Labels Analyzed by Model:")
    labels_info = get_labels_info()
    
    for i, label in enumerate(labels_info["labels"]):
        description = labels_info["descriptions"][label]
        print(f"   {i}: {label} - {description}")
    
    print(f"\n(Values range from 0.0 (not toxic) to 1.0 (highly toxic))")


def test_single_comment(comment_data: dict, model, vectorizer):
    """
    Test model on a single comment and display results.
    
    Args:
        comment_data (dict): Comment information including text and expected behavior
        model: Trained machine learning model
        vectorizer: Trained TF-IDF vectorizer
    """
    comment = comment_data["comment"]
    description = comment_data["description"]
    expected = comment_data["expected"]
    emoji = comment_data["emoji"]
    
    print(f"\n{emoji} TEST: {description}")
    print(f"Comment: '{comment}'")
    print(f"Expected: {expected}")
    
    # Get toxicity predictions
    results = get_comment_rating(comment, model, vectorizer)
    
    # Display main toxicity score
    main_toxicity = results[0]
    print(f"Predicted main toxicity: {main_toxicity:.3f}")
    
    # Display detailed breakdown
    print("Detailed scores:")
    for i, label in enumerate(LABELS):
        score = results[i]
        level = get_toxicity_level(score)
        print(f"  {label:>16}: {score:.3f} ({level})")


def get_toxicity_level(score: float) -> str:
    """
    Convert numerical toxicity score to descriptive level.
    
    Args:
        score (float): Toxicity score from 0.0 to 1.0
        
    Returns:
        str: Descriptive toxicity level
    """
    if score >= 0.7:
        return "VERY HIGH"
    elif score >= 0.5:
        return "HIGH"
    elif score >= 0.3:
        return "MEDIUM"
    elif score >= 0.1:
        return "LOW"
    else:
        return "VERY LOW"


def test_predefined_comments():
    """
    Test model on predefined set of example comments.
    
    Returns:
        bool: True if all tests completed successfully
    """
    print("=" * SEPARATOR_LENGTH)
    print("🧪 TESTING TOXICITY DETECTION MODEL")
    print("=" * SEPARATOR_LENGTH)
    
    # Check if model exists
    if not models_exist():
        print("❌ No trained model found!")
        print("Please run 'python train_model.py' first to train a model.")
        return False
    
    try:
        # Load trained model and vectorizer
        print("📦 Loading trained model...")
        model, vectorizer = load_model_and_vectorizer()
        
        # Display model information
        model_info = get_model_info()
        if model_info['model_exists'] and model_info['vectorizer_exists']:
            print(f"✅ Model loaded successfully!")
            if 'model_size_mb' in model_info:
                print(f"   Model size: {model_info['model_size_mb']} MB")
            if 'vectorizer_size_mb' in model_info:
                print(f"   Vectorizer size: {model_info['vectorizer_size_mb']} MB")
        
        # Display labels information
        display_labels_info()
        
        print(f"\n📊 RUNNING {len(TEST_COMMENTS)} PREDEFINED TESTS")
        print("-" * TEST_SEPARATOR_LENGTH)
        
        # Test each predefined comment
        for i, comment_data in enumerate(TEST_COMMENTS, 1):
            print(f"\n[Test {i}/{len(TEST_COMMENTS)}]")
            test_single_comment(comment_data, model, vectorizer)
        
        print(f"\n" + "=" * SEPARATOR_LENGTH)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * SEPARATOR_LENGTH)
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed with error: {e}")
        return False


def test_custom_comments(comments: List[str]):
    """
    Test model on custom list of comments.
    
    Args:
        comments (List[str]): List of comments to test
        
    Returns:
        List[dict]: Test results for each comment
    """
    if not comments:
        print("⚠️  No comments provided for testing.")
        return []
    
    if not models_exist():
        print("❌ No trained model found!")
        print("Please run 'python train_model.py' first to train a model.")
        return []
    
    try:
        # Load model
        model, vectorizer = load_model_and_vectorizer()
        
        print(f"\n🧪 TESTING {len(comments)} CUSTOM COMMENTS")
        print("-" * TEST_SEPARATOR_LENGTH)
        
        results = []
        
        for i, comment in enumerate(comments, 1):
            print(f"\n[Comment {i}/{len(comments)}]")
            
            # Get predictions
            predictions = get_comment_rating(comment, model, vectorizer)
            
            # Format and display results
            formatted_result = format_comment_results(comment, predictions)
            print(formatted_result)
            
            # Store results
            result = {
                'comment': comment,
                'predictions': predictions.tolist(),
                'main_toxicity': predictions[0]
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"❌ Custom testing failed with error: {e}")
        return []


def interactive_testing():
    """
    Interactive testing mode - allows user to input custom comments.
    """
    print("\n🎯 INTERACTIVE TESTING MODE")
    print("Enter comments to analyze (type 'quit' to exit):")
    
    if not models_exist():
        print("❌ No trained model found!")
        print("Please run 'python train_model.py' first.")
        return
    
    try:
        model, vectorizer = load_model_and_vectorizer()
        
        while True:
            print("\n" + "-" * 30)
            comment = input("Enter comment: ").strip()
            
            if comment.lower() in ['quit', 'exit', 'q']:
                print("👋 Exiting interactive mode.")
                break
            
            if not comment:
                print("⚠️  Please enter a comment.")
                continue
            
            # Analyze comment
            results = get_comment_rating(comment, model, vectorizer)
            formatted_result = format_comment_results(comment, results)
            print(f"\n📊 Results:")
            print(formatted_result)
            
    except KeyboardInterrupt:
        print("\n👋 Interactive testing interrupted.")
    except Exception as e:
        print(f"❌ Interactive testing failed: {e}")


def main():
    """
    Main testing script entry point.
    Provides menu for different testing options.
    """
    print("🤖 TOXICITY DETECTION - MODEL TESTING")
    
    while True:
        print("\n" + "=" * 40)
        print("TESTING OPTIONS:")
        print("1. Run predefined tests")
        print("2. Test custom comments")
        print("3. Interactive testing mode") 
        print("4. Model information")
        print("5. Exit")
        print("=" * 40)
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == '1':
            test_predefined_comments()
            
        elif choice == '2':
            print("\nEnter comments to test (one per line, empty line to finish):")
            comments = []
            while True:
                comment = input(f"Comment {len(comments)+1}: ").strip()
                if not comment:
                    break
                comments.append(comment)
            
            if comments:
                test_custom_comments(comments)
            else:
                print("No comments entered.")
                
        elif choice == '3':
            interactive_testing()
            
        elif choice == '4':
            model_info = get_model_info()
            print(f"\n📁 MODEL INFORMATION:")
            print(f"Model exists: {model_info['model_exists']}")
            print(f"Vectorizer exists: {model_info['vectorizer_exists']}")
            if model_info['model_exists']:
                print(f"Model file: {model_info['model_file']}")
                if 'model_size_mb' in model_info:
                    print(f"Model size: {model_info['model_size_mb']} MB")
            if model_info['vectorizer_exists']:
                print(f"Vectorizer file: {model_info['vectorizer_file']}")
                if 'vectorizer_size_mb' in model_info:
                    print(f"Vectorizer size: {model_info['vectorizer_size_mb']} MB")
            
        elif choice == '5':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option. Please select 1-5.")


if __name__ == "__main__":
    main()