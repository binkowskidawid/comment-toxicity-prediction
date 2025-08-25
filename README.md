# Toxicity Detection System - Machine Learning Course

## 📝 Project Description

This project demonstrates how to create an automated system for detecting toxicity in online comments using machine learning techniques. The program analyzes texts and predicts toxicity levels across different categories.

**✨ FEATURES:** Modular architecture with separate training, testing, and analysis components for professional development workflow!

## 🎯 What You'll Learn

- How to load and process text datasets
- How to convert text to numbers (TF-IDF)
- How to train machine learning models
- How to evaluate model quality
- How to use models for predictions
- **How to save and load trained models (optimization)**
- **Professional Python project structure**
- **Modular programming principles**

## 🏗️ Project Structure

```
comment-toxicity-prediction/
├── src/
│   └── toxicity_detector/           # Main package
│       ├── __init__.py              # Package initialization
│       ├── config.py                # Configuration and constants
│       ├── core/                    # Core functionality
│       │   ├── __init__.py
│       │   └── training.py          # Model training logic
│       ├── utils/                   # Utility modules
│       │   ├── __init__.py
│       │   ├── model_utils.py       # Model save/load utilities
│       │   └── text_processing.py   # Text processing functions
│       └── cli/                     # Command line interfaces
│           ├── __init__.py
│           ├── main.py              # Interactive analyzer
│           ├── train.py             # Training command
│           └── test.py              # Testing command
├── models/                          # Saved model files
│   ├── model.joblib
│   └── vectorizer.joblib
├── Makefile                         # Development commands
├── LICENSE                          # MIT License
├── CONTRIBUTING.md                  # Contributor guidelines
├── README.md                        # English documentation (this file)
├── README_PL.md                     # Polish documentation
├── pyproject.toml                   # Modern Python project configuration
└── uv.lock                          # Dependency lock file
```

## 📚 Installation

### Prerequisites
- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Install with uv (Recommended)
```bash
# Install dependencies
uv sync

# For development
uv sync --dev
```

### Install with pip (Alternative)
```bash
pip install -e .
```

### Libraries used in the project:

- **datasets** - loading ready-made datasets
- **pandas** - data manipulation
- **scikit-learn** - machine learning tools
- **joblib** - saving and loading models (built into scikit-learn)
- **numpy** - numerical computations
- **scipy** - scientific computing

## 🚀 Quick Start Guide

### Step 1: Install Dependencies
```bash
uv sync
```

### Step 2: Train the Model (First Time Only)
```bash
uv run train-toxicity-model
# or use Makefile
make train
```
*This takes 5-10 minutes but only needs to be done once*

### Step 3: Start Interactive Analysis
```bash
uv run toxicity-detector
# or use Makefile
make analyze
```
*Instant startup - analyze comments in real-time!*

### Step 4: Run Interactive Tests (Optional)
```bash
uv run test-toxicity-model
# or use Makefile
make test
```

## 🛠️ Development Commands

The project includes a Makefile for common development tasks:

```bash
make install-dev    # Install development dependencies
make train          # Train the toxicity detection model
make analyze        # Start interactive comment analyzer
make test           # Run interactive testing system
make clean          # Clean up build artifacts
make build          # Build the package
make setup          # Setup development environment
make help           # Show all available commands
```

## 💾 Automatic Model Persistence

### ⚡ Execution Speed

**First Training (`train-toxicity-model`) ~5-10 minutes:**
1. 🔄 Loading data from internet
2. 🧠 Training linear regression model
3. 📊 Testing model quality
4. 💾 Automatic model saving to disk

**Subsequent Analysis (`toxicity-detector`) ~2-5 seconds:**
1. ✅ Finding saved files
2. ⚡ Lightning-fast model loading
3. 🚀 Instant analysis ready

### 📁 Model Files

The system automatically creates two files in `models/` directory:

- **`model.joblib`** - trained linear regression model
- **`vectorizer.joblib`** - trained TF-IDF vectorizer

**⚠️ Important:** Both files are needed for the system to work. Don't delete them!

### 🔄 Re-training the Model

To train a new model from scratch:
1. Delete files `model.joblib` and `vectorizer.joblib` from `models/` directory
2. Run `train-toxicity-model` or `make train` - automatically trains a new model

## 🔬 How It Works - Step by Step

The system uses **intelligent architecture** - each component has a specific purpose!

### 1. 🧠 Training Pipeline (`train-toxicity-model`)

```python
# Complete training workflow
def train_toxicity_model():
    X, y = load_training_data()                    # Load civil_comments dataset
    X_train, X_test, y_train, y_test = split_data(X, y)  # 80% train, 20% test
    vectorizer = create_vectorizer()               # Create TF-IDF processor
    X_train_tfidf = vectorizer.fit_transform(X_train)    # Convert text to numbers
    model = LinearRegression()                     # Create model
    model.fit(X_train_tfidf, y_train)            # Train model
    evaluate_model(model, X_test_tfidf, y_test)   # Test performance
    save_model_and_vectorizer(model, vectorizer)  # Save for future use
```

**What happens:**
- Downloads Civil Comments dataset from Google (real internet comments with expert ratings)
- Converts to pandas DataFrame for easier manipulation
- Splits into training (80%) and testing (20%) sets
- Creates TF-IDF vectors from text
- Trains linear regression model
- Evaluates performance and saves model

### 2. 📊 Analysis Pipeline (`toxicity-detector`)

```python
# Interactive analysis workflow
def interactive_comment_analyzer():
    model, vectorizer = load_model_and_vectorizer()  # Load pre-trained model
    while True:
        comment = input("Enter comment: ")           # Get user input
        results = get_comment_rating(comment, model, vectorizer)  # Analyze
        display_results(results)                     # Show toxicity scores
```

**What happens:**
- Loads pre-trained model instantly (no training needed)
- Processes user input through TF-IDF vectorizer
- Gets toxicity predictions for 7 categories
- Displays results with interpretations

### 3. 🧪 Testing Pipeline (`test-toxicity-model`)

```python
# Comprehensive testing workflow
def test_predefined_comments():
    model, vectorizer = load_model_and_vectorizer()  # Load model
    for test_comment in TEST_COMMENTS:              # Test predefined examples
        results = get_comment_rating(test_comment, model, vectorizer)
        analyze_results(results)                     # Compare with expectations
```

**What happens:**
- Tests model on predefined comments with known expected behavior
- Provides interactive testing mode for custom comments
- Displays detailed breakdowns of toxicity categories

## 🏷️ Toxicity Categories

The model analyzes **7 types of toxicity**:

| Category | Description |
|----------|-------------|
| `toxicity` | General toxicity level |
| `severe_toxicity` | Severe toxicity level |
| `obscene` | Obscene language |
| `threat` | Threats and intimidation |
| `insult` | Insults and personal attacks |
| `identity_attack` | Identity-based attacks |
| `sexual_explicit` | Sexually explicit content |

## 🔧 Module Documentation

### `config.py` - Configuration Management
Contains all constants, file paths, and model parameters:
```python
MODEL_FILE = "models/model.joblib"
LABELS = ["toxicity", "severe_toxicity", ...]
MAX_FEATURES = 5000
```

### `model_utils.py` - Model Management
Functions for saving, loading, and checking models:
```python
models_exist() -> bool                    # Check if models exist
save_model_and_vectorizer(model, vec)     # Save trained models
load_model_and_vectorizer() -> tuple      # Load saved models
```

### `text_processing.py` - Text Processing
TF-IDF vectorization and comment analysis:
```python
create_vectorizer() -> TfidfVectorizer    # Create TF-IDF processor
get_comment_rating(comment, model, vec)   # Analyze single comment
batch_analyze_comments(comments, ...)     # Analyze multiple comments
```

## 📊 Understanding TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** converts text to numbers:

- **TF (Term Frequency)** - how often a word appears in a document
- **IDF (Inverse Document Frequency)** - how rare a word is across all documents
- **Effect:** Important words get higher values, common words get lower values

**Example:**
- Comment: "This movie is terrible"
- TF-IDF converts to vector: [0.0, 0.3, 0.0, 0.8, 0.5, ...]
- Each position corresponds to one word in the vocabulary

## 📈 Model Performance Metrics

### Mean Squared Error (MSE)
- Measures average squared difference between predictions and actual values
- **Lower is better** - shows how far our predictions are from reality

### R² Score (Coefficient of Determination)
- Values from 0 to 1 (can be negative for very bad models)
- **Closer to 1 is better** - shows how well the model explains the data
- > 0.7 = Excellent, > 0.5 = Good, > 0.3 = Moderate

## 🧪 Testing Examples

The system automatically tests these scenarios:

### Test 1: "This is a terrible comment."
- **Expected result:** Medium toxicity
- **Reason:** Contains negative language

### Test 2: "This is a very nice comment. Thank you!"
- **Expected result:** Low toxicity
- **Reason:** Contains positive words

### Test 3: "I want to harm you!"
- **Expected result:** High toxicity
- **Reason:** Explicit threat

## 📊 Results Interpretation

Each comment receives 7 scores (one for each label):

```python
# Example result:
[0.1, 0.05, 0.02, 0.8, 0.1, 0.03, 0.01]
#  |     |     |    |    |     |     |
#  |     |     |    |    |     |     └─ sexual_explicit: 0.01 (very low)
#  |     |     |    |    |     └─ identity_attack: 0.03 (low) 
#  |     |     |    |    └─ insult: 0.1 (low)
#  |     |     |    └─ threat: 0.8 (high!) 
#  |     |     └─ obscene: 0.02 (very low)
#  |     └─ severe_toxicity: 0.05 (low)
#  └─ toxicity: 0.1 (low)
```

**Interpretation:**
- Values close to 0: low toxicity
- Values close to 1: high toxicity  
- In this example: comment has high score for "threat"

## 💡 Usage Examples

### Interactive Analysis
```bash
$ toxicity-detector
💬 Enter comment to analyze: Hello everyone!

📊 TOXICITY ANALYSIS RESULTS
Comment: 'Hello everyone!'
Overall toxicity: 0.023

Detailed breakdown:
        toxicity: 0.023 (VERY LOW)
  severe_toxicity: 0.012 (VERY LOW)
          obscene: 0.008 (VERY LOW)
           threat: 0.015 (VERY LOW)
           insult: 0.019 (VERY LOW)
   identity_attack: 0.011 (VERY LOW)
   sexual_explicit: 0.007 (VERY LOW)

🎯 INTERPRETATION: ✅ Very low toxicity - comment appears safe
```

### Batch Testing
```bash
$ uv run test-toxicity-model
# Select option 1 for predefined tests
# Select option 2 for custom comment testing
# Select option 3 for interactive mode
```

## 🎓 Machine Learning Concepts Explained

### What is Supervised Learning?
- We have input data (comments) and expected outputs (toxicity ratings)
- Model learns from examples with correct answers
- Can then predict for new, unseen data

### Why Split Data into Train/Test?
- **Overfitting** - model might "memorize" training data
- Testing on separate data shows real performance
- Like an exam - can't study from the exam questions

### Regression vs Classification?
- **Classification:** Predicts categories (spam/not spam)
- **Regression:** Predicts numbers (toxicity level 0 to 1)
- We use regression because toxicity is a continuous value

## 📖 Advanced Features

### Custom Comment Analysis
```python
from text_processing import get_comment_rating
from model_utils import load_model_and_vectorizer

model, vectorizer = load_model_and_vectorizer()
result = get_comment_rating("Your comment here", model, vectorizer)
print(f"Toxicity score: {result[0]}")
```

### Batch Processing
```python
from text_processing import batch_analyze_comments

comments = ["Comment 1", "Comment 2", "Comment 3"]
results = batch_analyze_comments(comments, model, vectorizer)
```

## 🔄 Model Improvements

### 1. Better Text Processing
```python
# Enhanced TF-IDF configuration
vectorizer = TfidfVectorizer(
    max_features=10000,      # more words
    ngram_range=(1, 2),      # use word pairs
    min_df=2,               # ignore very rare words
    stop_words='english'    # remove stop words
)
```

### 2. Alternative Models
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Random Forest - usually better than linear regression
model = RandomForestRegressor(n_estimators=100)

# Support Vector Machine
model = SVR(kernel='rbf')
```

### 3. Cross Validation
```python
from sklearn.model_selection import cross_val_score

# Test model on different data splits
scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
print(f"Average score: {scores.mean():.3f}")
```

## ⚠️ Limitations

1. **Language:** Model trained on English text
2. **Context:** May not understand sarcasm or irony
3. **Bias:** May have biases from training dataset
4. **Simple Model:** Linear regression has limitations

## 🛠️ Troubleshooting

**Problem:** "No trained model found"
- **Solution:** Run `train-toxicity-model` first

**Problem:** Program crashes or shows errors
- **Solution:** Delete `.joblib` files and retrain model

**Problem:** Strange results after code updates
- **Solution:** Delete old model files to train fresh model

**Problem:** Insufficient disk space
- **Solution:** Model files take ~50MB - check disk space

## 🔄 Next Steps

1. **Try different models:** Random Forest, Neural Networks
2. **Better preprocessing:** stemming, lemmatization
3. **More data:** use larger datasets
4. **Deep learning:** BERT, transformers
5. **Better evaluation:** more metrics, confusion matrix

## 📖 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Civil Comments Dataset](https://www.tensorflow.org/datasets/catalog/civil_comments)
- [Polish Documentation](README_PL.md)

---

## 📄 Open Source

This project is open source and available for educational and research purposes. Feel free to:

- 🔍 Study the code and machine learning techniques
- 🛠️ Modify and experiment with different models
- 📚 Use it as a learning resource for ML projects
- 🤝 Contribute improvements and bug fixes
- 📖 Share knowledge and help others learn

**Contributing:**
- Fork the repository
- Create feature branches for your changes
- Submit pull requests with clear descriptions
- Follow existing code style and documentation standards

**Educational Use:**
Perfect for learning about text classification, TF-IDF vectorization, model persistence, and professional Python project structure.

## 📞 Support

- **English Documentation:** This file
- **Polish Documentation:** [README_PL.md](README_PL.md)
- **Issues:** Check your model files and dependencies first