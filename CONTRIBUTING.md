# Contributing to Toxicity Detection System

Thank you for your interest in contributing to the Toxicity Detection System! This guide will help you get started with development and contributing to the project.

## Development Setup

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/binkowskidawid/comment-toxicity-prediction.git
   cd comment-toxicity-prediction
   ```

2. **Install dependencies**
   ```bash
   make install-dev
   # or manually:
   uv sync --dev
   ```

3. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   make setup
   # or manually:
   uv run pre-commit install
   ```

## Development Workflow

### Running Tests

```bash
# Run interactive testing
make test

# Run specific functionality tests
uv run toxicity-detector  # Test interactive analyzer
uv run test-toxicity-model  # Test interactive testing system
uv run train-toxicity-model  # Test training functionality
```

### Code Quality

```bash
# Available development commands
make install-dev    # Install development dependencies
make clean          # Clean up build artifacts and cache
make build          # Build the package
make setup          # Setup development environment
```

### Training and Testing the Model

```bash
# Train a new model
make train
# or
uv run train-toxicity-model

# Start interactive analyzer
make analyze
# or
uv run toxicity-detector
```

## Project Structure

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

## Contributing Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) Python style guidelines
- Write clean, readable code with descriptive variable names
- Use type hints where appropriate
- Keep functions focused and modular

### Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use descriptive test names that explain what is being tested
- Place tests in the appropriate test file or create new ones as needed

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 50 characters
- Provide additional details in the body if needed

Example:
```
Add batch processing support for comment analysis

- Implement batch_analyze_comments function
- Add tests for batch processing
- Update documentation with usage examples
```

### Pull Requests

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**
   ```bash
   make test  # Test interactive functionality
   make train # Verify training works
   make analyze # Test interactive analyzer
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages (if any)

### Feature Requests

For new features, please provide:
- Clear description of the proposed feature
- Use cases and benefits
- Possible implementation approach
- Any relevant examples or references

### Code Contributions

Areas where contributions are welcome:
- **Model improvements**: Better algorithms, hyperparameter tuning
- **Text processing**: Enhanced preprocessing, feature engineering
- **Testing**: More comprehensive test coverage
- **Documentation**: Improved examples, tutorials
- **Performance**: Optimization and efficiency improvements
- **CLI enhancements**: Better user interface, more options

## Development Tips

### Working with Models

- Models are saved in `models/` directory as `.joblib` files
- Delete model files to force retraining during development
- Training takes 5-10 minutes but only needs to be done once

### Debugging

- Use `python -m pdb` for debugging
- Add print statements temporarily for quick debugging
- Check model files exist before loading: `make train` if needed

### Testing Locally

```bash
# Quick development cycle
make clean build test

# Test the full installation
uv build
uv sync
make train
make analyze
```

## Getting Help

- Check existing issues and documentation first
- Ask questions by creating an issue with the "question" label
- For development questions, include relevant code snippets
- Be specific about what you're trying to achieve

## Recognition

All contributors will be acknowledged in the project. Significant contributions may be recognized with maintainer status.

Thank you for contributing to the Toxicity Detection System!