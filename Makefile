# Makefile for toxicity detector project
.PHONY: help install install-dev test clean train analyze

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	uv sync

install-dev:  ## Install development dependencies
	uv sync --dev

test:  ## Run interactive testing
	uv run test-toxicity-model

clean:  ## Clean up build artifacts and cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/

train:  ## Train the toxicity detection model
	uv run train-toxicity-model

analyze:  ## Start interactive toxicity analyzer
	uv run toxicity-detector

setup: install-dev  ## Setup development environment
	uv run pre-commit install

build:  ## Build the package
	uv build