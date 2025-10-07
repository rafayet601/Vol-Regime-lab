.PHONY: help venv install format lint test train plot clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create virtual environment
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

install: ## Install dependencies
	pip install -e ".[dev]"
	pre-commit install

format: ## Format code with black and isort
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

lint: ## Lint code with flake8 and mypy
	flake8 src/ scripts/ tests/
	mypy src/

test: ## Run tests
	pytest tests/ -v

train: ## Train HMM model
	python -m scripts.train_hmm

plot: ## Generate regime plots
	python -m scripts.plot_regimes

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

docker-build: ## Build Docker image
	docker build -t regime-lab .

docker-run: ## Run in Docker container
	docker run -it --rm -v $(PWD)/data:/app/data -v $(PWD)/artifacts:/app/artifacts regime-lab
