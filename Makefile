# Bangla Multimodal Political Stance — common tasks
# Usage: make <target>.  `make help` lists everything.

PY      ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
CONFIG  ?= configs/text/banglabert.yaml
OUT     ?= experiments

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: $(BIN)/python ## Create .venv and install the package with dev extras
	$(BIN)/pip install -e ".[dev,viz,fetch]"

.PHONY: data
data: ## Mirror the Google Drive dataset into data/raw/
	$(BIN)/python -m bmpb.data.download

.PHONY: ingest
ingest: ## Normalize data/raw/ into the canonical tables in data/processed/
	$(BIN)/python -m bmpb.cli ingest

.PHONY: splits
splits: ## Rebuild stratified train/val/test splits
	$(BIN)/python -m bmpb.cli splits

.PHONY: train
train: ## Train one model: make train CONFIG=configs/multimodal/clip.yaml
	$(BIN)/python -m bmpb.cli train --config $(CONFIG) --out $(OUT)

.PHONY: evaluate
evaluate: ## Score a run's predictions: make evaluate OUT=experiments/<run>
	$(BIN)/python -m bmpb.cli evaluate --run $(OUT)

.PHONY: leaderboard
leaderboard: ## Rebuild reports/tables/leaderboard.md from all runs
	$(BIN)/python -m bmpb.cli leaderboard --runs experiments --out reports/tables

.PHONY: reproduce
reproduce: data ingest splits ## Full pipeline: fetch, normalize, split, train every config
	bash scripts/run_all.sh

.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest

.PHONY: lint
lint: ## Ruff + black --check
	$(BIN)/ruff check src tests
	$(BIN)/black --check src tests

.PHONY: format
format: ## Apply ruff --fix and black
	$(BIN)/ruff check --fix src tests
	$(BIN)/black src tests

.PHONY: clean
clean: ## Remove caches and build artifacts (keeps data/ and experiments/)
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
