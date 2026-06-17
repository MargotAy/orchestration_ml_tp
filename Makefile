# ==============================================================================
# Projet de classification - Makefile
# ==============================================================================
# Seuls les targets d'INSTALLATION sont fournis. Les autres sont a completer
# au fil des TP (un `# TODO (Sx)` indique la commande attendue).
# Environnement gere par uv (Python 3.13) a partir de pyproject.toml.
# Aide : make help
# ==============================================================================

SHELL        := /bin/sh
PYTHON       := uv run python
RUN          := uv run
VENV_DIR     := .venv
PYTHONPATH   ?= .
export PYTHONPATH
API_HOST     ?= 127.0.0.1
API_PORT     ?= 8000
FRONTEND_PORT ?= 8501
MLFLOW_PORT  := 5000
C            ?= 1.0
MAX_ITER     ?= 1000
CV           ?= 5
SCORING      ?= roc_auc
N_TRIALS     ?= 30
SRC_DIRS     := src scripts tests

# Couleurs ANSI
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help \
        check-uv check-venv venv-create install sync deps-sync lock reset-env doctor \
        data train train-models train-optuna evaluate mlflow api predict-client frontend \
        docker-build docker-run docker-up docker-down \
        lint format type test check


# ==============================================================================
# Help
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)


# ==============================================================================
# Setup - Installation de l'environnement Python (uv + pyproject.toml) [FOURNI]
# ==============================================================================

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] uv n'est pas installe$(RESET)"; \
		echo "  Installation : https://docs.astral.sh/uv/"; \
		exit 1; \
	}

check-venv:
	@test -d $(VENV_DIR) || { \
		echo "$(RED)[ERREUR] Virtualenv manquant : $(VENV_DIR)$(RESET)"; \
		echo "  Lance : make install"; \
		exit 1; \
	}

venv-create: check-uv ## Cree un virtualenv vide (.venv)
	@echo "$(YELLOW)>> Creation du virtualenv...$(RESET)"
	uv venv $(VENV_DIR)
	@echo "$(GREEN)[OK] Virtualenv cree$(RESET)"

deps-sync: check-uv ## Synchronise les dependances projet + dev (uv sync)
	@echo "$(YELLOW)>> Synchronisation des dependances...$(RESET)"
	uv sync --extra dev
	@echo "$(GREEN)[OK] Dependances installees$(RESET)"

install: deps-sync ## Cree le venv et installe le projet + dev (alias)

sync: deps-sync ## Alias de deps-sync

lock: check-uv ## Genere/actualise uv.lock depuis pyproject.toml
	@echo "$(YELLOW)>> Generation du lockfile...$(RESET)"
	uv lock
	@echo "$(GREEN)[OK] uv.lock genere$(RESET)"

reset-env: check-uv ## Reinitialise l'environnement (.venv + uv.lock)
	@echo "$(YELLOW)>> Reinitialisation de l'environnement...$(RESET)"
	rm -rf $(VENV_DIR) uv.lock
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement recree$(RESET)"

doctor: check-uv check-venv ## Diagnostique l'environnement de travail
	@uv --version
	@$(PYTHON) --version
	@echo "$(GREEN)[OK] Environnement pret$(RESET)"


# ==============================================================================
# Pipeline ML
# ==============================================================================

data: check-venv ## Prepare/genere le jeu de donnees dans data/
	$(PYTHON) prepare_data.py

train: check-venv ## Entraine la baseline -> models/model.joblib
	$(PYTHON) -m src.train

train-models: check-venv ## Compare RF / XGBoost / LightGBM (GridSearchCV) (CV=.. SCORING=..)
	$(PYTHON) -m src.train_models --cv $(CV) --scoring $(SCORING)

train-optuna: check-venv ## Optimise RF / XGBoost / LightGBM avec Optuna (N_TRIALS=.. CV=..)
	$(PYTHON) -m src.train_optuna --n-trials $(N_TRIALS) --cv $(CV)

evaluate: check-venv ## Evalue le modele du registry + porte qualite
	$(PYTHON) -m src.evaluate

mlflow: check-venv ## Demarre le serveur MLflow local (port $(MLFLOW_PORT))
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port $(MLFLOW_PORT)

api: check-venv ## Lance l'API FastAPI en rechargement auto (voir API_HOST/API_PORT)
	$(RUN) uvicorn src.api:app --reload --host $(API_HOST) --port $(API_PORT)

predict-client: check-venv ## Teste l'API (/health, /predict, /model-info)
	$(PYTHON) scripts/predict_client.py --url http://$(API_HOST):$(API_PORT)

frontend: ## Lance le frontend Streamlit (voir FRONTEND_PORT, API_URL)
	# TODO (S14bis) : $(RUN) streamlit run frontend/app.py --server.port $(FRONTEND_PORT)


# ==============================================================================
# Docker  [A COMPLETER]
# ==============================================================================

docker-build: ## Construit l'image d'entrainement
	# TODO (S8) : docker build -f docker/Dockerfile.train -t mlproject-train .

docker-run: ## Lance l'entrainement en conteneur
	# TODO (S8) : docker run --rm -v "$(CURDIR)/../models:/app/models" mlproject-train

docker-up: ## Demarre la stack (mlflow, api, frontend)
	# TODO (S14) : docker compose -f docker-compose.yml up -d --build mlflow api frontend

docker-down: ## Arrete et supprime les conteneurs (conserve les volumes)
	# TODO (S14) : docker compose -f docker-compose.yml down


# ==============================================================================
# Qualite (aligne sur .github/workflows/ci.yml)
# ==============================================================================

lint: check-venv ## Verifie le style (ruff)
	$(RUN) ruff check $(SRC_DIRS)

format: check-venv ## Formate le code (ruff)
	$(RUN) ruff format $(SRC_DIRS)

type: check-venv ## Verifie les types (mypy)
	$(RUN) mypy src

test: check-venv ## Lance la suite de tests (pytest)
	$(RUN) pytest

check: lint type test ## Workflow qualite complet (lint + types + tests)