BIN = .venv/bin
PYTHON ?= python3
FAST = --set training.n_trials=3 --set evaluation.bootstrap_reps=100 --set augmentation.ctgan.epochs=3

.PHONY: setup doctor run run-fast run-ibm get-amlsim run-amlsim run-amlsim-fast test lint serve serve-amlsim ui-dev ui-build docker

setup:
	$(PYTHON) -m venv .venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev,gan]"
	cd ui && npm install

doctor:
	$(PYTHON) scripts/doctor.py

run:
	$(BIN)/aml-triage run

run-fast:
	$(BIN)/aml-triage run --set run_name=fast $(FAST)

run-ibm:
	$(BIN)/aml-triage run --profile configs/ibm_hi_small.yaml

get-amlsim:
	$(BIN)/pip install -q kagglehub && $(BIN)/python scripts/get_amlsim.py

run-amlsim:
	$(BIN)/aml-triage run --profile configs/amlsim.yaml

run-amlsim-fast:
	$(BIN)/aml-triage run --profile configs/amlsim.yaml --set run_name=amlsim-fast $(FAST)

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests

serve:
	$(BIN)/aml-triage serve

serve-amlsim:
	$(BIN)/aml-triage serve --run amlsim

ui-dev:
	cd ui && npm run dev

ui-build:
	cd ui && npm run build

docker:
	docker compose up --build
