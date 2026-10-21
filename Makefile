PYTHON ?= python3
SIM_ARGS ?=
ANALYZE_ARGS ?=
PYTEST_ARGS ?=

.PHONY: all simulate analyze figures clean test

all: figures

setup: .venv/.deps-installed

.venv:
	$(PYTHON) -m venv .venv

# Install (or update) dependencies whenever requirements.txt changes. The stamp
# file records the last successful install so `make` skips reinstalling when
# nothing has changed.
.venv/.deps-installed: requirements.txt | .venv
	.venv/bin/python -m pip install --quiet --upgrade pip
	.venv/bin/python -m pip install --quiet -r requirements.txt
	touch .venv/.deps-installed

simulate: setup
	.venv/bin/python src/simulation.py

# analyze: simulate
# 	.venv/bin/python src.metrics

# figures: analyze
# 	.venv/bin/python src/metrics

test: setup
	.venv/bin/python -m pytest $(PYTEST_ARGS)

clean:
	rm -rf results/raw results/figures
	rm -f results/summary.csv results/paired.csv results/evaluation.json results/metrics_source.py
